'''Build or resume the tail-only seed24 extension and aggregate its labels.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig
from uav_rl.data.surrogate_dataset import collect_surrogate_samples
from uav_rl.data.tail_seed24 import (
    TailSeed24Config,
    aggregate_tail_seed24_dataset,
    build_tail_seed24_plan,
)
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--progress-interval', type=int, default=500)
    parser.add_argument(
        '--v2-plan', type=Path,
        default=Path('artifacts/data/codellama_surrogate_multiseed_v2_plan.json'),
    )
    parser.add_argument(
        '--v3-plan', type=Path,
        default=Path('artifacts/data/codellama_surrogate_tail_v3_plan.json'),
    )
    parser.add_argument(
        '--seed16-plan', type=Path,
        default=Path(
            'artifacts/data/codellama_surrogate_tail_v3_seed_extension_plan.json'
        ),
    )
    parser.add_argument(
        '--seed16-manifest', type=Path,
        default=Path(
            'artifacts/data/codellama_surrogate_tail_v3_seed16_manifest.json'
        ),
    )
    parser.add_argument(
        '--extension-plan', type=Path,
        default=Path(
            'artifacts/data/codellama_surrogate_tail_v3_seed24_extension_plan.json'
        ),
    )
    parser.add_argument(
        '--v2-cache', type=Path,
        default=Path('artifacts/cache/surrogate_multiseed_v2.jsonl'),
    )
    parser.add_argument(
        '--v3-cache', type=Path,
        default=Path('artifacts/cache/surrogate_tail_v3_development.jsonl'),
    )
    parser.add_argument(
        '--seed16-cache', type=Path,
        default=Path('artifacts/cache/surrogate_tail_v3_seed_extension.jsonl'),
    )
    parser.add_argument(
        '--extension-cache', type=Path,
        default=Path('artifacts/cache/surrogate_tail_v3_seed24_extension.jsonl'),
    )
    parser.add_argument(
        '--ppl-cache', type=Path,
        default=Path('artifacts/cache/surrogate_tail_v3_seed24_extension_ppl.jsonl'),
    )
    parser.add_argument('--output-directory', type=Path, default=Path('artifacts/data'))
    return parser.parse_args()


def _quality_metadata(evaluator: TruePPLQualityEvaluator, reference: dict[str, object]) -> dict[str, object]:
    if abs(evaluator.clean_perplexity - float(reference['clean_perplexity'])) > 1e-5:
        raise ValueError('seed24 clean PPL changed')
    if evaluator.evaluated_sequences != int(reference['evaluated_sequences']):
        raise ValueError('seed24 evaluated sequence count changed')
    if evaluator.evaluated_tokens != int(reference['evaluated_tokens']):
        raise ValueError('seed24 evaluated token count changed')
    return {
        'model_id': evaluator.generation.model_id,
        'clean_perplexity': evaluator.clean_perplexity,
        'evaluated_sequences': evaluator.evaluated_sequences,
        'evaluated_tokens': evaluator.evaluated_tokens,
    }


def main() -> None:
    args = parse_args()
    config = TailSeed24Config()
    plan = build_tail_seed24_plan(
        v2_plan_path=args.v2_plan,
        v3_plan_path=args.v3_plan,
        seed16_plan_path=args.seed16_plan,
        seed16_manifest_path=args.seed16_manifest,
        plan_path=args.extension_plan,
        config=config,
    )
    if args.plan_only:
        print(json.dumps({'audit': plan['audit'], 'config_fingerprint': plan['config_fingerprint']}, indent=2))
        return
    parent = json.loads(args.seed16_manifest.read_text(encoding='utf-8'))
    evaluator = TruePPLQualityEvaluator(
        DataGenerationConfig(**parent['generation']),
        device_name=args.device,
        cache_path=args.ppl_cache,
        progress_interval=args.progress_interval,
    )
    collection = collect_surrogate_samples(
        plan=plan,
        evaluator=evaluator,
        sample_cache_path=args.extension_cache,
        progress_interval=args.progress_interval,
    )
    manifest = aggregate_tail_seed24_dataset(
        v2_plan_path=args.v2_plan,
        v3_plan_path=args.v3_plan,
        seed16_plan_path=args.seed16_plan,
        seed16_manifest_path=args.seed16_manifest,
        extension_plan_path=args.extension_plan,
        v2_cache_path=args.v2_cache,
        v3_cache_path=args.v3_cache,
        seed16_cache_path=args.seed16_cache,
        extension_cache_path=args.extension_cache,
        output_directory=args.output_directory,
        quality_evaluator_metadata=_quality_metadata(
            evaluator, parent['quality_evaluator']
        ),
    )
    print(json.dumps({'collection': collection, 'manifest': manifest}, indent=2), flush=True)


if __name__ == '__main__':
    main()
