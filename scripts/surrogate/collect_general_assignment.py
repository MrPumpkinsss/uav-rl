'''Build or resume the real-PPL dataset for paper-style layer-to-UAV assignment.

The command never starts PPO. It writes every completed action/noise-seed label
to JSONL before proceeding, so it can resume after interruption without
repeating true CodeLlama forwards.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig
from uav_rl.data.general_assignment_dataset import (
    GeneralAssignmentDatasetConfig,
    aggregate_general_assignment_cache,
    build_general_assignment_plan,
    collect_general_assignment_labels,
)
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--progress-interval', type=int, default=25)
    parser.add_argument(
        '--reference-manifest',
        type=Path,
        default=Path('artifacts/data/codellama_surrogate_tail_seed24_v5_manifest.json'),
    )
    parser.add_argument(
        '--legacy-train',
        type=Path,
        default=Path('artifacts/data/codellama_surrogate_tail_seed24_v5_train.npz'),
    )
    parser.add_argument(
        '--plan',
        type=Path,
        default=Path('artifacts/data/general_assignment_surrogate_plan.json'),
    )
    parser.add_argument(
        '--sample-cache',
        type=Path,
        default=Path('artifacts/cache/general_assignment_surrogate_labels.jsonl'),
    )
    parser.add_argument(
        '--ppl-cache',
        type=Path,
        default=Path('artifacts/cache/general_assignment_surrogate_ppl.jsonl'),
    )
    parser.add_argument('--output-directory', type=Path, default=Path('artifacts/data'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = json.loads(args.reference_manifest.read_text(encoding='utf-8'))
    generation = DataGenerationConfig(**reference['generation'])
    plan = build_general_assignment_plan(
        config=ResourceConstrainedConfig(),
        generation=generation,
        dataset=GeneralAssignmentDatasetConfig(),
        plan_path=args.plan,
        existing_cache_paths=(args.sample_cache,),
    )
    summary = {
        'config_fingerprint': plan['config_fingerprint'],
        'actions': {
            split: sum(action['split'] == split for action in plan['actions'])
            for split in ('train', 'validation', 'test')
        },
        'true_ppl_forwards': sum(len(action['noise_seeds']) for action in plan['actions']),
    }
    if args.plan_only:
        print(json.dumps(summary, indent=2), flush=True)
        return

    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        cache_path=args.ppl_cache,
        progress_interval=args.progress_interval,
    )
    expected = reference['quality_evaluator']
    if abs(evaluator.clean_perplexity - float(expected['clean_perplexity'])) > 1e-5:
        raise ValueError('clean PPL changed from the reference labels')
    if evaluator.evaluated_sequences != int(expected['evaluated_sequences']):
        raise ValueError('evaluated sequence count changed from the reference labels')
    if evaluator.evaluated_tokens != int(expected['evaluated_tokens']):
        raise ValueError('evaluated token count changed from the reference labels')

    collection = collect_general_assignment_labels(
        plan, evaluator, args.sample_cache, progress_interval=args.progress_interval
    )
    outputs = aggregate_general_assignment_cache(
        plan,
        args.sample_cache,
        args.output_directory,
        legacy_train_path=args.legacy_train,
        quality_metadata=evaluator.metadata(),
    )
    print(json.dumps({**summary, 'collection': collection, 'outputs': outputs}, default=str, indent=2))


if __name__ == '__main__':
    main()
