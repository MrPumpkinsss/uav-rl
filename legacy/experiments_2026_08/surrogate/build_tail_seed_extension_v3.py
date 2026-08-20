"""Add eight independent train seeds to each existing tail-v3 training action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig
from uav_rl.data.surrogate_dataset import collect_surrogate_samples
from uav_rl.data.tail_dataset import (
    TailSeedExtensionConfig,
    aggregate_tail_seed_extension_dataset,
    build_tail_seed_extension_plan,
)
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_manifest.json"),
    )
    parser.add_argument(
        "--development-plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_plan.json"),
    )
    parser.add_argument(
        "--development-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_tail_v3_development.jsonl"),
    )
    parser.add_argument(
        "--extension-plan",
        type=Path,
        default=Path(
            "artifacts/data/codellama_surrogate_tail_v3_seed_extension_plan.json"
        ),
    )
    parser.add_argument(
        "--extension-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_tail_v3_seed_extension.jsonl"),
    )
    parser.add_argument(
        "--ppl-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_tail_v3_seed_extension_ppl.jsonl"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/data"))
    return parser.parse_args()


def _quality_metadata(
    evaluator: TruePPLQualityEvaluator, reference: dict[str, object]
) -> dict[str, object]:
    if abs(evaluator.clean_perplexity - float(reference["clean_perplexity"])) > 1e-5:
        raise ValueError("tail seed-extension clean PPL changed")
    if evaluator.evaluated_sequences != int(reference["evaluated_sequences"]):
        raise ValueError("tail seed-extension evaluated sequence count changed")
    if evaluator.evaluated_tokens != int(reference["evaluated_tokens"]):
        raise ValueError("tail seed-extension evaluated token count changed")
    return {
        "model_id": evaluator.generation.model_id,
        "clean_perplexity": evaluator.clean_perplexity,
        "evaluated_sequences": evaluator.evaluated_sequences,
        "evaluated_tokens": evaluator.evaluated_tokens,
    }


def main() -> None:
    args = parse_args()
    config = TailSeedExtensionConfig()
    plan = build_tail_seed_extension_plan(
        development_manifest_path=args.development_manifest,
        development_plan_path=args.development_plan,
        plan_path=args.extension_plan,
        config=config,
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "stage": plan["stage"],
                    "config_fingerprint": plan["config_fingerprint"],
                    "actions": len(plan["actions"]),
                    "expected_samples": sum(
                        len(action["noise_seeds"]) for action in plan["actions"]
                    ),
                    "extension_audit": plan["extension_audit"],
                },
                indent=2,
            ),
            flush=True,
        )
        return

    development = json.loads(
        args.development_manifest.read_text(encoding="utf-8")
    )
    evaluator = TruePPLQualityEvaluator(
        DataGenerationConfig(**plan["generation"]),
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
    manifest = aggregate_tail_seed_extension_dataset(
        extension_plan=plan,
        extension_cache_path=args.extension_cache,
        development_manifest_path=args.development_manifest,
        development_plan_path=args.development_plan,
        development_cache_path=args.development_cache,
        output_directory=args.output_directory,
        quality_evaluator_metadata=_quality_metadata(
            evaluator, development["quality_evaluator"]
        ),
    )
    print(
        json.dumps(
            {
                "collection": collection,
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "train": manifest["splits"]["train"],
                "validation": manifest["splits"]["validation"],
                "isolation_audit": manifest["isolation_audit"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
