"""Collect resumable tail-v3 train/validation data or the gated final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.data.surrogate_dataset import collect_surrogate_samples
from uav_rl.data.tail_dataset import (
    TailDatasetConfig,
    aggregate_tail_development_dataset,
    aggregate_tail_final_test,
    build_tail_development_plan,
    build_tail_final_test_plan,
)
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("development", "final-test"), default="development"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selector-device", default="cpu")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_manifest.json"),
    )
    parser.add_argument(
        "--base-selector",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_multiseed_ensemble_v2.pth"),
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_manifest.json"),
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_validation.json"),
    )
    parser.add_argument(
        "--selected-checkpoint",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_tail_ensemble_v3_selected.pth"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/data"))
    return parser.parse_args()


def _quality_metadata(
    evaluator: TruePPLQualityEvaluator, reference: dict[str, object]
) -> dict[str, object]:
    clean_reference = float(reference["clean_perplexity"])
    if abs(evaluator.clean_perplexity - clean_reference) > 1e-5:
        raise ValueError("tail-v3 clean PPL does not match the v2 reference")
    if evaluator.evaluated_sequences != int(reference["evaluated_sequences"]):
        raise ValueError("tail-v3 evaluated sequence count changed")
    if evaluator.evaluated_tokens != int(reference["evaluated_tokens"]):
        raise ValueError("tail-v3 evaluated token count changed")
    return {
        "model_id": evaluator.generation.model_id,
        "clean_perplexity": evaluator.clean_perplexity,
        "evaluated_sequences": evaluator.evaluated_sequences,
        "evaluated_tokens": evaluator.evaluated_tokens,
    }


def main() -> None:
    args = parse_args()
    system = SystemConfig()
    config = TailDatasetConfig()
    if args.stage == "development":
        plan_path = args.output_directory / "codellama_surrogate_tail_v3_plan.json"
        sample_cache = Path("artifacts/cache/surrogate_tail_v3_development.jsonl")
        ppl_cache = Path("artifacts/cache/surrogate_tail_v3_development_ppl.jsonl")
        plan = build_tail_development_plan(
            base_manifest_path=args.base_manifest,
            selector_checkpoint=args.base_selector,
            plan_path=plan_path,
            system=system,
            config=config,
            device_name=args.selector_device,
        )
        reference_manifest = json.loads(
            args.base_manifest.read_text(encoding="utf-8")
        )
    else:
        summary = json.loads(args.validation_summary.read_text(encoding="utf-8"))
        if not summary.get("passed", False):
            raise RuntimeError("validation gate failed; refusing to generate final test")
        plan_path = args.output_directory / "codellama_surrogate_tail_v3_final_test_plan.json"
        sample_cache = Path("artifacts/cache/surrogate_tail_v3_final_test.jsonl")
        ppl_cache = Path("artifacts/cache/surrogate_tail_v3_final_test_ppl.jsonl")
        plan = build_tail_final_test_plan(
            development_manifest_path=args.development_manifest,
            selector_checkpoint=args.selected_checkpoint,
            plan_path=plan_path,
            system=system,
            config=config,
            device_name=args.selector_device,
        )
        reference_manifest = json.loads(
            args.development_manifest.read_text(encoding="utf-8")
        )

    if args.plan_only:
        counts: dict[str, int] = {}
        for action in plan["actions"]:
            source = str(action["source"])
            counts[source] = counts.get(source, 0) + 1
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "config_fingerprint": plan["config_fingerprint"],
                    "actions": len(plan["actions"]),
                    "source_counts": counts,
                    "isolation_audit": plan["isolation_audit"],
                },
                indent=2,
            ),
            flush=True,
        )
        return

    generation = DataGenerationConfig(**plan["generation"])
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        cache_path=ppl_cache,
        progress_interval=args.progress_interval,
    )
    collection = collect_surrogate_samples(
        plan=plan,
        evaluator=evaluator,
        sample_cache_path=sample_cache,
        progress_interval=args.progress_interval,
    )
    quality = _quality_metadata(evaluator, reference_manifest["quality_evaluator"])
    if args.stage == "development":
        manifest = aggregate_tail_development_dataset(
            plan=plan,
            sample_cache_path=sample_cache,
            base_manifest_path=args.base_manifest,
            output_directory=args.output_directory,
            quality_evaluator_metadata=quality,
        )
    else:
        manifest = aggregate_tail_final_test(
            plan=plan,
            sample_cache_path=sample_cache,
            development_manifest_path=args.development_manifest,
            output_directory=args.output_directory,
            quality_evaluator_metadata=quality,
        )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "collection": collection,
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "isolation_audit": manifest["isolation_audit"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
