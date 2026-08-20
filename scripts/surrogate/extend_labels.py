"""Extend high-variance surrogate training labels with independent true-PPL seeds.

The current targeted plan raises coverage labels from 4 to 16 seeds and
tail-disagreement labels from 24 to 48 seeds.  It is train-only: the frozen
validation data is never sampled, reused, or overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig
from uav_rl.data.surrogate_dataset import collect_surrogate_samples
from uav_rl.data.targeted_labels import (
    TargetedLabelConfig,
    aggregate_targeted_training_data,
    build_targeted_seed_plan,
)
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--parent-manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_seed24_v5_manifest.json"),
    )
    parser.add_argument(
        "--v2-plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_plan.json"),
    )
    parser.add_argument(
        "--v3-plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_plan.json"),
    )
    parser.add_argument(
        "--seed16-plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_seed_extension_plan.json"),
    )
    parser.add_argument(
        "--seed24-plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_seed24_extension_plan.json"),
    )
    parser.add_argument(
        "--ppo-cache", type=Path, default=Path("artifacts/cache/ppo_true_ppl_multiseed.jsonl")
    )
    parser.add_argument(
        "--sample-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_targeted_seed_extension.jsonl"),
    )
    parser.add_argument(
        "--ppl-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_targeted_seed_extension_ppl.jsonl"),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("artifacts/data/surrogate_targeted_seed_plan.json"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_train.npz"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_targeted_seed_plan(
        v2_plan_path=args.v2_plan,
        v3_plan_path=args.v3_plan,
        seed16_plan_path=args.seed16_plan,
        seed24_plan_path=args.seed24_plan,
        ppo_cache_path=args.ppo_cache,
        plan_path=args.plan_output,
        config=TargetedLabelConfig(),
    )
    if args.plan_only:
        print(json.dumps({"audit": plan["audit"], "fingerprint": plan["config_fingerprint"]}, indent=2))
        return

    parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    evaluator = TruePPLQualityEvaluator(
        DataGenerationConfig(**parent_manifest["generation"]),
        device_name=args.device,
        cache_path=args.ppl_cache,
        progress_interval=args.progress_interval,
    )
    reference = parent_manifest["quality_evaluator"]
    if abs(evaluator.clean_perplexity - float(reference["clean_perplexity"])) > 1e-5:
        raise ValueError("clean PPL changed from the parent dataset")
    if evaluator.evaluated_sequences != int(reference["evaluated_sequences"]):
        raise ValueError("evaluated sequence count changed from the parent dataset")
    if evaluator.evaluated_tokens != int(reference["evaluated_tokens"]):
        raise ValueError("evaluated token count changed from the parent dataset")

    collection = collect_surrogate_samples(
        plan=plan,
        evaluator=evaluator,
        sample_cache_path=args.sample_cache,
        progress_interval=args.progress_interval,
    )
    manifest = aggregate_targeted_training_data(
        parent_manifest_path=args.parent_manifest,
        v2_plan_path=args.v2_plan,
        v3_plan_path=args.v3_plan,
        seed16_plan_path=args.seed16_plan,
        seed24_plan_path=args.seed24_plan,
        targeted_plan_path=args.plan_output,
        cache_paths=(
            Path("artifacts/cache/surrogate_multiseed_v2.jsonl"),
            Path("artifacts/cache/surrogate_tail_v3_development.jsonl"),
            Path("artifacts/cache/surrogate_tail_v3_seed_extension.jsonl"),
            Path("artifacts/cache/surrogate_tail_v3_seed24_extension.jsonl"),
            args.sample_cache,
        ),
        output_train_path=args.train_output,
        output_manifest_path=args.manifest_output,
    )
    print(json.dumps({"collection": collection, "manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
