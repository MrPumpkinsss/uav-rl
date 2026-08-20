"""Strictly re-aggregate completed v2 samples, then train and gate the ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.data.surrogate_dataset import aggregate_surrogate_datasets
from uav_rl.surrogate_training import (
    EnsembleTrainingConfig,
    SurrogateAcceptanceCriteria,
    train_and_evaluate_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_plan.json"),
    )
    parser.add_argument(
        "--sample-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_multiseed_v2.jsonl"),
    )
    parser.add_argument(
        "--existing-ppo-cache",
        type=Path,
        default=Path("artifacts/cache/ppo_true_ppl_multiseed.jsonl"),
    )
    parser.add_argument(
        "--existing-ppo-context",
        type=Path,
        default=Path("artifacts/data/ppo_true_ppl_multiseed_training_context.npz"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/data"))
    parser.add_argument(
        "--quality-reference",
        type=Path,
        default=Path("artifacts/results/ppo_true_ppl_multiseed_evaluation.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = SystemConfig()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    quality_reference = json.loads(args.quality_reference.read_text(encoding="utf-8"))
    if quality_reference["generation"] != plan["generation"]:
        raise ValueError("quality reference generation config does not match the v2 plan")
    clean_values = []
    for line in args.sample_cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        clean_values.append(
            float(record["perplexity"]) / math.exp(float(record["log_ppl_ratio"]))
        )
    if len(clean_values) != 8192:
        raise ValueError(f"expected 8192 completed samples, found {len(clean_values)}")
    reference_metadata = quality_reference["quality_evaluator"]
    clean_perplexity = float(reference_metadata["clean_perplexity"])
    maximum_clean_deviation = max(abs(value - clean_perplexity) for value in clean_values)
    if maximum_clean_deviation > 1e-5:
        raise ValueError("sample-derived clean PPL does not match the quality reference")
    quality_metadata = {
        "model_id": reference_metadata["model_id"],
        "clean_perplexity": clean_perplexity,
        "evaluated_sequences": int(reference_metadata["evaluated_sequences"]),
        "evaluated_tokens": int(reference_metadata["evaluated_tokens"]),
        "sample_derived_clean_ppl_maximum_deviation": maximum_clean_deviation,
        "reference_path": str(args.quality_reference),
        "reference_sha256": hashlib.sha256(args.quality_reference.read_bytes()).hexdigest(),
    }
    manifest = aggregate_surrogate_datasets(
        plan=plan,
        sample_cache_path=args.sample_cache,
        existing_ppo_cache=args.existing_ppo_cache,
        existing_ppo_context=args.existing_ppo_context,
        system=system,
        output_directory=args.output_directory,
        quality_evaluator_metadata=quality_metadata,
    )
    result = train_and_evaluate_ensemble(
        train_path=args.output_directory / "codellama_surrogate_multiseed_v2_train.npz",
        validation_path=args.output_directory
        / "codellama_surrogate_multiseed_v2_validation.npz",
        test_path=args.output_directory / "codellama_surrogate_multiseed_v2_test.npz",
        dataset_manifest_path=args.output_directory
        / "codellama_surrogate_multiseed_v2_manifest.json",
        checkpoint_path=Path(
            "artifacts/models/ppl_surrogate_multiseed_ensemble_v2.pth"
        ),
        metrics_path=Path(
            "artifacts/results/ppl_surrogate_multiseed_ensemble_v2_metrics.json"
        ),
        report_path=Path(
            "artifacts/results/ppl_surrogate_multiseed_ensemble_v2_report.md"
        ),
        plot_directory=Path(
            "artifacts/results/ppl_surrogate_multiseed_ensemble_v2_plots"
        ),
        training_config=EnsembleTrainingConfig(
            epochs=args.epochs,
            patience=args.patience,
        ),
        acceptance_criteria=SurrogateAcceptanceCriteria(),
        system=system,
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    summary = {
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "acceptance": result["acceptance"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not result["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
