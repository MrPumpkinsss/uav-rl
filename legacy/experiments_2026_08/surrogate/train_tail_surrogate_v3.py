"""Train three tail-v3 ensemble variants and apply the validation-only gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.surrogate_training import EnsembleTrainingConfig
from uav_rl.tail_training import TailValidationCriteria, run_tail_validation_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_train.npz"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_validation.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_manifest.json"),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path(
            "artifacts/results/ppl_surrogate_multiseed_ensemble_v2_metrics.json"
        ),
    )
    parser.add_argument(
        "--output-model-directory",
        type=Path,
        default=Path("artifacts/models/tail_v3_ablation"),
    )
    parser.add_argument(
        "--output-metrics-directory",
        type=Path,
        default=Path("artifacts/results/tail_v3_ablation"),
    )
    parser.add_argument(
        "--selected-checkpoint",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_tail_ensemble_v3_selected.pth"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_validation.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "artifacts/results/ppl_surrogate_tail_v3_validation_report.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tail_validation_ablation(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        baseline_metrics_path=args.baseline_metrics,
        output_model_directory=args.output_model_directory,
        output_metrics_directory=args.output_metrics_directory,
        selected_checkpoint_path=args.selected_checkpoint,
        summary_path=args.summary,
        report_path=args.report,
        training_config=EnsembleTrainingConfig(
            epochs=args.epochs,
            patience=args.patience,
        ),
        criteria=TailValidationCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "selected_variant": result["selected_variant"],
                "selected_checkpoint": result["selected_checkpoint"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
