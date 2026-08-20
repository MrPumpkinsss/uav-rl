"""Train a validation-gated tail-only expert over the frozen seed16 ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.tail_expert import TailExpertTrainingConfig, train_tail_gated_surrogate
from uav_rl.tail_training import TailValidationCriteria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_seed16_train.npz"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_validation.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_seed16_manifest.json"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(
            "artifacts/models/ppl_surrogate_tail_seed16_ensemble_v3_selected.pth"
        ),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path(
            "artifacts/results/ppl_surrogate_multiseed_ensemble_v2_metrics.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_tail_gated_expert_v4.pth"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_gated_expert_v4_metrics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_gated_expert_v4_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_tail_gated_surrogate(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        baseline_metrics_path=args.baseline_metrics,
        base_checkpoint_path=args.base_checkpoint,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        report_path=args.report,
        training_config=TailExpertTrainingConfig(
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
                "checkpoint": result["checkpoint"],
                "selected_gate": result["selected_gate"],
                "overall_mae": result["validation_metrics"]["mae"],
                "tail_mae": result["tail_metrics"]["mae"],
                "tail_spearman": result["tail_metrics"]["spearman"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
