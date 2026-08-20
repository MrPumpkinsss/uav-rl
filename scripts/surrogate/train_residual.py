"""Train or resume the targeted tail residual surrogate on frozen validation data.

The command never loads final-test data and never starts PPO. Its state file is
written after every epoch, including optimizer and RNG state, so interruption
can resume the active ensemble member without discarding progress.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.tail_residual import TailResidualTrainingConfig, train_tail_residual_surrogate
from uav_rl.tail_training import TailValidationCriteria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--train", type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_train.npz"),
    )
    parser.add_argument(
        "--validation", type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_seed24_v5_validation.npz"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_manifest.json"),
    )
    parser.add_argument(
        "--global-validation-summary", type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_validation.json"),
    )
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_global.pth"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual.pth"),
    )
    parser.add_argument(
        "--state", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual_state.pth"),
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_residual_metrics.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_residual_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_tail_residual_surrogate(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        global_validation_summary_path=args.global_validation_summary,
        base_checkpoint_path=args.base_checkpoint,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        report_path=args.report,
        state_path=args.state,
        training_config=TailResidualTrainingConfig(
            epochs=args.epochs,
            patience=args.patience,
        ),
        criteria=TailValidationCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
