"""Train and evaluate the five-member multi-seed PPL surrogate ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.surrogate_training import (
    EnsembleTrainingConfig,
    SurrogateAcceptanceCriteria,
    train_and_evaluate_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_train.npz"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_validation.npz"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_test.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_multiseed_ensemble_v2.pth"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_multiseed_ensemble_v2_metrics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_multiseed_ensemble_v2_report.md"),
    )
    parser.add_argument(
        "--plot-directory",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_multiseed_ensemble_v2_plots"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_evaluate_ensemble(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        dataset_manifest_path=args.manifest,
        checkpoint_path=args.output,
        metrics_path=args.metrics,
        report_path=args.report,
        plot_directory=args.plot_directory,
        training_config=EnsembleTrainingConfig(epochs=args.epochs, patience=args.patience),
        acceptance_criteria=SurrogateAcceptanceCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(json.dumps(result["acceptance"], indent=2), flush=True)
    if not result["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
