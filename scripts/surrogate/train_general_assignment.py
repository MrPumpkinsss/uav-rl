'''Train and evaluate the surrogate for general paper-style UAV assignments.

This command trains a five-member ensemble on the train split, selects epochs
with the frozen validation split, then evaluates the held-out test split once.
It never starts PPO.
'''

from __future__ import annotations

import argparse
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.surrogate_training import (
    EnsembleTrainingConfig,
    SurrogateAcceptanceCriteria,
    train_and_evaluate_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument('--train', type=Path, default=Path('artifacts/data/general_assignment_train.npz'))
    parser.add_argument('--validation', type=Path, default=Path('artifacts/data/general_assignment_validation.npz'))
    parser.add_argument('--test', type=Path, default=Path('artifacts/data/general_assignment_test.npz'))
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/data/general_assignment_manifest.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/models/ppl_surrogate_general_assignment_ensemble.pth'))
    parser.add_argument('--metrics', type=Path, default=Path('artifacts/results/ppl_surrogate_general_assignment_metrics.json'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/results/ppl_surrogate_general_assignment_report.md'))
    parser.add_argument('--plots', type=Path, default=Path('artifacts/results/ppl_surrogate_general_assignment_plots'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_evaluate_ensemble(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        dataset_manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        report_path=args.report,
        plot_directory=args.plots,
        training_config=EnsembleTrainingConfig(epochs=args.epochs, patience=args.patience),
        acceptance_criteria=SurrogateAcceptanceCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(result['acceptance'], flush=True)


if __name__ == '__main__':
    main()
