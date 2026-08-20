"""Evaluate the validation-selected tail-v3 ensemble once on the fresh final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.surrogate_training import SurrogateAcceptanceCriteria
from uav_rl.tail_training import TailValidationCriteria, evaluate_tail_final_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/data/codellama_surrogate_tail_v3_final_manifest.json"
        ),
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_validation.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_tail_ensemble_v3_selected.pth"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_tail_final_test(
        final_manifest_path=args.manifest,
        validation_summary_path=args.validation_summary,
        checkpoint_path=args.checkpoint,
        metrics_path=Path(
            "artifacts/results/ppl_surrogate_tail_ensemble_v3_metrics.json"
        ),
        report_path=Path(
            "artifacts/results/ppl_surrogate_tail_ensemble_v3_report.md"
        ),
        plot_directory=Path(
            "artifacts/results/ppl_surrogate_tail_ensemble_v3_plots"
        ),
        acceptance_criteria=SurrogateAcceptanceCriteria(),
        tail_criteria=TailValidationCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "accepted": result["acceptance"]["passed"],
                "test_mae": result["test_metrics"]["mae"],
                "test_spearman": result["test_metrics"]["spearman"],
                "tail_mae": result["tail_aggregate_metrics"]["mae"],
                "tail_spearman": result["tail_aggregate_metrics"]["spearman"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not result["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
