"""Write post-selection diagnostics for the validation-selected tail-v3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.tail_training import run_tail_post_selection_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_manifest.json"),
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_validation.json"),
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
        default=Path("artifacts/models/ppl_surrogate_tail_ensemble_v3_selected.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_diagnostics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "artifacts/results/ppl_surrogate_tail_v3_diagnostic_report.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tail_post_selection_diagnostics(
        development_manifest_path=args.manifest,
        validation_summary_path=args.validation_summary,
        baseline_metrics_path=args.baseline_metrics,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        report_path=args.report,
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    change = result["diagnostic_v2_test_tail_mae_change"]
    print(
        json.dumps(
            {
                "validation_gate_passed": result["validation_gate_passed"],
                "selected_variant": result["selected_variant"],
                "diagnostic_tail_mae": change["tail_v3"],
                "relative_improvement": change["relative_improvement"],
                "not_a_final_acceptance_test": result[
                    "not_a_final_acceptance_test"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
