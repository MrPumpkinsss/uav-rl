"""Diagnose surrogate data coverage without true-PPL calls or model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_rl.coverage_diagnostics import diagnose_surrogate_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neighbors", type=int, default=8)
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
        "--checkpoint", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/results/surrogate_data_coverage_diagnostic.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/results/surrogate_data_coverage_diagnostic.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = diagnose_surrogate_coverage(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        report_path=args.report,
        device_name=args.device,
        neighbors=args.neighbors,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
