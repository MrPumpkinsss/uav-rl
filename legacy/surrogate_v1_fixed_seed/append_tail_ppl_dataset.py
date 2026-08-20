"""Append analytically selected packet-loss tail samples to a PPL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.data.ppl_dataset import append_tail_ppl_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/codellama_ppl_dataset_1024.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/codellama_ppl_dataset_1280_tail.npz"),
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    metadata = append_tail_ppl_samples(
        args.dataset,
        args.output,
        DataGenerationConfig(num_samples=args.samples, seed=args.seed),
        SystemConfig(),
        candidate_pool_size=args.candidate_pool,
        device_name=args.device,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
