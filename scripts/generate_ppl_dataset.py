"""Generate an offline CodeLlama PPL dataset for surrogate training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.data import generate_ppl_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/codellama_ppl_dataset.npz")
    )
    args = parser.parse_args()
    generation = DataGenerationConfig(num_samples=args.samples, seed=args.seed)
    metadata = generate_ppl_dataset(
        args.output, generation, SystemConfig(), device_name=args.device
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
