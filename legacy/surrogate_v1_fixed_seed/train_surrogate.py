"""Train the offline PPL reward surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.surrogate import SurrogateTrainingConfig, train_surrogate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/codellama_ppl_dataset_3328_coverage.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/models/ppl_surrogate_3328_coverage.pth")
    )
    args = parser.parse_args()
    result = train_surrogate(args.dataset, args.output, SurrogateTrainingConfig())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
