"""Run an exact grouped-action oracle for a small optimality-gap diagnostic.

The command deliberately does not alter the main 32-layer action definition.
Instead, it groups adjacent layers into a configurable number of super-layers,
enumerates every UAV assignment in that reduced action set, and records the
exact optimum under the frozen surrogate reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from uav_rl.system_baselines import exact_grouped_reward_oracle
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.policy_io import resource_config_from_dict
from uav_rl.surrogate import load_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20"),
    )
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path(
            "artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/exact_grouped_oracle/grouped_8_2026-08-31"),
    )
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--channel-seed", type=int, default=20260920)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-assignments", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.channels < 1:
        raise ValueError("channels must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError("output directory is not empty; pass --allow-overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.run_dir / "best_policy.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = resource_config_from_dict(payload["resource_config"])
    run_config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    surrogate = load_surrogate(args.surrogate, device=torch.device(args.device))
    environment = ResourceDeploymentEnvironment(
        config, surrogate, float(run_config["latency_reference_seconds"])
    )
    channels = generate_resource_channels(args.channels, args.channel_seed, config)

    deployments: list[np.ndarray] = []
    rows: list[dict[str, float | int]] = []
    for index, channel in enumerate(channels):
        result = exact_grouped_reward_oracle(
            channel,
            environment,
            num_groups=args.groups,
            batch_size=args.batch_size,
            max_assignments=args.max_assignments,
        )
        deployments.append(result.deployment)
        rows.append(
            {
                "channel_index": index,
                "reward": result.reward,
                "feasible_assignments": result.feasible_assignments,
                "total_assignments": result.total_assignments,
            }
        )
        print(
            f"channel={index} reward={result.reward:.6f} "
            f"feasible={result.feasible_assignments}/{result.total_assignments}",
            flush=True,
        )

    summary = {
        "stage": "exact_grouped_surrogate_oracle",
        "interpretation": (
            "Exact only within the grouped action set; not a global oracle over all 32-layer "
            "assignments and not true-LLM evidence."
        ),
        "groups": args.groups,
        "channels": args.channels,
        "channel_seed": args.channel_seed,
        "reward_mean": float(np.mean([row["reward"] for row in rows])),
        "rows": rows,
    }
    np.save(args.output_dir / "channels.npy", channels)
    np.save(args.output_dir / "oracle_deployments.npy", np.stack(deployments))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
