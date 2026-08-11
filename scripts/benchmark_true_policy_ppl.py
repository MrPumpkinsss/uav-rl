"""Benchmark PPO and baselines with true CodeLlama corpus PPL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.benchmarks.true_policy import run_true_policy_benchmark
from uav_rl.config import DataGenerationConfig, SystemConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("artifacts/models/ppo_policy_3328_local_teacher_ppo.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/ppo_true_llm_evaluation_3328_local_teacher_ppo_full.json"),
    )
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_3328_coverage.pth"),
        help="Surrogate used only to select oracle deployments.",
    )
    parser.add_argument("--test-channels", type=int, default=256)
    parser.add_argument("--test-seed", type=int, default=20260811)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--latency-reference", type=float, default=1.3077757414751234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "ppo",
            "random",
            "compute_greedy",
            "strong_link",
            "dynamic_programming",
            "four_segment_surrogate_oracle",
            "full_surrogate_oracle",
        ),
        default=(
            "ppo",
            "random",
            "compute_greedy",
            "strong_link",
            "dynamic_programming",
            "four_segment_surrogate_oracle",
            "full_surrogate_oracle",
        ),
    )
    args = parser.parse_args()
    result = run_true_policy_benchmark(
        args.policy,
        args.output,
        channel_count=args.test_channels,
        test_seed=args.test_seed,
        random_seed=args.random_seed,
        latency_reference=args.latency_reference,
        surrogate_path=args.surrogate,
        generation=DataGenerationConfig(),
        system=SystemConfig(),
        device_name=args.device,
        method_names=tuple(args.methods),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
