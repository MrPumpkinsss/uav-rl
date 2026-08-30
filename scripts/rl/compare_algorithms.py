"""Train PPO, A2C and DQN under one reproducible surrogate-reward protocol.

This is the algorithmic comparison entry point.  It intentionally trains every
algorithm from scratch, with the same channel seeds, episode budget, state,
action mask, boundary freeze threshold, reward weights and held-out evaluation channels.
Production PPO results that use teacher warm-starting belong in a separate
"best system" table and must not be mixed with this controlled ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.experiment import estimate_latency_reference
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.algorithms import A2CConfig, DQNConfig, LayerwiseA2CTrainer, LayerwiseDQNTrainer
from uav_rl.rl.layerwise_ppo import LayerwisePPOTrainer
from uav_rl.surrogate import load_surrogate

ALGORITHMS = ("ppo", "a2c", "dqn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/rl_algorithm_comparison/ppo_a2c_dqn_2026-08-30"),
    )
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path(
            "artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth"
        ),
    )
    parser.add_argument("--algorithms", choices=ALGORITHMS, nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260830, 20260831, 20260902])
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--rollout-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--validation-channels", type=int, default=256)
    parser.add_argument("--evaluation-channels", type=int, default=512)
    parser.add_argument("--evaluation-seed", type=int, default=20260903)
    parser.add_argument(
        "--boundary-freeze-threshold", dest="max_boundaries", type=int, default=4
    )
    parser.add_argument("--surrogate-device", default="cuda")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evaluation(
    trainer: Any,
    environment: ResourceDeploymentEnvironment,
    channels: np.ndarray,
    *,
    boundary_freeze_threshold: int,
) -> dict[str, float]:
    deployments = trainer.deployments(channels, deterministic=True)
    rewards, details = environment.evaluate(channels, deployments)
    boundary_counts = np.count_nonzero(deployments[:, 1:] != deployments[:, :-1], axis=1)
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_standard_error": float(rewards.std(ddof=1) / np.sqrt(len(rewards))),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
        "invalid_fraction": float(details["invalid"].mean()),
        "boundary_count_mean": float(boundary_counts.mean()),
        "boundary_threshold_exceeded_fraction": float(
            np.mean(boundary_counts > boundary_freeze_threshold)
        ),
    }


def _ppo_config(args: argparse.Namespace, seed: int, system: SystemConfig) -> PPOConfig:
    return PPOConfig(
        seed=seed,
        hidden_dim=args.hidden_dim,
        rollout_size=args.rollout_size,
        training_episodes=args.episodes,
        teacher_channels=0,
        behavior_cloning_epochs=0,
        validation_channels=args.validation_channels,
        validation_seed=20260901,
        test_seed=args.evaluation_seed,
        validation_interval=4,
        system=system,
    )


def main() -> None:
    args = parse_args()
    if min(args.episodes, args.rollout_size, args.validation_channels, args.evaluation_channels) < 1:
        raise ValueError("all training and evaluation budgets must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError("output directory is not empty; pass --allow-overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    surrogate_device = torch.device(args.surrogate_device)
    policy_device = torch.device(args.policy_device)
    system = SystemConfig()
    resource_config = ResourceConstrainedConfig(system=system)
    surrogate = load_surrogate(args.surrogate, device=surrogate_device)
    latency_reference = estimate_latency_reference(system, 20260819)
    environment = ResourceDeploymentEnvironment(resource_config, surrogate, latency_reference)
    evaluation_channels = generate_resource_channels(
        args.evaluation_channels, args.evaluation_seed, resource_config
    )

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for algorithm in args.algorithms:
            run_dir = args.output_dir / algorithm / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "best_policy.pth"
            if checkpoint.exists() and not args.allow_overwrite:
                raise FileExistsError(f"checkpoint already exists: {checkpoint}")

            training_started = time.perf_counter()
            if algorithm == "ppo":
                config = _ppo_config(args, seed, system)
                trainer = LayerwisePPOTrainer(
                    config,
                    resource_config,
                    environment,
                    max_policy_boundaries=args.max_boundaries,
                )
                history = trainer.train(
                    checkpoint,
                    state_path=run_dir / "training_state.pth",
                    run_metadata={
                        "algorithm": algorithm,
                        "comparison_protocol": "from_scratch_common_budget_v1",
                    },
                    candidate_directory=run_dir / "candidate_policies",
                    candidate_interval_episodes=args.episodes,
                    resume=False,
                )
                config_payload = asdict(config)
            elif algorithm == "a2c":
                config = A2CConfig(
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    rollout_size=args.rollout_size,
                    training_episodes=args.episodes,
                    validation_channels=args.validation_channels,
                    validation_seed=20260901,
                    max_boundaries=args.max_boundaries,
                )
                trainer = LayerwiseA2CTrainer(
                    config, resource_config, environment, device=policy_device
                )
                history = trainer.train(checkpoint)
                config_payload = asdict(config)
            else:
                config = DQNConfig(
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    rollout_size=args.rollout_size,
                    training_episodes=args.episodes,
                    validation_channels=args.validation_channels,
                    validation_seed=20260901,
                    max_boundaries=args.max_boundaries,
                )
                trainer = LayerwiseDQNTrainer(
                    config, resource_config, environment, device=policy_device
                )
                history = trainer.train(checkpoint)
                config_payload = asdict(config)

            training_seconds = time.perf_counter() - training_started
            evaluation_started = time.perf_counter()
            metrics = _evaluation(
                trainer,
                environment,
                evaluation_channels,
                boundary_freeze_threshold=args.max_boundaries,
            )
            evaluation_seconds = time.perf_counter() - evaluation_started
            record = {
                "algorithm": algorithm,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "training_wall_clock_seconds": training_seconds,
                "evaluation_wall_clock_seconds": evaluation_seconds,
                **metrics,
            }
            rows.append(record)
            _write_json(run_dir / "training_history.json", history)
            _write_json(
                run_dir / "evaluation.json",
                {
                    "format_version": 1,
                    "stage": "surrogate_reward_algorithm_comparison",
                    "not_true_model_evaluation": True,
                    "algorithm": algorithm,
                    "seed": seed,
                    "algorithm_config": config_payload,
                    "resource_config": resource_config.to_dict(),
                    "surrogate": str(args.surrogate),
                    "latency_reference_seconds": latency_reference,
                    "evaluation_channels": args.evaluation_channels,
                    "evaluation_seed": args.evaluation_seed,
                    "metrics": metrics,
                },
            )

    aggregate = {}
    for algorithm in args.algorithms:
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
        aggregate[algorithm] = {
            key: {
                "mean": float(np.mean([row[key] for row in algorithm_rows])),
                "std_across_seeds": float(np.std([row[key] for row in algorithm_rows], ddof=1))
                if len(algorithm_rows) > 1
                else 0.0,
            }
            for key in (
                "reward_mean",
                "log_ppl_ratio_mean",
                "latency_mean_seconds",
                "invalid_fraction",
                "boundary_count_mean",
                "boundary_threshold_exceeded_fraction",
                "training_wall_clock_seconds",
            )
        }
    summary = {
        "format_version": 1,
        "protocol": "from_scratch_common_budget_v1",
        "important_note": (
            "This table isolates the RL update rule. It does not replace the separate "
            "best-system table containing teacher-warm-started production PPO and heuristics."
        ),
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "evaluation_channels": args.evaluation_channels,
        "evaluation_seed": args.evaluation_seed,
        "boundary_freeze_threshold": args.max_boundaries,
        "boundary_threshold_is_strict_cap": False,
        "rows": rows,
        "aggregate": aggregate,
    }
    _write_json(args.output_dir / "comparison_summary.json", summary)
    with (args.output_dir / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
