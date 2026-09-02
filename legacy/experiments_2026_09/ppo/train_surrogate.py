"""Train or losslessly resume PPO using a frozen PPL surrogate reward only.

This command performs zero true-PPL evaluations. It writes periodic immutable
policy candidates; use ``validate_true_policy.py`` to select among them with
independent real-model validation channels and noise seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.baselines import dynamic_programming_baseline, strong_link_baseline
from uav_rl.config import DataGenerationConfig, PPOConfig, SystemConfig
from uav_rl.experiment import estimate_latency_reference
from uav_rl.rl import DeploymentEnvironment, PPOTrainer
from uav_rl.surrogate import load_surrogate


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("artifacts/runs/surrogate_ppo/next-round"),
    )
    parser.add_argument("--surrogate", type=Path,
                        default=Path("artifacts/models/ppl_surrogate_targeted_residual.pth"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--rollout-size", type=int, default=128)
    parser.add_argument(
        "--teacher",
        choices=("strong_link", "dynamic_programming", "none"),
        default="strong_link",
        help="Structured baseline used for behavior-cloning warm start.",
    )
    parser.add_argument(
        "--teacher-channels",
        type=int,
        default=4096,
        help="Independent synthetic channel contexts used for behavior cloning.",
    )
    parser.add_argument(
        "--behavior-cloning-epochs",
        type=int,
        default=30,
        help="Behavior-cloning epochs before surrogate-reward PPO fine-tuning.",
    )
    parser.add_argument(
        "--teacher-relative-reward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optimize frozen-surrogate reward improvement over the teacher.",
    )
    parser.add_argument(
        "--online-bc-coefficient",
        type=float,
        default=None,
        help="Teacher behavior-cloning anchor applied during PPO updates.",
    )
    parser.add_argument(
        "--checkpoint-interval-episodes",
        type=int,
        default=200,
        help="Export a frozen candidate exactly every N PPO episodes.",
    )
    parser.add_argument(
        "--monitor-interval-rollouts",
        type=int,
        default=4,
        help="Evaluate the inexpensive frozen-surrogate monitor every N rollouts.",
    )
    parser.add_argument("--monitor-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def teacher_action_provider(
    teacher: str,
    system: SystemConfig,
    latency_reference: float,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Return the deterministic structured teacher requested for warm start."""

    if teacher == "none":
        return None
    if teacher == "strong_link":
        return lambda channels: strong_link_baseline(channels, system)
    if teacher == "dynamic_programming":
        return lambda channels: dynamic_programming_baseline(
            channels, system, latency_reference
        )
    raise ValueError(f"unsupported behavior-cloning teacher: {teacher}")


def run_paths(run_directory: Path) -> dict[str, Path]:
    return {
        "surrogate_monitor_best": run_directory / "surrogate_monitor_best_policy.pth",
        "state": run_directory / "training_state.pth",
        "launch_config": run_directory / "run_config.json",
        "training_result": run_directory / "training.json",
        "candidates": run_directory / "candidate_policies",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_run_directory(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    if args.resume:
        if not paths["state"].is_file():
            raise FileNotFoundError(f"resume state does not exist: {paths['state']}")
        return
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(
            "run directory already contains artifacts; choose a new --run-dir or pass --resume"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if min(
        args.episodes,
        args.rollout_size,
        args.checkpoint_interval_episodes,
        args.monitor_interval_rollouts,
    ) < 1:
        raise ValueError("episodes, rollout size, and checkpoint/monitor intervals must be positive")
    if args.teacher == "none":
        if args.teacher_channels != 4096 or args.behavior_cloning_epochs != 30:
            raise ValueError("--teacher none cannot be combined with behavior-cloning options")
        teacher_channels = 0
        behavior_cloning_epochs = 0
        teacher_relative_rewards = False
        online_bc_coefficient = 0.0
    else:
        if args.teacher_channels < 1 or args.behavior_cloning_epochs < 1:
            raise ValueError("behavior-cloning channels and epochs must be positive with a teacher")
        teacher_channels = args.teacher_channels
        behavior_cloning_epochs = args.behavior_cloning_epochs
        teacher_relative_rewards = (
            True if args.teacher_relative_reward is None else args.teacher_relative_reward
        )
        online_bc_coefficient = (
            0.1 if args.online_bc_coefficient is None else args.online_bc_coefficient
        )
        if online_bc_coefficient < 0.0:
            raise ValueError("online behavior-cloning coefficient cannot be negative")
    if not args.surrogate.is_file():
        raise FileNotFoundError(f"surrogate checkpoint does not exist: {args.surrogate}")
    paths = run_paths(args.run_dir)
    prepare_run_directory(args, paths)

    system = SystemConfig()
    config = replace(
        PPOConfig(system=system),
        training_episodes=args.episodes,
        rollout_size=args.rollout_size,
        teacher_channels=teacher_channels,
        behavior_cloning_epochs=behavior_cloning_epochs,
        teacher_relative_rewards=teacher_relative_rewards,
        online_behavior_cloning_coefficient=online_bc_coefficient,
        validation_channels=args.monitor_channels,
        validation_interval=args.monitor_interval_rollouts,
    )
    latency_reference = estimate_latency_reference(system, config.seed)
    surrogate = load_surrogate(args.surrogate, device=torch.device(args.device))
    environment = DeploymentEnvironment(system, surrogate, latency_reference)
    trainer = PPOTrainer(
        config,
        environment,
        teacher_quality_model=None,
        require_ppo_checkpoint=True,
        teacher_action_provider=teacher_action_provider(
            args.teacher, system, latency_reference
        ),
    )
    generation = DataGenerationConfig()
    run_metadata = {
        "quality_backend": "frozen_surrogate_training_only",
        "surrogate_checkpoint": str(args.surrogate),
        "surrogate_checkpoint_sha256": _sha256(args.surrogate),
        "true_ppl_evaluations_during_training": 0,
        "candidate_selection": "external_true_model_validation_required",
        "behavior_cloning_teacher": args.teacher,
        "behavior_cloning_teacher_channels": teacher_channels,
        "behavior_cloning_epochs": behavior_cloning_epochs,
        "teacher_relative_rewards": teacher_relative_rewards,
        "online_behavior_cloning_coefficient": online_bc_coefficient,
        "candidate_checkpoint_interval_episodes": args.checkpoint_interval_episodes,
    }
    if not args.resume:
        _write_json(
            paths["launch_config"],
            {
                "format_version": 1,
                "purpose": "surrogate-reward PPO training only",
                "ppo_config": asdict(config),
                "latency_reference_seconds": latency_reference,
                "true_validation_generation": asdict(generation),
                "run_metadata": run_metadata,
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
        )

    print(
        f"starting_surrogate_ppo run_dir={args.run_dir} target_episodes={args.episodes} "
        f"surrogate_sha256={run_metadata['surrogate_checkpoint_sha256']}",
        flush=True,
    )
    started = time.perf_counter()
    history = trainer.train(
        paths["surrogate_monitor_best"],
        state_path=paths["state"],
        resume=args.resume,
        run_metadata=run_metadata,
        candidate_checkpoint_directory=paths["candidates"],
        candidate_checkpoint_interval_episodes=args.checkpoint_interval_episodes,
    )
    result = {
        "format_version": 1,
        "stage": "surrogate_training_only",
        "not_a_true_model_validation": True,
        "not_a_final_test": True,
        "run_directory": str(args.run_dir),
        "training_seconds": time.perf_counter() - started,
        "resumed": args.resume,
        "run_metadata": run_metadata,
        "training_state": str(paths["state"]),
        "surrogate_monitor_best": str(paths["surrogate_monitor_best"]),
        "candidate_checkpoints": [str(path) for path in sorted(paths["candidates"].glob("episode_*.pth"))],
        "training_history": history,
    }
    _write_json(paths["training_result"], result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
