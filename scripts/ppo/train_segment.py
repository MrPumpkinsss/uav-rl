"""Train or losslessly resume independent variable-length segment PPO.

Training uses the frozen surrogate only. Real CodeLlama PPL is reserved for a
separate staged validation of the exported candidates.
"""

from __future__ import annotations

import argparse
import hashlib
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
from uav_rl.rl import DeploymentEnvironment, SegmentPPOTrainer
from uav_rl.rl.segment_ppo import SegmentPPOOptions
from uav_rl.surrogate import load_surrogate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/runs/surrogate_ppo/segment_multistep")
    )
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual.pth"),
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--rollout-size", type=int, default=128)
    parser.add_argument("--teacher-channels", type=int, default=4096)
    parser.add_argument("--behavior-cloning-epochs", type=int, default=30)
    parser.add_argument(
        "--teacher", choices=("strong_link", "dynamic_programming", "none"), default="strong_link"
    )
    parser.add_argument("--checkpoint-interval-episodes", type=int, default=200)
    parser.add_argument("--monitor-interval-rollouts", type=int, default=4)
    parser.add_argument("--monitor-channels", type=int, default=32)
    parser.add_argument(
        "--fixed-capacity-segments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Choose only the next UAV; every segment fills its configured capacity.",
    )
    parser.add_argument(
        "--positive-reference-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update only on sampled deployments that beat the training-only teacher.",
    )
    parser.add_argument("--improvement-margin", type=float, default=0.0)
    parser.add_argument("--reference-kl-coefficient", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _teacher(
    name: str, system: SystemConfig, latency_reference: float
) -> Callable[[np.ndarray], np.ndarray] | None:
    if name == "none":
        return None
    if name == "strong_link":
        return lambda channels: strong_link_baseline(channels, system)
    return lambda channels: dynamic_programming_baseline(channels, system, latency_reference)


def run_paths(directory: Path) -> dict[str, Path]:
    return {
        "surrogate_monitor_best": directory / "surrogate_monitor_best_policy.pth",
        "state": directory / "training_state.pth",
        "launch_config": directory / "run_config.json",
        "training_result": directory / "training.json",
        "candidates": directory / "candidate_policies",
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
        raise FileExistsError("run directory already contains artifacts; choose another or pass --resume")
    args.run_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if min(
        args.episodes,
        args.rollout_size,
        args.checkpoint_interval_episodes,
        args.monitor_interval_rollouts,
        args.monitor_channels,
    ) < 1:
        raise ValueError("episode, rollout, checkpoint, and monitor values must be positive")
    if args.teacher == "none":
        if args.teacher_channels != 4096 or args.behavior_cloning_epochs != 30:
            raise ValueError("--teacher none cannot be combined with behavior-cloning options")
        teacher_channels = 0
        cloning_epochs = 0
    else:
        teacher_channels = args.teacher_channels
        cloning_epochs = args.behavior_cloning_epochs
    if args.positive_reference_improvement and args.teacher == "none":
        raise ValueError("positive-reference improvement requires a structured training teacher")
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
        behavior_cloning_epochs=cloning_epochs,
        validation_channels=args.monitor_channels,
        validation_interval=args.monitor_interval_rollouts,
        entropy_coefficient=0.0 if args.positive_reference_improvement else PPOConfig().entropy_coefficient,
    )
    options = SegmentPPOOptions(
        fixed_capacity_segments=args.fixed_capacity_segments,
        positive_reference_improvement=args.positive_reference_improvement,
        improvement_margin=args.improvement_margin,
        reference_kl_coefficient=args.reference_kl_coefficient,
    )
    latency_reference = estimate_latency_reference(system, config.seed)
    surrogate = load_surrogate(args.surrogate, device=torch.device(args.device))
    environment = DeploymentEnvironment(system, surrogate, latency_reference)
    metadata = {
        "quality_backend": "frozen_surrogate_training_only",
        "surrogate_checkpoint": str(args.surrogate),
        "surrogate_checkpoint_sha256": _sha256(args.surrogate),
        "true_ppl_evaluations_during_training": 0,
        "candidate_selection": "external_true_model_validation_required",
        "behavior_cloning_teacher": args.teacher,
        "behavior_cloning_teacher_channels": teacher_channels,
        "behavior_cloning_epochs": cloning_epochs,
        "episode_termination": "all_layers_assigned",
        "candidate_checkpoint_interval_episodes": args.checkpoint_interval_episodes,
        "segment_ppo_options": asdict(options),
    }
    if not args.resume:
        _write_json(
            paths["launch_config"],
            {
                "format_version": 1,
                "purpose": "independent variable-length segment PPO, surrogate reward only",
                "ppo_config": asdict(config),
                "segment_ppo_options": asdict(options),
                "latency_reference_seconds": latency_reference,
                "true_validation_generation": asdict(DataGenerationConfig()),
                "run_metadata": metadata,
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
        )
    trainer = SegmentPPOTrainer(
        config,
        environment,
        _teacher(args.teacher, system, latency_reference),
        options=options,
    )
    print(
        f"starting_segment_ppo run_dir={args.run_dir} target_episodes={args.episodes} "
        f"surrogate_sha256={metadata['surrogate_checkpoint_sha256']}",
        flush=True,
    )
    started = time.perf_counter()
    history = trainer.train(
        paths["surrogate_monitor_best"],
        state_path=paths["state"],
        resume=args.resume,
        run_metadata=metadata,
        candidate_directory=paths["candidates"],
        candidate_interval_episodes=args.checkpoint_interval_episodes,
    )
    result = {
        "format_version": 1,
        "stage": "surrogate_training_only",
        "not_a_true_model_validation": True,
        "not_a_final_test": True,
        "training_seconds": time.perf_counter() - started,
        "resumed": args.resume,
        "run_metadata": metadata,
        "training_state": str(paths["state"]),
        "surrogate_monitor_best": str(paths["surrogate_monitor_best"]),
        "candidate_checkpoints": [str(path) for path in sorted(paths["candidates"].glob("episode_*.pth"))],
        "training_history": history,
    }
    _write_json(paths["training_result"], result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
