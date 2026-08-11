"""Train PPO with the PPL surrogate and compare fixed-channel baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.deployment import random_continuous_deployment
from uav_rl.evaluation import evaluate_methods
from uav_rl.rl import DeploymentEnvironment, PPOTrainer
from uav_rl.rl.environment import generate_channels
from uav_rl.rl.oracle import LocalFourSegmentQualityModel
from uav_rl.surrogate import load_surrogate
from uav_rl.wireless import collaborative_latency, sample_channel


def estimate_latency_reference(config: SystemConfig, seed: int, samples: int = 1024) -> float:
    rng = np.random.default_rng(seed)
    latencies = []
    for _ in range(samples):
        channel = sample_channel(rng, config)
        deployment = random_continuous_deployment(rng, config)
        latencies.append(collaborative_latency(deployment, channel, config).total_seconds)
    return float(np.mean(latencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_3328_coverage.pth"),
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path("artifacts/data/codellama_ppl_dataset_3328_coverage.json"),
    )
    parser.add_argument("--episodes", type=int, default=8192)
    parser.add_argument("--teacher-channels", type=int, default=8192)
    parser.add_argument("--behavior-cloning-epochs", type=int, default=60)
    parser.add_argument("--behavior-cloning-learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument(
        "--teacher-quality-dataset",
        type=Path,
        default=Path("artifacts/data/codellama_ppl_dataset_3328_coverage.npz"),
        help="Use a local four-segment k-NN teacher fitted from this PPL dataset.",
    )
    parser.add_argument("--teacher-quality-neighbors", type=int, default=8)
    parser.add_argument("--test-channels", type=int, default=256)
    parser.add_argument("--initial-policy", type=Path)
    parser.add_argument(
        "--require-ppo-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/models/ppo_policy_3328_local_teacher_ppo.pth"),
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=Path("artifacts/results/ppo_surrogate_evaluation_3328_local_teacher_ppo.json"),
    )
    args = parser.parse_args()

    system = SystemConfig()
    config = replace(
        PPOConfig(system=system),
        training_episodes=args.episodes,
        teacher_channels=args.teacher_channels,
        behavior_cloning_epochs=args.behavior_cloning_epochs,
        behavior_cloning_learning_rate=args.behavior_cloning_learning_rate,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.entropy_coefficient,
        test_channels=args.test_channels,
    )
    metadata = json.loads(args.dataset_metadata.read_text(encoding="utf-8"))
    clean_perplexity = float(metadata["clean_perplexity"])
    surrogate = load_surrogate(args.surrogate)
    latency_reference = estimate_latency_reference(system, config.seed)
    environment = DeploymentEnvironment(system, surrogate, latency_reference)
    teacher_quality_model = (
        LocalFourSegmentQualityModel.from_dataset(
            str(args.teacher_quality_dataset),
            system,
            neighbors=args.teacher_quality_neighbors,
        )
        if args.teacher_quality_dataset is not None
        else None
    )
    trainer = PPOTrainer(
        config,
        environment,
        teacher_quality_model,
        require_ppo_checkpoint=args.require_ppo_checkpoint,
    )
    if args.initial_policy is not None:
        trainer.load_initial_policy(args.initial_policy)
    history = trainer.train(args.model_output)

    test_channels = generate_channels(config.test_channels, config.test_seed, system)
    methods = evaluate_methods(
        environment,
        trainer.model,
        test_channels,
        clean_perplexity,
        random_seed=config.test_seed + 1,
    )
    result = {
        "ppo_config": {
            "seed": config.seed,
            "training_episodes": config.training_episodes,
            "teacher_channels": config.teacher_channels,
            "teacher_seed": config.teacher_seed,
            "behavior_cloning_epochs": config.behavior_cloning_epochs,
            "behavior_cloning_learning_rate": config.behavior_cloning_learning_rate,
            "hidden_dim": config.hidden_dim,
            "learning_rate": config.learning_rate,
            "entropy_coefficient": config.entropy_coefficient,
            "teacher_quality_dataset": (
                str(args.teacher_quality_dataset)
                if args.teacher_quality_dataset is not None
                else None
            ),
            "teacher_quality_neighbors": (
                args.teacher_quality_neighbors if args.teacher_quality_dataset is not None else None
            ),
            "initial_policy": (
                str(args.initial_policy) if args.initial_policy is not None else None
            ),
            "require_ppo_checkpoint": args.require_ppo_checkpoint,
            "validation_seed": config.validation_seed,
            "test_channels": config.test_channels,
            "test_seed": config.test_seed,
        },
        "latency_reference_seconds": latency_reference,
        "clean_perplexity": clean_perplexity,
        "training_history": history,
        "methods": methods,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
