"""Generate and persist diverse Top-K PPO candidates using only the surrogate."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.diverse_topk import diverse_top_k_deployments
from uav_rl.rl.layerwise_ppo import LayerwisePPOTrainer
from uav_rl.surrogate import load_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--channel-seed", type=int, default=20260824)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--min-hamming-fraction", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_trainer(
    checkpoint_path: Path,
    surrogate_path: Path,
    device: torch.device,
) -> LayerwisePPOTrainer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ppo_payload = checkpoint["ppo_config"]
    system = SystemConfig(**ppo_payload["system"])
    ppo_config = PPOConfig(
        **{
            field.name: ppo_payload[field.name]
            for field in fields(PPOConfig)
            if field.name != "system"
        },
        system=system,
    )
    resource_payload = checkpoint["resource_config"]
    resource_config = ResourceConstrainedConfig(
        **{
            key: value for key, value in resource_payload.items() if key != "system"
        },
        system=SystemConfig(**resource_payload["system"]),
    )
    surrogate = load_surrogate(surrogate_path, device=device)
    environment = ResourceDeploymentEnvironment(resource_config, surrogate, 1.3077757414751234)
    trainer = LayerwisePPOTrainer(
        ppo_config,
        resource_config,
        environment,
        max_policy_boundaries=4,
    )
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.model.eval()
    return trainer


def main() -> None:
    args = parse_args()
    if min(args.channels, args.top_k, args.candidate_pool_size) < 1:
        raise ValueError("channels, top-k, and candidate pool size must be positive")
    if args.candidate_pool_size < args.top_k:
        raise ValueError("candidate pool size must be at least top-k")
    device = torch.device(args.device)
    trainer = _load_trainer(args.checkpoint, args.surrogate, device)
    resource_config = trainer.resource_config
    channels = generate_resource_channels(args.channels, args.channel_seed, resource_config)
    deployments, rewards, diagnostics = diverse_top_k_deployments(
        trainer,
        channels,
        k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        diversity_weight=args.diversity_weight,
        min_hamming_fraction=args.min_hamming_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        channels=channels,
        deployments=deployments,
        surrogate_rewards=rewards,
    )
    metadata = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "surrogate": str(args.surrogate),
        "channels": args.channels,
        "channel_seed": args.channel_seed,
        "top_k": args.top_k,
        "candidate_pool_size": args.candidate_pool_size,
        "diversity_weight": args.diversity_weight,
        "min_hamming_fraction": args.min_hamming_fraction,
        "diagnostics": diagnostics,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
