"""Load a trained layerwise RL policy behind one deployment interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment
from uav_rl.rl.algorithms import A2CConfig, DQNConfig, LayerwiseA2CTrainer, LayerwiseDQNTrainer
from uav_rl.rl.layerwise_ppo import LayerwisePPOTrainer


class LayerwiseDeploymentPolicy(Protocol):
    """Minimal interface required by common RL evaluation scripts."""

    def deployments(self, channels: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        ...


def resource_config_from_dict(payload: dict) -> ResourceConstrainedConfig:
    """Rebuild nested frozen dataclasses stored in a checkpoint."""

    values = dict(payload)
    values["system"] = SystemConfig(**values["system"])
    return ResourceConstrainedConfig(**values)


def load_layerwise_policy(
    checkpoint_path: Path,
    environment: ResourceDeploymentEnvironment,
    *,
    policy_device: torch.device | str = "cpu",
) -> tuple[LayerwiseDeploymentPolicy, dict]:
    """Load PPO, A2C or DQN without hiding the checkpoint metadata."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resource_config = resource_config_from_dict(payload["resource_config"])
    if resource_config.to_dict() != environment.config.to_dict():
        raise ValueError("checkpoint resource configuration differs from the environment")

    algorithm = payload.get("algorithm")
    if algorithm == "a2c":
        trainer = LayerwiseA2CTrainer(
            A2CConfig(**payload["algorithm_config"]),
            resource_config,
            environment,
            device=policy_device,
        )
        trainer.model.load_state_dict(payload["model_state"])
    elif algorithm == "dqn":
        trainer = LayerwiseDQNTrainer(
            DQNConfig(**payload["algorithm_config"]),
            resource_config,
            environment,
            device=policy_device,
        )
        trainer.online.load_state_dict(payload["model_state"])
        trainer.target.load_state_dict(payload["model_state"])
    elif payload.get("policy_type") == LayerwisePPOTrainer.policy_type:
        trainer = LayerwisePPOTrainer(
            PPOConfig(**{
                **payload["ppo_config"],
                "system": SystemConfig(**payload["ppo_config"]["system"]),
            }),
            resource_config,
            environment,
            max_policy_boundaries=int(payload.get("max_policy_boundaries", 4)),
        )
        trainer.model.load_state_dict(payload["model_state"])
    else:
        raise ValueError(f"unsupported layerwise policy checkpoint: {checkpoint_path}")
    return trainer, payload
