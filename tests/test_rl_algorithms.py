"""Tests for the common layerwise MDP and non-PPO RL baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest

from uav_rl.config import SystemConfig
from uav_rl.resource_assignment import ResourceConstrainedConfig, validate_layerwise_deployment
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.algorithms import A2CConfig, DQNConfig, LayerwiseA2CTrainer, LayerwiseDQNTrainer
from uav_rl.rl.layerwise_episode import LayerwiseEpisode
from uav_rl.rl.policy_io import load_layerwise_policy


class DropSumQualityEvaluator:
    """Cheap deterministic quality backend used only for trainer regression tests."""

    def evaluate(
        self, drop_probabilities: np.ndarray, *, noise_seeds: np.ndarray | None = None
    ) -> np.ndarray:
        del noise_seeds
        return np.asarray(drop_probabilities, dtype=np.float32).sum(axis=1)


def _small_config() -> ResourceConstrainedConfig:
    system = SystemConfig(
        num_layers=4,
        num_uavs=3,
        max_layers_per_uav=4,
        compute_speed=(1.0, 1.2, 0.9),
    )
    return ResourceConstrainedConfig(
        system=system,
        layer_memory_units=(1.0, 1.0, 1.0, 1.0),
        layer_compute_seconds_at_unit_speed=(0.01, 0.01, 0.01, 0.01),
        activation_mbit_by_boundary=(1.0, 1.0, 1.0),
        uav_memory_capacity_units=(4.0, 4.0, 4.0),
        uav_energy_budget_joule=(3.0, 3.0, 3.0),
        uav_hover_energy_joule=(1.0, 1.0, 1.0),
        compute_energy_coefficient=0.1,
    )


def test_layerwise_episode_builds_a_valid_boundary_limited_assignment() -> None:
    config = _small_config()
    channel = np.eye(3, dtype=np.float32)
    episode = LayerwiseEpisode(channel, config, max_boundaries=1)
    for action in (0, 1, 1, 1):
        _, mask = episode.observation_and_mask()
        assert mask[action]
        episode.step(action)
    deployment = episode.completed_deployment()
    assert np.count_nonzero(deployment[1:] != deployment[:-1]) == 1
    validate_layerwise_deployment(deployment, config)


def test_dqn_epsilon_schedule_reaches_configured_floor() -> None:
    config = _small_config()
    environment = ResourceDeploymentEnvironment(config, DropSumQualityEvaluator(), 1.0)
    trainer = LayerwiseDQNTrainer(
        DQNConfig(
            hidden_dim=16,
            rollout_size=2,
            training_episodes=4,
            replay_capacity=32,
            replay_warmup_transitions=4,
            batch_size=4,
            gradient_steps_per_rollout=1,
            target_update_interval=2,
            epsilon_decay_episodes=4,
            validation_channels=2,
            max_boundaries=2,
        ),
        config,
        environment,
    )
    assert trainer.epsilon(0) == 1.0
    assert trainer.epsilon(4) == pytest.approx(0.05)
    assert trainer.epsilon(100) == pytest.approx(0.05)


def test_a2c_and_dqn_complete_tiny_training_runs(tmp_path: Path) -> None:
    config = _small_config()
    environment = ResourceDeploymentEnvironment(config, DropSumQualityEvaluator(), 1.0)
    channels = generate_resource_channels(3, 77, config)

    a2c = LayerwiseA2CTrainer(
        A2CConfig(
            hidden_dim=16,
            rollout_size=2,
            training_episodes=4,
            validation_channels=2,
            validation_interval_rollouts=1,
            max_boundaries=2,
        ),
        config,
        environment,
    )
    a2c_path = tmp_path / "a2c.pth"
    a2c.train(a2c_path)
    assert a2c_path.is_file()
    assert a2c.deployments(channels).shape == (3, 4)
    loaded_a2c, _ = load_layerwise_policy(a2c_path, environment)
    assert loaded_a2c.deployments(channels).shape == (3, 4)

    dqn = LayerwiseDQNTrainer(
        DQNConfig(
            hidden_dim=16,
            rollout_size=2,
            training_episodes=4,
            replay_capacity=64,
            replay_warmup_transitions=4,
            batch_size=4,
            gradient_steps_per_rollout=2,
            target_update_interval=2,
            epsilon_decay_episodes=4,
            validation_channels=2,
            validation_interval_rollouts=1,
            max_boundaries=2,
        ),
        config,
        environment,
    )
    dqn_path = tmp_path / "dqn.pth"
    dqn.train(dqn_path)
    assert dqn_path.is_file()
    assert dqn.deployments(channels).shape == (3, 4)
    loaded_dqn, _ = load_layerwise_policy(dqn_path, environment)
    assert loaded_dqn.deployments(channels).shape == (3, 4)
    assert all(torch.isfinite(parameter).all() for parameter in dqn.online.parameters())
