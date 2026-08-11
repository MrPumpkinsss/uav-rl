import numpy as np
import pytest
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.deployment import validate_deployment
from uav_rl.rl.environment import DeploymentEnvironment, generate_channels
from uav_rl.rl.oracle import (
    LocalFourSegmentQualityModel,
    continuous_deployment_candidates,
    four_segment_candidates,
    surrogate_oracle_deployments,
)
from uav_rl.surrogate import PPLSurrogate


def test_capacity_filling_candidates_are_complete_and_contiguous() -> None:
    config = SystemConfig()
    candidates = four_segment_candidates(config)

    assert candidates.shape == (120, config.num_layers)
    assert np.all(np.sum(candidates[:, 1:] != candidates[:, :-1], axis=1) == 3)
    for uav in range(config.num_uavs):
        counts = np.sum(candidates == uav, axis=1)
        assert np.all((counts == 0) | (counts == config.max_layers_per_uav))


def test_all_continuous_candidates_are_enumerated_once() -> None:
    small = SystemConfig(
        num_layers=4,
        num_uavs=3,
        max_layers_per_uav=2,
        compute_speed=(1.0, 1.5, 0.8),
    )
    candidates = continuous_deployment_candidates(small)

    assert candidates.shape == (24, small.num_layers)
    assert len(np.unique(candidates, axis=0)) == len(candidates)
    for deployment in candidates:
        validate_deployment(deployment, small)
    assert continuous_deployment_candidates(SystemConfig()).shape == (58920, 32)


def test_vectorized_surrogate_oracle_matches_environment_brute_force() -> None:
    torch.manual_seed(123)
    config = SystemConfig(
        num_layers=4,
        num_uavs=3,
        max_layers_per_uav=2,
        compute_speed=(1.0, 1.5, 0.8),
    )
    environment = DeploymentEnvironment(
        config,
        PPLSurrogate(num_boundaries=3, hidden_dim=16),
        latency_reference=0.8,
    )
    channels = generate_channels(3, seed=456, config=config)
    candidates = continuous_deployment_candidates(config)

    selected = surrogate_oracle_deployments(
        channels,
        environment,
        candidates,
        candidate_batch_size=7,
    )

    selected_rewards, _ = environment.evaluate(channels, selected)
    for channel, selected_reward in zip(channels, selected_rewards, strict=True):
        repeated_channels = np.repeat(channel[None, ...], len(candidates), axis=0)
        candidate_rewards, _ = environment.evaluate(repeated_channels, candidates)
        assert selected_reward == pytest.approx(candidate_rewards.max(), abs=1e-6)


def test_ppo_data_split_seeds_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        PPOConfig(teacher_seed=20260811)


def test_local_quality_model_reproduces_an_exact_neighbor() -> None:
    features = np.asarray([[0.05, 0.06, 0.07], [0.1, 0.11, 0.12]], dtype=np.float32)
    model = LocalFourSegmentQualityModel(
        features,
        np.asarray([0.2, 0.8], dtype=np.float32),
        np.asarray([0, 1, 2]),
        neighbors=1,
    )

    assert model.predict(features[:1]) == pytest.approx([0.2])
