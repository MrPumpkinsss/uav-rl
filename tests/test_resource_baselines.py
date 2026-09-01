"""Tests for baselines under the arbitrary-assignment resource model."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from uav_rl.config import SystemConfig
from uav_rl.resource_assignment import ResourceConstrainedConfig, validate_layerwise_deployment
from uav_rl.resource_environment import ResourceDeploymentEnvironment
from uav_rl.system_baselines import (
    edge_shard_uav_baseline,
    exact_grouped_reward_oracle,
    hexgen_inspired_search_baseline,
    lingualinked_uav_baseline,
)
from uav_rl.resource_baselines import (
    constrained_genetic_surrogate_baseline,
    coedge_adaptive_partition_baseline,
    dynamic_programming_baseline,
    dynamic_programming_proxy_cost,
    jointdnn_multi_uav_baseline,
    milp_proxy_oracle_baseline,
    neurosurgeon_best_split_baseline,
    petals_balanced_pipeline_baseline,
    pipeedge_uav_latency_baseline,
    proxy_beam_baseline,
    proxy_beam_surrogate_local_search,
    surrogate_simulated_annealing_baseline,
)

def _feasible_segment_deployments(config: ResourceConstrainedConfig) -> list[np.ndarray]:
    system = config.system
    deployments: list[np.ndarray] = []
    for segment_count in range(2, system.num_uavs + 1):
        for order in itertools.permutations(range(system.num_uavs), segment_count):
            for lengths in itertools.product(
                range(1, system.max_layers_per_uav + 1), repeat=segment_count
            ):
                if sum(lengths) != system.num_layers:
                    continue
                deployment = np.concatenate(
                    [np.full(length, uav, dtype=np.int64) for uav, length in zip(order, lengths)]
                )
                try:
                    validate_layerwise_deployment(deployment, config, channel=None)
                except ValueError:
                    continue
                deployments.append(deployment)
    return deployments


def _small_config() -> ResourceConstrainedConfig:
    system = SystemConfig(
        num_layers=4,
        num_uavs=3,
        max_layers_per_uav=2,
        compute_speed=(1.0, 1.5, 0.8),
    )
    return ResourceConstrainedConfig(
        system=system,
        uav_memory_capacity_units=(3.0, 3.0, 3.0),
        uav_energy_budget_joule=(5.0, 5.0, 5.0),
        uav_hover_energy_joule=(1.0, 1.0, 1.0),
    )


def test_dynamic_programming_resource_baseline_is_deterministic_and_feasible() -> None:
    config = _small_config()
    rng = np.random.default_rng(22)
    channels = rng.uniform(2.0, 20.0, size=(3, 3, 3))

    first = dynamic_programming_baseline(channels, config, latency_reference=1.0)
    second = dynamic_programming_baseline(channels, config, latency_reference=1.0)

    assert np.array_equal(first, second)
    for channel, deployment in zip(channels, first, strict=True):
        validate_layerwise_deployment(deployment, config, channel=channel)
        assert 1 <= int(np.sum(deployment[:-1] != deployment[1:])) <= 3


def test_dynamic_programming_resource_baseline_matches_small_bruteforce() -> None:
    config = _small_config()
    channel = np.asarray(
        [
            [20.0, 3.0, 15.0],
            [3.0, 20.0, 8.0],
            [15.0, 8.0, 20.0],
        ]
    )
    latency_reference = 0.8
    deployment = dynamic_programming_baseline(
        channel[None, ...], config, latency_reference
    )[0]
    dp_cost = dynamic_programming_proxy_cost(deployment, channel, config, latency_reference)
    brute_force_cost = min(
        dynamic_programming_proxy_cost(candidate, channel, config, latency_reference)
        for candidate in _feasible_segment_deployments(config)
    )

    assert dp_cost == pytest.approx(brute_force_cost)
class _DropSumQuality:
    def evaluate(self, drop_probabilities: np.ndarray, *, noise_seeds=None) -> np.ndarray:
        del noise_seeds
        return np.asarray(drop_probabilities, dtype=np.float32).sum(axis=1)


def test_wide_beam_and_surrogate_local_search_are_feasible_and_non_degrading() -> None:
    config = _small_config()
    channel = np.asarray(
        [
            [20.0, 3.0, 15.0],
            [3.0, 20.0, 8.0],
            [15.0, 8.0, 20.0],
        ],
        dtype=np.float32,
    )
    channels = channel[None, ...]
    latency_reference = 0.8
    environment = ResourceDeploymentEnvironment(
        config, _DropSumQuality(), latency_reference
    )

    wide = proxy_beam_baseline(channels, config, latency_reference, beam_width=8)
    improved = proxy_beam_surrogate_local_search(
        channels,
        config,
        environment,
        latency_reference,
        beam_width=8,
        rounds=2,
    )
    initial_rewards, _ = environment.evaluate(channels, wide)
    improved_rewards, _ = environment.evaluate(channels, improved)

    validate_layerwise_deployment(improved[0], config, channel=channel)
    assert improved.shape == wide.shape == (1, config.system.num_layers)
    assert float(improved_rewards[0]) >= float(initial_rewards[0]) - 1e-8

def test_paper_style_baselines_are_feasible_and_deterministic() -> None:
    config = _small_config()
    channel = np.asarray(
        [
            [20.0, 3.0, 15.0],
            [3.0, 20.0, 8.0],
            [15.0, 8.0, 20.0],
        ],
        dtype=np.float32,
    )
    channels = channel[None, ...]
    environment = ResourceDeploymentEnvironment(config, _DropSumQuality(), latency_reference=0.8)
    methods = [
        constrained_genetic_surrogate_baseline(
            channels, config, environment, population_size=8, generations=2, seed=7
        ),
        surrogate_simulated_annealing_baseline(
            channels, config, environment, steps=16, seed=8
        ),
        coedge_adaptive_partition_baseline(channels, config, latency_reference=0.8),
        neurosurgeon_best_split_baseline(channels, config, latency_reference=0.8),
        pipeedge_uav_latency_baseline(channels, config, latency_reference=0.8),
        petals_balanced_pipeline_baseline(channels, config, latency_reference=0.8),
        jointdnn_multi_uav_baseline(
            channels, config, latency_reference=0.8, time_limit_seconds=1.0
        ),
        milp_proxy_oracle_baseline(
            channels, config, latency_reference=0.8, time_limit_seconds=1.0
        ),
    ]
    for deployments in methods:
        assert deployments.shape == (1, config.system.num_layers)
        validate_layerwise_deployment(deployments[0], config, channel=channel)

def test_pipeline_baselines_use_unique_contiguous_uav_blocks() -> None:
    config = _small_config()
    rng = np.random.default_rng(91)
    channels = rng.uniform(2.0, 20.0, size=(2, 3, 3)).astype(np.float32)
    for method in (pipeedge_uav_latency_baseline, petals_balanced_pipeline_baseline):
        deployments = method(channels, config, latency_reference=0.8)
        for channel, deployment in zip(channels, deployments, strict=True):
            validate_layerwise_deployment(deployment, config, channel=channel)
            block_uavs = deployment[np.r_[True, deployment[1:] != deployment[:-1]]]
            assert len(block_uavs) == len(set(block_uavs.tolist()))


def test_recent_llm_baselines_are_feasible_reproducible_and_contiguous() -> None:
    config = _small_config()
    channels = np.asarray(
        [
            [
                [20.0, 3.0, 15.0],
                [3.0, 20.0, 8.0],
                [15.0, 8.0, 20.0],
            ]
        ],
        dtype=np.float32,
    )
    environment = ResourceDeploymentEnvironment(config, _DropSumQuality(), latency_reference=0.8)

    edge_first = edge_shard_uav_baseline(channels, config, plans_per_state=4)
    edge_second = edge_shard_uav_baseline(channels, config, plans_per_state=4)
    hex_first = hexgen_inspired_search_baseline(
        channels,
        config,
        environment,
        population_size=8,
        generations=2,
        seed=17,
    )
    hex_second = hexgen_inspired_search_baseline(
        channels,
        config,
        environment,
        population_size=8,
        generations=2,
        seed=17,
    )

    assert np.array_equal(edge_first, edge_second)
    assert np.array_equal(hex_first, hex_second)
    for deployment in (edge_first[0], hex_first[0]):
        validate_layerwise_deployment(deployment, config, channel=channels[0])
        block_uavs = deployment[np.r_[True, deployment[1:] != deployment[:-1]]]
        assert len(block_uavs) == len(set(block_uavs.tolist()))


def test_exact_grouped_oracle_matches_manual_group_enumeration() -> None:
    config = _small_config()
    channel = np.full((3, 3), 20.0, dtype=np.float32)
    environment = ResourceDeploymentEnvironment(config, _DropSumQuality(), latency_reference=0.8)
    result = exact_grouped_reward_oracle(
        channel,
        environment,
        num_groups=3,
        batch_size=5,
        max_assignments=100,
    )

    candidates = []
    for assignment in itertools.product(range(config.system.num_uavs), repeat=3):
        cuts = np.linspace(0, config.system.num_layers, 4, dtype=np.int64)
        candidate = np.empty(config.system.num_layers, dtype=np.int64)
        for group, uav in enumerate(assignment):
            candidate[cuts[group] : cuts[group + 1]] = uav
        try:
            validate_layerwise_deployment(candidate, config, channel=channel)
        except ValueError:
            continue
        candidates.append(candidate)
    deployments = np.stack(candidates)
    rewards, _ = environment.evaluate(
        np.repeat(channel[None, ...], len(deployments), axis=0), deployments
    )

    assert result.total_assignments == 27
    assert result.feasible_assignments == len(deployments)
    assert result.reward == pytest.approx(float(rewards.max()))


def test_lingualinked_uav_is_deterministic_capability_balanced_and_feasible() -> None:
    config = _small_config()
    channels = np.asarray(
        [
            [
                [20.0, 3.0, 15.0],
                [3.0, 20.0, 8.0],
                [15.0, 8.0, 20.0],
            ]
        ],
        dtype=np.float32,
    )
    first = lingualinked_uav_baseline(channels, config)
    second = lingualinked_uav_baseline(channels, config)
    assert np.array_equal(first, second)
    validate_layerwise_deployment(first[0], config, channel=channels[0])
    block_uavs = first[0][np.r_[True, first[0][1:] != first[0][:-1]]]
    assert len(block_uavs) == len(set(block_uavs.tolist()))
