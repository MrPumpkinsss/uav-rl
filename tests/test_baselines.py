"""Tests for deterministic deployment baselines."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from uav_rl.baselines import (
    dynamic_programming_baseline,
    dynamic_programming_proxy_cost,
)
from uav_rl.config import SystemConfig
from uav_rl.deployment import validate_deployment


def _small_feasible_deployments(config: SystemConfig) -> list[np.ndarray]:
    deployments: list[np.ndarray] = []
    minimum_segments = int(np.ceil(config.num_layers / config.max_layers_per_uav))
    for segment_count in range(minimum_segments, config.num_uavs + 1):
        for order in itertools.permutations(range(config.num_uavs), segment_count):
            for lengths in itertools.product(
                range(1, config.max_layers_per_uav + 1), repeat=segment_count
            ):
                if sum(lengths) == config.num_layers:
                    deployments.append(np.repeat(order, lengths).astype(np.int64))
    return deployments


def test_dynamic_programming_deployments_are_valid_and_deterministic() -> None:
    config = SystemConfig()
    rng = np.random.default_rng(123)
    channels = rng.uniform(2.0, 20.0, size=(8, config.num_uavs, config.num_uavs))

    first = dynamic_programming_baseline(channels, config, latency_reference=1.3)
    second = dynamic_programming_baseline(channels, config, latency_reference=1.3)

    assert np.array_equal(first, second)
    for deployment in first:
        validate_deployment(deployment, config)


def test_dynamic_programming_matches_brute_force_proxy_minimum() -> None:
    config = SystemConfig(
        num_layers=4,
        num_uavs=3,
        max_layers_per_uav=2,
        compute_speed=(1.0, 1.5, 0.8),
    )
    channel = np.asarray(
        [
            [20.0, 3.0, 15.0],
            [3.0, 20.0, 8.0],
            [15.0, 8.0, 20.0],
        ]
    )
    latency_reference = 0.8
    deployment = dynamic_programming_baseline(channel[None, ...], config, latency_reference)[0]
    dp_cost = dynamic_programming_proxy_cost(deployment, channel, config, latency_reference)
    brute_force_cost = min(
        dynamic_programming_proxy_cost(candidate, channel, config, latency_reference)
        for candidate in _small_feasible_deployments(config)
    )

    assert dp_cost == pytest.approx(brute_force_cost)
