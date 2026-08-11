"""Continuous layer-to-UAV deployment constraints and sampling."""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np

from uav_rl.config import SystemConfig


@cache
def _feasible_segment_lengths(
    num_layers: int, num_segments: int, max_layers_per_uav: int
) -> np.ndarray:
    combinations = [
        lengths
        for lengths in itertools.product(range(1, max_layers_per_uav + 1), repeat=num_segments)
        if sum(lengths) == num_layers
    ]
    result = np.asarray(combinations, dtype=np.int64)
    result.setflags(write=False)
    return result


def validate_deployment(deployment: np.ndarray, config: SystemConfig) -> None:
    """Raise when a deployment is incomplete, over capacity, or non-contiguous."""

    if deployment.shape != (config.num_layers,):
        raise ValueError(f"deployment must have shape ({config.num_layers},)")
    if np.any(deployment < 0) or np.any(deployment >= config.num_uavs):
        raise ValueError("deployment contains an invalid UAV index")
    for uav in range(config.num_uavs):
        positions = np.flatnonzero(deployment == uav)
        if positions.size > config.max_layers_per_uav:
            raise ValueError(f"UAV {uav} exceeds its layer capacity")
        if positions.size and positions[-1] - positions[0] + 1 != positions.size:
            raise ValueError(f"layers assigned to UAV {uav} are not contiguous")


def random_continuous_deployment(
    rng: np.random.Generator,
    config: SystemConfig,
) -> np.ndarray:
    """Sample a valid deployment without favoring a fixed UAV order."""

    minimum_segments = int(np.ceil(config.num_layers / config.max_layers_per_uav))
    num_segments = int(rng.integers(minimum_segments, config.num_uavs + 1))
    uav_order = rng.choice(config.num_uavs, size=num_segments, replace=False)

    feasible_lengths = _feasible_segment_lengths(
        config.num_layers, num_segments, config.max_layers_per_uav
    )
    if not len(feasible_lengths):
        raise RuntimeError("no feasible continuous deployment lengths")
    lengths = feasible_lengths[int(rng.integers(len(feasible_lengths)))]

    deployment = np.repeat(uav_order, lengths).astype(np.int64)
    validate_deployment(deployment, config)
    return deployment


def coverage_continuous_deployment(
    rng: np.random.Generator,
    config: SystemConfig,
) -> np.ndarray:
    """Sample a valid deployment while balancing layer-boundary coverage."""

    num_segments = config.num_uavs
    feasible_lengths = _feasible_segment_lengths(
        config.num_layers, num_segments, config.max_layers_per_uav
    )
    target_boundary = int(rng.integers(config.num_layers - 1))
    candidates = [
        lengths for lengths in feasible_lengths if target_boundary in (np.cumsum(lengths)[:-1] - 1)
    ]
    if not candidates:
        raise RuntimeError(f"no deployment contains target boundary {target_boundary}")
    lengths = candidates[int(rng.integers(len(candidates)))]
    uav_order = rng.permutation(config.num_uavs)
    deployment = np.repeat(uav_order, lengths).astype(np.int64)
    validate_deployment(deployment, config)
    return deployment


def deployment_boundaries(deployment: np.ndarray) -> np.ndarray:
    """Return layer indices whose activation crosses between two UAVs."""

    return np.flatnonzero(deployment[:-1] != deployment[1:])
