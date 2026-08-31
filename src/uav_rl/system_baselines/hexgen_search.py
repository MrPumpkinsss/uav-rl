"""HexGen-inspired topology-aware evolutionary deployment search.

HexGen optimizes asymmetric parallel plans for heterogeneous LLM serving. This
adaptation uses the portion compatible with the repository's action space: an
ordered, non-repeating UAV pipeline with variable contiguous shard boundaries.
The frozen surrogate reward is used as fitness, so results must be labelled
surrogate-assisted offline search rather than an original HexGen reproduction.
"""

from __future__ import annotations

import numpy as np

from uav_rl.system_baselines.edge_shard_uav import edge_shard_uav_baseline
from uav_rl.resource_assignment import ResourceConstrainedConfig, validate_layerwise_deployment
from uav_rl.resource_environment import ResourceDeploymentEnvironment


def _blocks(deployment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(deployment[1:] != deployment[:-1]) + 1]
    return deployment[starts].astype(np.int64), starts[1:].astype(np.int64)


def _from_blocks(order: np.ndarray, boundaries: np.ndarray, layers: int) -> np.ndarray:
    cuts = np.r_[0, np.sort(boundaries), layers]
    deployment = np.empty(layers, dtype=np.int64)
    for index, uav in enumerate(order):
        deployment[cuts[index] : cuts[index + 1]] = int(uav)
    return deployment


def _random_pipeline(
    rng: np.random.Generator,
    config: ResourceConstrainedConfig,
    channel: np.ndarray,
    *,
    attempts: int = 512,
) -> np.ndarray | None:
    layers = config.system.num_layers
    for _ in range(attempts):
        blocks = int(rng.integers(2, min(config.system.num_uavs, layers) + 1))
        order = rng.choice(config.system.num_uavs, size=blocks, replace=False)
        boundaries = np.sort(rng.choice(np.arange(1, layers), size=blocks - 1, replace=False))
        candidate = _from_blocks(order, boundaries, layers)
        try:
            validate_layerwise_deployment(candidate, config, channel=channel)
        except ValueError:
            continue
        return candidate
    return None


def _mutate_pipeline(
    deployment: np.ndarray,
    rng: np.random.Generator,
    config: ResourceConstrainedConfig,
    channel: np.ndarray,
) -> np.ndarray:
    order, boundaries = _blocks(deployment)
    layers = config.system.num_layers
    for _ in range(32):
        next_order = order.copy()
        next_boundaries = boundaries.copy()
        operation = int(rng.integers(0, 3))
        if operation == 0 and len(next_boundaries):
            index = int(rng.integers(0, len(next_boundaries)))
            next_boundaries[index] = int(
                np.clip(next_boundaries[index] + rng.choice((-2, -1, 1, 2)), 1, layers - 1)
            )
            if len(np.unique(next_boundaries)) != len(next_boundaries):
                continue
        elif operation == 1 and len(next_order) > 1:
            first, second = rng.choice(len(next_order), size=2, replace=False)
            next_order[first], next_order[second] = next_order[second], next_order[first]
        else:
            unused = np.setdiff1d(np.arange(config.system.num_uavs), next_order)
            if len(unused):
                next_order[int(rng.integers(0, len(next_order)))] = int(rng.choice(unused))
        candidate = _from_blocks(next_order, next_boundaries, layers)
        try:
            validate_layerwise_deployment(candidate, config, channel=channel)
        except ValueError:
            continue
        return candidate
    return deployment.copy()


def hexgen_inspired_search_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    environment: ResourceDeploymentEnvironment,
    *,
    population_size: int = 48,
    generations: int = 48,
    elite_fraction: float = 0.25,
    seed: int = 20260831,
) -> np.ndarray:
    """Run reproducible constrained evolutionary search on each channel.

    The chromosome is ``(ordered unique UAVs, contiguous boundaries)``. Initial
    populations include EdgeShard-UAV and random feasible pipelines. Selection
    uses the common frozen-surrogate reward; mutation changes device order,
    device membership, or a shard boundary. Elites survive unchanged.
    """

    if population_size < 4 or generations < 1:
        raise ValueError("population_size must be >= 4 and generations must be positive")
    if not 0.0 < elite_fraction <= 0.5:
        raise ValueError("elite_fraction must be in (0, 0.5]")
    values = np.asarray(channels, dtype=np.float32)
    rng = np.random.default_rng(seed)
    edge_starts = edge_shard_uav_baseline(values, config)
    selected: list[np.ndarray] = []
    for channel, edge_start in zip(values, edge_starts, strict=True):
        population = [edge_start.copy()]
        seen = {tuple(int(value) for value in edge_start)}
        attempts = 0
        while len(population) < population_size and attempts < population_size * 100:
            attempts += 1
            candidate = _random_pipeline(rng, config, channel)
            if candidate is None:
                break
            key = tuple(int(value) for value in candidate)
            if key not in seen:
                seen.add(key)
                population.append(candidate)
        while len(population) < population_size:
            population.append(population[int(rng.integers(0, len(population)))].copy())

        population_array = np.stack(population)
        repeated_channels = np.repeat(channel[None, ...], len(population_array), axis=0)
        rewards, _ = environment.evaluate(repeated_channels, population_array)
        elite_count = max(2, int(round(population_size * elite_fraction)))
        for _ in range(generations):
            order = np.argsort(rewards)[::-1]
            elites = population_array[order[:elite_count]]
            next_population = [elite.copy() for elite in elites]
            while len(next_population) < population_size:
                parent = elites[int(rng.integers(0, len(elites)))]
                next_population.append(_mutate_pipeline(parent, rng, config, channel))
            population_array = np.stack(next_population)
            repeated_channels = np.repeat(channel[None, ...], len(population_array), axis=0)
            rewards, _ = environment.evaluate(repeated_channels, population_array)
        selected.append(population_array[int(np.argmax(rewards))].copy())
    return np.stack(selected)
