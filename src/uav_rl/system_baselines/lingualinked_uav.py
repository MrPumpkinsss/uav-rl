"""LinguaLinked-style capability-balanced pipeline for UAV inference.

This is a transparent adaptation of the LinguaLinked mobile-device idea to the
repository's layer-assignment action space. It first allocates contiguous layer
counts in proportion to feasible device capability, then orders the selected
UAVs by a channel-aware communication cost and performs a small local rebalance.
It does not use the learned PPL surrogate during selection.
"""

from __future__ import annotations

import itertools

import numpy as np

from uav_rl.resource_assignment import (
    ResourceConstrainedConfig,
    layerwise_latency,
    validate_layerwise_deployment,
)


def _segment_deployment(order: tuple[int, ...], lengths: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(
        [np.full(length, uav, dtype=np.int64) for uav, length in zip(order, lengths, strict=True)]
    )


def _capability(config: ResourceConstrainedConfig) -> np.ndarray:
    system = config.system
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    memory = np.asarray(config.uav_memory_capacity_units, dtype=np.float64)
    budgets = np.asarray(config.uav_energy_budget_joule, dtype=np.float64)
    hover = np.asarray(config.uav_hover_energy_joule, dtype=np.float64)
    # Capacity is the largest positive factor that can sustain additional layers.
    energy_capacity = np.maximum(budgets - hover, 1e-9)
    return np.minimum(speeds / speeds.max(), energy_capacity / energy_capacity.max()) * memory / memory.max()


def _candidate_lengths(order: tuple[int, ...], config: ResourceConstrainedConfig) -> list[tuple[int, ...]]:
    """Enumerate bounded integer loads, ordered by capability mismatch."""

    layers = config.system.num_layers
    max_segment = config.system.max_layers_per_uav
    capability = _capability(config)[list(order)]
    target = layers * capability / capability.sum()
    candidates: list[tuple[int, ...]] = []

    def visit(index: int, remaining: int, parts: list[int]) -> None:
        if index == len(order) - 1:
            if 1 <= remaining <= max_segment:
                candidates.append((*parts, remaining))
            return
        minimum_after = len(order) - index - 1
        lower = max(1, remaining - minimum_after * max_segment)
        upper = min(max_segment, remaining - minimum_after)
        for length in range(lower, upper + 1):
            visit(index + 1, remaining - length, [*parts, length])

    visit(0, layers, [])
    candidates.sort(
        key=lambda lengths: (
            sum((length - wanted) ** 2 for length, wanted in zip(lengths, target, strict=True)),
            lengths,
        )
    )
    return candidates[: min(64, len(candidates))]

def lingualinked_uav_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
) -> np.ndarray:
    """Choose a capability-balanced, channel-aware contiguous UAV pipeline."""

    values = np.asarray(channels, dtype=np.float64)
    expected = (config.system.num_uavs, config.system.num_uavs)
    if values.ndim != 3 or values.shape[1:] != expected:
        raise ValueError(f"channels must have shape (N, {expected[0]}, {expected[1]})")
    selected: list[np.ndarray] = []
    system = config.system
    for channel in values:
        feasible: list[tuple[float, tuple[int, ...], tuple[int, ...], np.ndarray]] = []
        required = int(np.ceil(system.num_layers / system.max_layers_per_uav))
        for count in range(required, min(system.num_uavs, system.num_layers) + 1):
            for order in itertools.permutations(range(system.num_uavs), count):
                for lengths in _candidate_lengths(order, config):
                    candidate = _segment_deployment(order, lengths)
                    try:
                        validate_layerwise_deployment(candidate, config, channel=channel)
                    except ValueError:
                        continue
                    latency = layerwise_latency(candidate, channel, config).total_seconds
                    feasible.append((latency, order, lengths, candidate))
        if not feasible:
            raise RuntimeError("LinguaLinked-UAV found no feasible deployment")
        selected.append(min(feasible, key=lambda item: (item[0], item[1], item[2]))[3])
    return np.stack(selected)
