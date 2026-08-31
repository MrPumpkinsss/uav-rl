"""EdgeShard-style contiguous LLM partitioning adapted to a UAV network.

This is an adaptation rather than a byte-for-byte reproduction of EdgeShard.
It retains the relevant system principle: jointly choose an ordered subset of
heterogeneous devices and contiguous Transformer-layer shards with dynamic
programming. The selector minimizes a transparent analytical latency objective
and never queries the learned PPL surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_rl.resource_assignment import (
    ResourceConstrainedConfig,
    layerwise_latency,
    validate_layerwise_deployment,
)


@dataclass(frozen=True)
class _Plan:
    """One partial contiguous partition retained by the DP beam."""

    proxy_latency: float
    parts: tuple[tuple[int, int, int], ...]  # (start, end, uav)


def _communication_latency(
    boundary: int,
    sender: int,
    receiver: int,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
) -> float:
    system = config.system
    gain = float(channel[sender, receiver])
    spectral_efficiency = np.log2(
        1.0 + system.transmit_power * gain / system.noise_power
    )
    return float(
        config.activation_mbit_by_boundary[boundary]
        / (system.total_bandwidth_mhz * spectral_efficiency)
    )


def _single_channel_edge_shard(
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
    *,
    plans_per_state: int,
) -> np.ndarray:
    """Run subset DP and exact-rerank its feasible terminal candidates."""

    system = config.system
    layers = system.num_layers
    uavs = system.num_uavs
    memory = np.asarray(config.layer_memory_units, dtype=np.float64)
    compute = np.asarray(config.layer_compute_seconds_at_unit_speed, dtype=np.float64)
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    capacities = np.asarray(config.uav_memory_capacity_units, dtype=np.float64)
    budgets = np.asarray(config.uav_energy_budget_joule, dtype=np.float64)
    hover = np.asarray(config.uav_hover_energy_joule, dtype=np.float64)
    memory_prefix = np.concatenate(([0.0], np.cumsum(memory)))
    compute_prefix = np.concatenate(([0.0], np.cumsum(compute)))

    # State = (next layer, used-device mask, last device). Keeping several plans
    # per state makes terminal reranking robust to shared-bandwidth link energy,
    # which is not additive in the DP proxy.
    states: dict[tuple[int, int, int], list[_Plan]] = {(0, 0, -1): [_Plan(0.0, ())]}
    for start in range(layers):
        frontier = [(key, plans) for key, plans in states.items() if key[0] == start]
        for (_, used_mask, previous), plans in frontier:
            for uav in range(uavs):
                if used_mask & (1 << uav):
                    continue
                for end in range(start + 1, layers + 1):
                    segment_memory = memory_prefix[end] - memory_prefix[start]
                    segment_compute = compute_prefix[end] - compute_prefix[start]
                    compute_energy = (
                        config.compute_energy_coefficient * speeds[uav] ** 2 * segment_compute
                    )
                    if segment_memory > capacities[uav] + 1e-9:
                        break
                    if compute_energy + hover[uav] > budgets[uav] + 1e-9:
                        break
                    increment = segment_compute / speeds[uav]
                    if previous >= 0:
                        increment += _communication_latency(
                            start - 1, previous, uav, channel, config
                        )
                    key = (end, used_mask | (1 << uav), uav)
                    bucket = states.setdefault(key, [])
                    bucket.extend(
                        _Plan(
                            plan.proxy_latency + float(increment),
                            plan.parts + ((start, end, uav),),
                        )
                        for plan in plans
                    )
                    bucket.sort(key=lambda plan: (plan.proxy_latency, plan.parts))
                    del bucket[plans_per_state:]

    feasible: list[tuple[float, float, tuple[tuple[int, int, int], ...], np.ndarray]] = []
    for (assigned, _, _), plans in states.items():
        if assigned != layers:
            continue
        for plan in plans:
            deployment = np.empty(layers, dtype=np.int64)
            for start, end, uav in plan.parts:
                deployment[start:end] = uav
            try:
                validate_layerwise_deployment(deployment, config, channel=channel)
            except ValueError:
                continue
            exact_latency = layerwise_latency(deployment, channel, config).total_seconds
            feasible.append((exact_latency, plan.proxy_latency, plan.parts, deployment))
    if not feasible:
        raise RuntimeError("EdgeShard-UAV found no feasible contiguous deployment")
    return min(feasible, key=lambda item: (item[0], item[1], item[2]))[3]


def edge_shard_uav_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    *,
    plans_per_state: int = 8,
) -> np.ndarray:
    """Select an ordered UAV subset and contiguous shards by subset DP.

    Parameters
    ----------
    channels:
        Array shaped ``(N, U, U)`` containing channel gains.
    config:
        Shared layer, compute, memory, energy, and wireless configuration.
    plans_per_state:
        Number of low-proxy plans retained for each DP state before exact
        latency/resource reranking. Larger values improve robustness at a
        modest cost; selection remains independent of the PPL surrogate.
    """

    if plans_per_state < 1:
        raise ValueError("plans_per_state must be positive")
    values = np.asarray(channels, dtype=np.float64)
    expected = (config.system.num_uavs, config.system.num_uavs)
    if values.ndim != 3 or values.shape[1:] != expected:
        raise ValueError(f"channels must have shape (N, {expected[0]}, {expected[1]})")
    return np.stack(
        [
            _single_channel_edge_shard(channel, config, plans_per_state=plans_per_state)
            for channel in values
        ]
    )
