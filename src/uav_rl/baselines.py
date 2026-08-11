"""Deployment baselines evaluated on the same channel realizations as PPO."""

from __future__ import annotations

import itertools

import numpy as np

from uav_rl.config import SystemConfig
from uav_rl.deployment import deployment_boundaries, random_continuous_deployment
from uav_rl.wireless import packet_drop_probability

DPState = tuple[int, int, int]


def random_baseline(channels: np.ndarray, seed: int, config: SystemConfig) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([random_continuous_deployment(rng, config) for _ in range(len(channels))])


def compute_greedy_baseline(channels: np.ndarray, config: SystemConfig) -> np.ndarray:
    """Fill the fastest UAVs to capacity, independent of instantaneous channel."""

    del channels
    order = np.argsort(-np.asarray(config.compute_speed))
    deployment = np.repeat(order, config.max_layers_per_uav)[: config.num_layers]
    return deployment.astype(np.int64)


def strong_link_baseline(channels: np.ndarray, config: SystemConfig) -> np.ndarray:
    """Choose the four-UAV path with the smallest sum of adjacent drop rates."""

    required = int(np.ceil(config.num_layers / config.max_layers_per_uav))
    deployments: list[np.ndarray] = []
    for channel in channels:
        best_order: tuple[int, ...] | None = None
        best_cost = float("inf")
        for order in itertools.permutations(range(config.num_uavs), required):
            cost = sum(
                packet_drop_probability(channel[sender, receiver], config)
                for sender, receiver in itertools.pairwise(order)
            )
            if cost < best_cost:
                best_cost = cost
                best_order = order
        assert best_order is not None
        deployment = np.repeat(best_order, config.max_layers_per_uav)[: config.num_layers]
        deployments.append(deployment)
    return np.asarray(deployments, dtype=np.int64)


def dynamic_programming_proxy_cost(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: SystemConfig,
    latency_reference: float,
) -> float:
    """Return the additive quality-latency proxy minimized by the DP baseline.

    The true surrogate PPL and optimally shared communication latency are nonlinear
    functions of the complete deployment. They therefore do not admit the compact
    Bellman state used below. The baseline replaces them only during action selection
    with the sum of boundary drop rates and independent full-bandwidth link latencies.
    Final evaluation continues to use the unchanged experiment reward.
    """

    if latency_reference <= 0.0:
        raise ValueError("latency_reference must be positive")

    computation = sum(
        config.compute_seconds_per_layer / config.compute_speed[int(uav)] for uav in deployment
    )
    transition_drop = 0.0
    transition_latency = 0.0
    for layer in deployment_boundaries(deployment):
        sender = int(deployment[layer])
        receiver = int(deployment[layer + 1])
        gain = float(channel[sender, receiver])
        snr = config.transmit_power * gain / config.noise_power
        spectral_efficiency = np.log2(1.0 + snr)
        transition_drop += packet_drop_probability(gain, config)
        transition_latency += config.activation_size_mbit / (
            config.total_bandwidth_mhz * spectral_efficiency
        )

    return float(
        config.quality_weight * transition_drop
        + (1.0 - config.quality_weight) * (computation + transition_latency) / latency_reference
    )


def _dynamic_programming_deployment(
    channel: np.ndarray,
    config: SystemConfig,
    latency_reference: float,
) -> np.ndarray:
    """Find the exact minimum-cost deployment for the additive DP proxy."""

    if latency_reference <= 0.0:
        raise ValueError("latency_reference must be positive")

    initial_state: DPState = (0, 0, -1)
    costs: dict[DPState, float] = {initial_state: 0.0}
    predecessors: dict[DPState, tuple[DPState, int, int]] = {}

    for assigned_layers in range(config.num_layers):
        states_at_layer = sorted(
            (state for state in costs if state[0] == assigned_layers),
            key=lambda state: (state[1], state[2]),
        )
        for state in states_at_layer:
            _, used_mask, previous_uav = state
            unused_uavs = config.num_uavs - used_mask.bit_count()
            for uav in range(config.num_uavs):
                if used_mask & (1 << uav):
                    continue
                new_mask = used_mask | (1 << uav)
                for segment_length in range(1, config.max_layers_per_uav + 1):
                    new_assigned = assigned_layers + segment_length
                    if new_assigned > config.num_layers:
                        break

                    remaining_layers = config.num_layers - new_assigned
                    remaining_uavs = unused_uavs - 1
                    if remaining_layers > remaining_uavs * config.max_layers_per_uav:
                        continue

                    computation = (
                        segment_length
                        * config.compute_seconds_per_layer
                        / config.compute_speed[uav]
                    )
                    added_cost = (1.0 - config.quality_weight) * computation / latency_reference
                    if previous_uav >= 0:
                        gain = float(channel[previous_uav, uav])
                        snr = config.transmit_power * gain / config.noise_power
                        spectral_efficiency = np.log2(1.0 + snr)
                        link_latency = config.activation_size_mbit / (
                            config.total_bandwidth_mhz * spectral_efficiency
                        )
                        added_cost += config.quality_weight * packet_drop_probability(gain, config)
                        added_cost += (
                            (1.0 - config.quality_weight) * link_latency / latency_reference
                        )

                    next_state = (new_assigned, new_mask, uav)
                    candidate_cost = costs[state] + added_cost
                    if candidate_cost < costs.get(next_state, float("inf")):
                        costs[next_state] = candidate_cost
                        predecessors[next_state] = (state, uav, segment_length)

    terminal_states = [state for state in costs if state[0] == config.num_layers]
    if not terminal_states:
        raise RuntimeError("dynamic programming found no feasible deployment")
    state = min(terminal_states, key=lambda item: (costs[item], item[1], item[2]))

    segments: list[tuple[int, int]] = []
    while state != initial_state:
        previous_state, uav, segment_length = predecessors[state]
        segments.append((uav, segment_length))
        state = previous_state
    segments.reverse()
    return np.concatenate([np.full(length, uav, dtype=np.int64) for uav, length in segments])


def dynamic_programming_baseline(
    channels: np.ndarray,
    config: SystemConfig,
    latency_reference: float,
) -> np.ndarray:
    """Optimize a continuous, capacity-limited deployment on every channel."""

    channels = np.asarray(channels)
    expected_shape = (config.num_uavs, config.num_uavs)
    if channels.ndim != 3 or channels.shape[1:] != expected_shape:
        raise ValueError(f"channels must have shape (N, {expected_shape[0]}, {expected_shape[1]})")
    return np.stack(
        [
            _dynamic_programming_deployment(channel, config, latency_reference)
            for channel in channels
        ]
    )
