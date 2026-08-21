"""Baselines for the resource-constrained arbitrary layer assignment model.

These baselines are deliberately independent of PPO.  They are used only for
held-out comparison and never write labels back to surrogate training data.
"""

from __future__ import annotations

import itertools

import numpy as np

from uav_rl.data.general_assignment_dataset import sample_general_assignment
from uav_rl.resource_assignment import (
    ResourceConstrainedConfig,
    layerwise_drop_probabilities,
    layerwise_latency,
    validate_layerwise_deployment,
)
from uav_rl.resource_environment import ResourceDeploymentEnvironment
from uav_rl.rl.layerwise_policy import valid_layer_action_mask
from uav_rl.wireless import packet_drop_probability


def fixed_eight_candidates(config: ResourceConstrainedConfig) -> np.ndarray:
    """All four-segment, eight-layer paths, including only feasible paths."""

    system = config.system
    if system.num_layers != 4 * system.max_layers_per_uav:
        raise ValueError("fixed-eight candidates require exactly four full segments")
    candidates: list[np.ndarray] = []
    for order in itertools.permutations(range(system.num_uavs), 4):
        deployment = np.repeat(np.asarray(order, dtype=np.int64), system.max_layers_per_uav)
        try:
            validate_layerwise_deployment(deployment, config)
        except ValueError:
            continue
        candidates.append(deployment)
    if not candidates:
        raise RuntimeError("no feasible fixed-eight candidate exists")
    return np.stack(candidates)


def _score_proxy(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
    *,
    include_latency: bool = True,
) -> float:
    """Additive drop/latency proxy used only for baseline action selection."""

    drops = layerwise_drop_probabilities(deployment, channel, config)
    latency = layerwise_latency(deployment, channel, config).total_seconds
    latency_term = latency / latency_reference if include_latency else 0.0
    return float(config.system.quality_weight * drops.sum() + (1.0 - config.system.quality_weight) * latency_term)


def fixed_eight_proxy_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
    *,
    score: str = "proxy",
) -> np.ndarray:
    """Choose the best feasible fixed-eight path under a transparent proxy."""

    candidates = fixed_eight_candidates(config)
    selected: list[np.ndarray] = []
    for channel in np.asarray(channels):
        if score == "strong_link":
            values = [
                sum(
                    packet_drop_probability(
                        float(channel[sender, receiver]), config.system
                    )
                    for sender, receiver in itertools.pairwise(
                        tuple(int(value) for value in candidate[:: config.system.max_layers_per_uav])
                    )
                )
                for candidate in candidates
            ]
        elif score == "compute":
            values = [
                float(
                    np.sum(
                        np.asarray(config.layer_compute_seconds_at_unit_speed)
                        / np.asarray(config.system.compute_speed)[candidate]
                    )
                )
                for candidate in candidates
            ]
        else:
            values = [
                _score_proxy(candidate, channel, config, latency_reference)
                for candidate in candidates
            ]
        selected.append(candidates[int(np.argmin(values))])
    return np.stack(selected)


def random_feasible_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    *,
    seed: int,
    candidates_per_channel: int = 512,
) -> np.ndarray:
    """Sample feasible arbitrary assignments and return one reproducible sample."""

    if candidates_per_channel < 1:
        raise ValueError("candidates_per_channel must be positive")
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for channel in np.asarray(channels):
        target = int(rng.integers(3, min(14, config.system.num_layers - 1) + 1))
        result.append(
            sample_general_assignment(
                rng,
                channel,
                config,
                target_boundaries=target,
                max_attempts=20_000,
            )
        )
    return np.stack(result)


def surrogate_random_search(
    channels: np.ndarray,
    environment: ResourceDeploymentEnvironment,
    *,
    seed: int,
    candidates_per_channel: int = 1024,
) -> np.ndarray:
    """Monte-Carlo surrogate oracle over arbitrary feasible assignments."""

    if candidates_per_channel < 1:
        raise ValueError("candidates_per_channel must be positive")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for channel in np.asarray(channels):
        candidates = []
        for _ in range(candidates_per_channel):
            target = int(rng.integers(3, min(14, environment.system.num_layers - 1) + 1))
            candidates.append(
                sample_general_assignment(
                    rng,
                    channel,
                    environment.config,
                    target_boundaries=target,
                    max_attempts=20_000,
                )
            )
        candidate_array = np.stack(candidates)
        repeated = np.repeat(channel[None, :, :], len(candidate_array), axis=0)
        rewards, _ = environment.evaluate(repeated, candidate_array)
        selected.append(candidate_array[int(np.argmax(rewards))])
    return np.stack(selected)


def proxy_beam_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
    *,
    beam_width: int = 128,
    max_boundaries: int | None = None,
) -> np.ndarray:
    """Beam search arbitrary assignments under the additive proxy objective."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    layers = config.system.num_layers
    uavs = config.system.num_uavs
    speeds = np.asarray(config.system.compute_speed, dtype=np.float64)
    result: list[np.ndarray] = []
    for channel in np.asarray(channels):
        # deployment, memory, energy, previous, boundaries
        beams: list[tuple[float, np.ndarray, np.ndarray, int | None, int, list[int]]] = [
            (0.0, np.zeros(uavs), np.zeros(uavs), None, 0, [])
        ]
        for layer_index in range(layers):
            expanded: list[tuple[float, np.ndarray, np.ndarray, int, int, list[int]]] = []
            for cost, memory, energy, previous, boundaries, deployment in beams:
                mask = valid_layer_action_mask(
                    layer_index=layer_index,
                    memory_used=memory,
                    energy_used=energy,
                    config=config,
                )
                if max_boundaries is not None and boundaries >= max_boundaries and previous is not None:
                    forced = np.zeros_like(mask)
                    forced[previous] = mask[previous]
                    if forced.any():
                        mask = forced
                for action in np.flatnonzero(mask):
                    action = int(action)
                    next_memory = memory.copy()
                    next_energy = energy.copy()
                    next_memory[action] += config.layer_memory_units[layer_index]
                    next_energy[action] += config.compute_energy_coefficient * speeds[action] ** 2 * config.layer_compute_seconds_at_unit_speed[layer_index]
                    next_boundaries = boundaries + int(previous is not None and action != previous)
                    # Incremental compute cost; communication/drop is paid at boundaries.
                    added = (1.0 - config.system.quality_weight) * (
                        config.layer_compute_seconds_at_unit_speed[layer_index] / speeds[action]
                    ) / latency_reference
                    if previous is not None and action != previous:
                        gain = float(channel[previous, action])
                        added += config.system.quality_weight * packet_drop_probability(gain, config.system)
                        spectral = np.log2(1.0 + config.system.transmit_power * gain / config.system.noise_power)
                        added += (1.0 - config.system.quality_weight) * (
                            config.activation_mbit_by_boundary[layer_index - 1]
                            / (config.system.total_bandwidth_mhz * spectral)
                        ) / latency_reference
                    expanded.append((cost + float(added), next_memory, next_energy, action, next_boundaries, deployment + [action]))
            expanded.sort(key=lambda item: item[0])
            beams = expanded[:beam_width]
        candidates = np.stack([np.asarray(item[-1], dtype=np.int64) for item in beams])
        values = [
            _score_proxy(candidate, channel, config, latency_reference)
            for candidate in candidates
        ]
        result.append(candidates[int(np.argmin(values))])
    return np.stack(result)



def dynamic_programming_proxy_cost(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
) -> float:
    """Return the additive proxy minimized by the continuous-segment DP.

    The proxy is used only to select an action.  It sums per-layer compute
    latency and independent full-bandwidth latency at each boundary, plus the
    boundary drop probabilities.  Final evaluation still uses the unchanged
    shared-bandwidth resource model and the true CodeLlama evaluator.
    """

    if latency_reference <= 0.0:
        raise ValueError("latency_reference must be positive")
    values = np.asarray(deployment, dtype=np.int64)
    validate_layerwise_deployment(values, config, channel=channel)
    system = config.system
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    compute_seconds = np.asarray(config.layer_compute_seconds_at_unit_speed, dtype=np.float64)
    computation = float(np.sum(compute_seconds / speeds[values]))
    transition_drop = 0.0
    transition_latency = 0.0
    activation = np.asarray(config.activation_mbit_by_boundary, dtype=np.float64)
    for boundary in np.flatnonzero(values[:-1] != values[1:]):
        sender = int(values[boundary])
        receiver = int(values[boundary + 1])
        gain = float(channel[sender, receiver])
        spectral_efficiency = np.log2(
            1.0 + system.transmit_power * gain / system.noise_power
        )
        transition_drop += packet_drop_probability(gain, system)
        transition_latency += activation[boundary] / (
            system.total_bandwidth_mhz * spectral_efficiency
        )
    return float(
        system.quality_weight * transition_drop
        + (1.0 - system.quality_weight)
        * (computation + transition_latency)
        / latency_reference
    )


def _dynamic_programming_deployment(
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
) -> np.ndarray:
    """Find the best feasible variable-length contiguous-segment deployment.

    The recurrence chooses an unused UAV and the length of its next segment.
    Unlike fixed-eight baselines, segment boundaries may occur at any layer and
    the DP may use four or five distinct UAVs.  We retain complete transition
    paths because exact shared-bandwidth energy feasibility is checked only at
    the terminal deployment.
    """

    if latency_reference <= 0.0:
        raise ValueError("latency_reference must be positive")
    system = config.system
    layers = system.num_layers
    max_segment = system.max_layers_per_uav
    memory_profile = np.asarray(config.layer_memory_units, dtype=np.float64)
    capacities = np.asarray(config.uav_memory_capacity_units, dtype=np.float64)
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    compute_seconds = np.asarray(config.layer_compute_seconds_at_unit_speed, dtype=np.float64)
    activation = np.asarray(config.activation_mbit_by_boundary, dtype=np.float64)
    best: tuple[float, np.ndarray] | None = None

    def visit(
        assigned_layers: int,
        used_mask: int,
        previous_uav: int,
        parts: tuple[np.ndarray, ...],
        cost: float,
    ) -> None:
        nonlocal best
        if assigned_layers == layers:
            candidate = np.concatenate(parts).astype(np.int64, copy=False)
            try:
                validate_layerwise_deployment(candidate, config, channel=channel)
            except ValueError:
                return
            if best is None or cost < best[0]:
                best = (cost, candidate.copy())
            return

        unused_uavs = system.num_uavs - used_mask.bit_count()
        if layers - assigned_layers > unused_uavs * max_segment:
            return
        for uav in range(system.num_uavs):
            if used_mask & (1 << uav):
                continue
            for segment_length in range(1, max_segment + 1):
                end = assigned_layers + segment_length
                if end > layers:
                    break
                segment_memory = float(memory_profile[assigned_layers:end].sum())
                if segment_memory > capacities[uav] + 1e-9:
                    break
                remaining_layers = layers - end
                remaining_uavs = unused_uavs - 1
                if remaining_layers > remaining_uavs * max_segment:
                    continue

                segment_cost = float(
                    (1.0 - system.quality_weight)
                    * np.sum(compute_seconds[assigned_layers:end] / speeds[uav])
                    / latency_reference
                )
                if previous_uav >= 0:
                    boundary = assigned_layers - 1
                    gain = float(channel[previous_uav, uav])
                    spectral_efficiency = np.log2(
                        1.0 + system.transmit_power * gain / system.noise_power
                    )
                    segment_cost += system.quality_weight * packet_drop_probability(
                        gain, system
                    )
                    segment_cost += (
                        (1.0 - system.quality_weight)
                        * activation[boundary]
                        / (system.total_bandwidth_mhz * spectral_efficiency)
                        / latency_reference
                    )
                visit(
                    end,
                    used_mask | (1 << uav),
                    uav,
                    parts + (np.full(segment_length, uav, dtype=np.int64),),
                    cost + segment_cost,
                )

    visit(0, 0, -1, (), 0.0)
    if best is None:
        raise RuntimeError("dynamic programming found no feasible deployment")
    return best[1]


def dynamic_programming_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
) -> np.ndarray:
    """Optimize variable-length contiguous segments for every channel."""

    values = np.asarray(channels, dtype=np.float64)
    expected_shape = (config.system.num_uavs, config.system.num_uavs)
    if values.ndim != 3 or values.shape[1:] != expected_shape:
        raise ValueError(f"channels must have shape (N, {expected_shape[0]}, {expected_shape[1]})")
    return np.stack(
        [
            _dynamic_programming_deployment(channel, config, latency_reference)
            for channel in values
        ]
    )

__all__ = [
    "fixed_eight_candidates",
    "fixed_eight_proxy_baseline",
    "random_feasible_baseline",
    "surrogate_random_search",
    "proxy_beam_baseline",
    "dynamic_programming_proxy_cost",
    "dynamic_programming_baseline",
]
