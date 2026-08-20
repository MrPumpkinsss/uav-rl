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


__all__ = [
    "fixed_eight_candidates",
    "fixed_eight_proxy_baseline",
    "random_feasible_baseline",
    "surrogate_random_search",
    "proxy_beam_baseline",
]
