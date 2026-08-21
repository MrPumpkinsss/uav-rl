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



def _local_search_neighbors(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
) -> np.ndarray:
    """Generate feasible one-step neighbors around an arbitrary deployment."""

    values = np.asarray(deployment, dtype=np.int64)
    system = config.system
    candidates: set[tuple[int, ...]] = set()

    # Move one layer to another UAV. This also shifts a boundary by one layer.
    for layer_index in range(system.num_layers):
        current_uav = int(values[layer_index])
        for uav in range(system.num_uavs):
            if uav == current_uav:
                continue
            candidate = values.copy()
            candidate[layer_index] = uav
            try:
                validate_layerwise_deployment(candidate, config, channel=channel)
            except ValueError:
                continue
            candidates.add(tuple(int(item) for item in candidate))

    # Move a whole contiguous run to another UAV. Adjacent runs may merge.
    starts = [0]
    starts.extend((np.flatnonzero(values[:-1] != values[1:]) + 1).tolist())
    ends = starts[1:] + [system.num_layers]
    for start, end in zip(starts, ends, strict=True):
        current_uav = int(values[start])
        for uav in range(system.num_uavs):
            if uav == current_uav:
                continue
            candidate = values.copy()
            candidate[start:end] = uav
            try:
                validate_layerwise_deployment(candidate, config, channel=channel)
            except ValueError:
                continue
            candidates.add(tuple(int(item) for item in candidate))

    if not candidates:
        return np.empty((0, system.num_layers), dtype=np.int64)
    return np.asarray(sorted(candidates), dtype=np.int64)


def proxy_beam_surrogate_local_search(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    environment: ResourceDeploymentEnvironment,
    latency_reference: float,
    *,
    beam_width: int = 512,
    rounds: int = 3,
) -> np.ndarray:
    """Improve a wide proxy-beam deployment with surrogate-guided local search.

    The proxy beam supplies the initial arbitrary assignment. Each local-search
    round evaluates feasible one-layer and one-run neighbors with the frozen
    surrogate plus exact analytic latency, then accepts the best improvement.
    No true CodeLlama call is made by this function.
    """

    if rounds < 0:
        raise ValueError("rounds cannot be negative")
    initial = proxy_beam_baseline(
        channels, config, latency_reference, beam_width=beam_width
    )
    selected: list[np.ndarray] = []
    for channel, deployment in zip(np.asarray(channels), initial, strict=True):
        current = np.asarray(deployment, dtype=np.int64).copy()
        current_rewards, _ = environment.evaluate(
            channel[None, :, :], current[None, :]
        )
        current_reward = float(current_rewards[0])
        for _ in range(rounds):
            neighbors = _local_search_neighbors(current, channel, config)
            if len(neighbors) == 0:
                break
            repeated_channels = np.repeat(channel[None, :, :], len(neighbors), axis=0)
            rewards, _ = environment.evaluate(repeated_channels, neighbors)
            best_index = int(np.argmax(rewards))
            best_reward = float(rewards[best_index])
            if best_reward <= current_reward + 1e-8:
                break
            current = neighbors[best_index].copy()
            current_reward = best_reward
        selected.append(current)
    return np.stack(selected)
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


def _safe_sample_assignment(
    rng: np.random.Generator,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
) -> np.ndarray:
    """Sample a feasible arbitrary assignment with a deterministic fallback."""

    target = int(rng.integers(3, min(14, config.system.num_layers - 1) + 1))
    try:
        return sample_general_assignment(
            rng, channel, config, target_boundaries=target, max_attempts=500
        )
    except RuntimeError:
        return proxy_beam_baseline(
            channel[None, ...], config, latency_reference, beam_width=128
        )[0]


def _mutate_assignment(
    assignment: np.ndarray,
    rng: np.random.Generator,
    config: ResourceConstrainedConfig,
    channel: np.ndarray,
) -> np.ndarray:
    """Apply a one-layer or contiguous-run mutation and retain feasibility."""

    candidate = np.asarray(assignment, dtype=np.int64).copy()
    if rng.random() < 0.5:
        index = int(rng.integers(0, config.system.num_layers))
        candidate[index] = int(rng.integers(0, config.system.num_uavs))
    else:
        start = int(rng.integers(0, config.system.num_layers))
        end = int(rng.integers(start + 1, config.system.num_layers + 1))
        candidate[start:end] = int(rng.integers(0, config.system.num_uavs))
    try:
        validate_layerwise_deployment(candidate, config, channel=channel)
    except ValueError:
        return np.asarray(assignment, dtype=np.int64).copy()
    return candidate


def constrained_genetic_surrogate_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    environment: ResourceDeploymentEnvironment,
    *,
    population_size: int = 64,
    generations: int = 64,
    mutation_rate: float = 0.08,
    seed: int = 20260828,
) -> np.ndarray:
    """Constrained GA over arbitrary assignments using frozen surrogate reward."""

    if population_size < 4 or generations < 1:
        raise ValueError("population_size must be >= 4 and generations must be positive")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    starts = proxy_beam_baseline(
        np.asarray(channels), config, environment.latency_reference, beam_width=512
    )
    selected: list[np.ndarray] = []
    for channel, start in zip(np.asarray(channels), starts, strict=True):
        population = [np.asarray(start, dtype=np.int64).copy()]
        seen = {tuple(int(value) for value in population[0])}
        attempts = 0
        while len(population) < population_size and attempts < population_size * 20:
            attempts += 1
            candidate = _safe_sample_assignment(
                rng, channel, config, environment.latency_reference
            )
            key = tuple(int(value) for value in candidate)
            if key not in seen:
                seen.add(key)
                population.append(candidate)
        while len(population) < population_size:
            population.append(population[int(rng.integers(0, len(population)))].copy())
        population_array = np.stack(population)
        repeated = np.repeat(channel[None, ...], len(population_array), axis=0)
        rewards, _ = environment.evaluate(repeated, population_array)
        best_index = int(np.argmax(rewards))
        best = population_array[best_index].copy()
        best_reward = float(rewards[best_index])
        elite_count = max(2, population_size // 5)
        for _ in range(generations):
            order = np.argsort(rewards)[::-1]
            elites = population_array[order[:elite_count]]
            next_population = [elite.copy() for elite in elites]
            while len(next_population) < population_size:
                first = elites[int(rng.integers(0, len(elites)))]
                second = elites[int(rng.integers(0, len(elites)))]
                cut = int(rng.integers(1, config.system.num_layers))
                child = np.concatenate((first[:cut], second[cut:])).astype(np.int64)
                if rng.random() < mutation_rate:
                    child = _mutate_assignment(child, rng, config, channel)
                try:
                    validate_layerwise_deployment(child, config, channel=channel)
                except ValueError:
                    child = _safe_sample_assignment(
                        rng, channel, config, environment.latency_reference
                    )
                next_population.append(child)
            population_array = np.stack(next_population)
            repeated = np.repeat(channel[None, ...], len(population_array), axis=0)
            rewards, _ = environment.evaluate(repeated, population_array)
            best_index = int(np.argmax(rewards))
            if float(rewards[best_index]) > best_reward:
                best = population_array[best_index].copy()
                best_reward = float(rewards[best_index])
        selected.append(best)
    return np.stack(selected)


def surrogate_simulated_annealing_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    environment: ResourceDeploymentEnvironment,
    *,
    steps: int = 4096,
    initial_temperature: float = 0.02,
    final_temperature: float = 0.0005,
    seed: int = 20260829,
) -> np.ndarray:
    """Improve beam-512 assignments with surrogate simulated annealing."""

    if steps < 1 or initial_temperature <= 0.0 or final_temperature <= 0.0:
        raise ValueError("steps and temperatures must be positive")
    if final_temperature > initial_temperature:
        raise ValueError("final_temperature cannot exceed initial_temperature")
    rng = np.random.default_rng(seed)
    starts = proxy_beam_baseline(
        np.asarray(channels), config, environment.latency_reference, beam_width=512
    )
    selected: list[np.ndarray] = []
    for channel, start in zip(np.asarray(channels), starts, strict=True):
        current = np.asarray(start, dtype=np.int64).copy()
        current_reward = float(
            environment.evaluate(channel[None, ...], current[None, :])[0][0]
        )
        best = current.copy()
        best_reward = current_reward
        for step in range(steps):
            candidate = _mutate_assignment(current, rng, config, channel)
            candidate_reward = float(
                environment.evaluate(channel[None, ...], candidate[None, :])[0][0]
            )
            fraction = step / max(1, steps - 1)
            temperature = initial_temperature * (1.0 - fraction) + final_temperature * fraction
            delta = candidate_reward - current_reward
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current = candidate
                current_reward = candidate_reward
            if current_reward > best_reward:
                best = current.copy()
                best_reward = current_reward
        selected.append(best)
    return np.stack(selected)


def coedge_adaptive_partition_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
) -> np.ndarray:
    """Choose dynamic contiguous segments using local compute/link marginal cost."""

    system = config.system
    memory_profile = np.asarray(config.layer_memory_units, dtype=np.float64)
    compute_seconds = np.asarray(config.layer_compute_seconds_at_unit_speed, dtype=np.float64)
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    capacities = np.asarray(config.uav_memory_capacity_units, dtype=np.float64)
    energy_budget = np.asarray(config.uav_energy_budget_joule, dtype=np.float64)
    hover = np.asarray(config.uav_hover_energy_joule, dtype=np.float64)
    selected: list[np.ndarray] = []
    for channel in np.asarray(channels):
        deployment: list[int] = []
        memory = np.zeros(system.num_uavs, dtype=np.float64)
        compute_energy = np.zeros(system.num_uavs, dtype=np.float64)
        previous: int | None = None
        for layer in range(system.num_layers):
            candidates: list[tuple[float, int]] = []
            for uav in range(system.num_uavs):
                next_memory = memory[uav] + memory_profile[layer]
                next_energy = compute_energy[uav] + (
                    config.compute_energy_coefficient * speeds[uav] ** 2 * compute_seconds[layer]
                )
                if next_memory > capacities[uav] + 1e-9:
                    continue
                if next_energy + hover[uav] > energy_budget[uav] + 1e-9:
                    continue
                cost = (1.0 - system.quality_weight) * compute_seconds[layer] / speeds[uav]
                cost /= latency_reference
                if previous is not None and previous != uav:
                    gain = float(channel[previous, uav])
                    spectral = np.log2(
                        1.0 + system.transmit_power * gain / system.noise_power
                    )
                    cost += system.quality_weight * packet_drop_probability(gain, system)
                    cost += (1.0 - system.quality_weight) * (
                        config.activation_mbit_by_boundary[layer - 1]
                        / (system.total_bandwidth_mhz * spectral * latency_reference)
                    )
                load = next_memory / capacities[uav]
                cost += 0.05 * float(load * load)
                candidates.append((float(cost), uav))
            if not candidates:
                deployment = []
                break
            _, chosen = min(candidates)
            deployment.append(chosen)
            memory[chosen] += memory_profile[layer]
            compute_energy[chosen] += (
                config.compute_energy_coefficient * speeds[chosen] ** 2 * compute_seconds[layer]
            )
            previous = chosen
        candidate = np.asarray(deployment, dtype=np.int64)
        if candidate.shape != (system.num_layers,):
            candidate = proxy_beam_baseline(
                channel[None, ...], config, latency_reference, beam_width=128
            )[0]
        try:
            validate_layerwise_deployment(candidate, config, channel=channel)
        except ValueError:
            candidate = dynamic_programming_baseline(
                channel[None, ...], config, latency_reference
            )[0]
        selected.append(candidate)
    return np.stack(selected)


def neurosurgeon_best_split_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
) -> np.ndarray:
    """Enumerate the best two-UAV contiguous split, Neurosurgeon-style."""

    system = config.system
    selected: list[np.ndarray] = []
    for channel in np.asarray(channels):
        candidates: list[tuple[float, np.ndarray]] = []
        for split in range(1, system.num_layers):
            for left in range(system.num_uavs):
                for right in range(system.num_uavs):
                    if left == right:
                        continue
                    candidate = np.concatenate(
                        (
                            np.full(split, left, dtype=np.int64),
                            np.full(system.num_layers - split, right, dtype=np.int64),
                        )
                    )
                    try:
                        validate_layerwise_deployment(candidate, config, channel=channel)
                    except ValueError:
                        continue
                    candidates.append(
                        (_score_proxy(candidate, channel, config, latency_reference), candidate)
                    )
        if candidates:
            selected.append(min(candidates, key=lambda item: item[0])[1])
        else:
            selected.append(
                proxy_beam_baseline(
                    channel[None, ...], config, latency_reference, beam_width=128
                )[0]
            )
    return np.stack(selected)


def milp_proxy_oracle_baseline(
    channels: np.ndarray,
    config: ResourceConstrainedConfig,
    latency_reference: float,
    *,
    time_limit_seconds: float = 5.0,
) -> np.ndarray:
    """Solve a linearized drop/compute proxy with SciPy/HiGHS MILP."""

    if time_limit_seconds <= 0.0:
        raise ValueError("time_limit_seconds must be positive")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError:
        return proxy_beam_baseline(
            np.asarray(channels), config, latency_reference, beam_width=512
        )
    system = config.system
    layers = system.num_layers
    uavs = system.num_uavs
    pairs = [
        (sender, receiver)
        for sender in range(uavs)
        for receiver in range(uavs)
        if sender != receiver
    ]
    selected: list[np.ndarray] = []
    memory = np.asarray(config.layer_memory_units, dtype=np.float64)
    compute = np.asarray(config.layer_compute_seconds_at_unit_speed, dtype=np.float64)
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    for channel in np.asarray(channels):
        x_count = layers * uavs
        y_count = (layers - 1) * len(pairs)
        variable_count = x_count + y_count

        def x_index(layer: int, uav: int) -> int:
            return layer * uavs + uav

        def y_index(boundary: int, pair_index: int) -> int:
            return x_count + boundary * len(pairs) + pair_index

        objective = np.zeros(variable_count, dtype=np.float64)
        for layer in range(layers):
            for uav in range(uavs):
                objective[x_index(layer, uav)] = (
                    (1.0 - system.quality_weight) * compute[layer] / speeds[uav] / latency_reference
                )
        for boundary in range(layers - 1):
            for pair_index, (sender, receiver) in enumerate(pairs):
                gain = float(channel[sender, receiver])
                spectral = np.log2(
                    1.0 + system.transmit_power * gain / system.noise_power
                )
                edge_cost = system.quality_weight * packet_drop_probability(gain, system)
                edge_cost += (1.0 - system.quality_weight) * (
                    config.activation_mbit_by_boundary[boundary]
                    / (system.total_bandwidth_mhz * spectral * latency_reference)
                )
                objective[y_index(boundary, pair_index)] = edge_cost

        rows: list[dict[int, float]] = []
        lower: list[float] = []
        upper: list[float] = []

        def add_row(entries: dict[int, float], low: float, high: float) -> None:
            rows.append(entries)
            lower.append(low)
            upper.append(high)

        for layer in range(layers):
            add_row({x_index(layer, uav): 1.0 for uav in range(uavs)}, 1.0, 1.0)
        for uav in range(uavs):
            add_row(
                {x_index(layer, uav): float(memory[layer]) for layer in range(layers)},
                -np.inf,
                float(config.uav_memory_capacity_units[uav]),
            )
            add_row(
                {
                    x_index(layer, uav): float(
                        config.compute_energy_coefficient * speeds[uav] ** 2 * compute[layer]
                    )
                    for layer in range(layers)
                },
                -np.inf,
                float(config.uav_energy_budget_joule[uav] - config.uav_hover_energy_joule[uav]),
            )
        for boundary in range(layers - 1):
            for pair_index, (sender, receiver) in enumerate(pairs):
                edge = y_index(boundary, pair_index)
                add_row({edge: 1.0, x_index(boundary, sender): -1.0}, -np.inf, 0.0)
                add_row({edge: 1.0, x_index(boundary + 1, receiver): -1.0}, -np.inf, 0.0)
                add_row(
                    {
                        edge: -1.0,
                        x_index(boundary, sender): 1.0,
                        x_index(boundary + 1, receiver): 1.0,
                    },
                    -np.inf,
                    1.0,
                )
        matrix = lil_matrix((len(rows), variable_count), dtype=np.float64)
        for row_index, entries in enumerate(rows):
            for column, value in entries.items():
                matrix[row_index, column] = value
        result = milp(
            objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(
                matrix.tocsr(), np.asarray(lower), np.asarray(upper)
            ),
            options={"time_limit": float(time_limit_seconds), "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            candidate = proxy_beam_baseline(
                channel[None, ...], config, latency_reference, beam_width=512
            )[0]
        else:
            candidate = np.asarray(
                [
                    int(np.argmax(result.x[layer * uavs : (layer + 1) * uavs]))
                    for layer in range(layers)
                ],
                dtype=np.int64,
            )
            try:
                validate_layerwise_deployment(candidate, config, channel=channel)
            except ValueError:
                candidate = proxy_beam_baseline(
                    channel[None, ...], config, latency_reference, beam_width=512
                )[0]
        selected.append(candidate)
    return np.stack(selected)
__all__ = [
    "fixed_eight_candidates",
    "fixed_eight_proxy_baseline",
    "random_feasible_baseline",
    "surrogate_random_search",
    "proxy_beam_baseline",
    "dynamic_programming_proxy_cost",
    "dynamic_programming_baseline",
    "proxy_beam_surrogate_local_search",
    "constrained_genetic_surrogate_baseline",
    "surrogate_simulated_annealing_baseline",
    "coedge_adaptive_partition_baseline",
    "neurosurgeon_best_split_baseline",
    "milp_proxy_oracle_baseline",
]
