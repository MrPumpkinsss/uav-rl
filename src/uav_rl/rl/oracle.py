"""Reward-aware oracle search for PPO teachers and evaluation baselines."""

from __future__ import annotations

import itertools
import time
from functools import cache

import numpy as np

from uav_rl.config import SystemConfig
from uav_rl.rl.environment import DeploymentEnvironment


class LocalFourSegmentQualityModel:
    """k-NN quality model specialized to capacity-filling four-segment actions."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        boundary_indices: np.ndarray,
        neighbors: int = 8,
    ) -> None:
        if not 1 <= neighbors <= len(features):
            raise ValueError("neighbors must be between one and the sample count")
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
        self.neighbors = neighbors

    @classmethod
    def from_dataset(
        cls,
        dataset_path: str,
        config: SystemConfig,
        *,
        neighbors: int = 8,
    ) -> LocalFourSegmentQualityModel:
        data = np.load(dataset_path)
        deployments = data["deployments"]
        segment_counts = 1 + np.sum(deployments[:, 1:] != deployments[:, :-1], axis=1)
        required_segments = int(np.ceil(config.num_layers / config.max_layers_per_uav))
        selected = segment_counts == required_segments
        boundary_indices = np.arange(
            config.max_layers_per_uav - 1,
            config.num_layers - 1,
            config.max_layers_per_uav,
        )
        features = data["drop_probabilities"][selected][:, boundary_indices]
        targets = data["log_ppl_ratio"][selected]
        if len(features) < neighbors:
            raise ValueError("dataset has too few four-segment samples")
        return cls(features, targets, boundary_indices, neighbors)

    def predict(self, drop_probabilities: np.ndarray) -> np.ndarray:
        queries = np.asarray(drop_probabilities, dtype=np.float32)[:, self.boundary_indices]
        squared_distances = np.sum(
            (queries[:, None, :] - self.features[None, :, :]) ** 2,
            axis=2,
        )
        nearest = np.argpartition(squared_distances, self.neighbors - 1, axis=1)[
            :, : self.neighbors
        ]
        distances = np.take_along_axis(squared_distances, nearest, axis=1)
        weights = 1.0 / (distances + 1e-8)
        neighbor_targets = self.targets[nearest]
        return np.sum(neighbor_targets * weights, axis=1) / np.sum(weights, axis=1)


@cache
def four_segment_candidates(config: SystemConfig) -> np.ndarray:
    """Enumerate deployments that fill the minimum number of UAVs to capacity."""

    segment_count = int(np.ceil(config.num_layers / config.max_layers_per_uav))
    if segment_count > config.num_uavs:
        raise ValueError("not enough UAVs for a capacity-filling deployment")
    candidates = [
        np.repeat(order, config.max_layers_per_uav)[: config.num_layers]
        for order in itertools.permutations(range(config.num_uavs), segment_count)
    ]
    result = np.asarray(candidates, dtype=np.int64)
    result.setflags(write=False)
    return result


@cache
def continuous_deployment_candidates(config: SystemConfig) -> np.ndarray:
    """Enumerate every deployment satisfying capacity and continuity constraints."""

    minimum_segments = int(np.ceil(config.num_layers / config.max_layers_per_uav))
    candidates: list[np.ndarray] = []
    for segment_count in range(minimum_segments, config.num_uavs + 1):
        lengths = [
            values
            for values in itertools.product(
                range(1, config.max_layers_per_uav + 1), repeat=segment_count
            )
            if sum(values) == config.num_layers
        ]
        for order in itertools.permutations(range(config.num_uavs), segment_count):
            candidates.extend(
                np.repeat(order, segment_lengths).astype(np.int64) for segment_lengths in lengths
            )
    if not candidates:
        raise RuntimeError("no feasible continuous deployment candidates")
    result = np.stack(candidates)
    result.setflags(write=False)
    return result


def surrogate_oracle_deployments(
    channels: np.ndarray,
    environment: DeploymentEnvironment,
    candidates: np.ndarray,
    *,
    candidate_batch_size: int = 8192,
    progress_label: str | None = None,
) -> np.ndarray:
    """Select the exact best candidate under the complete surrogate reward.

    Candidate-independent deployment metadata is vectorized once. For each channel,
    packet drops and the non-additive shared-bandwidth latency are then evaluated for
    every candidate. Only surrogate inference is chunked to bound accelerator memory.
    """

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    config = environment.config
    channels = np.asarray(channels)
    candidates = np.asarray(candidates, dtype=np.int64)
    expected_channel_shape = (config.num_uavs, config.num_uavs)
    if channels.ndim != 3 or channels.shape[1:] != expected_channel_shape:
        raise ValueError(f"channels must have shape (N, {config.num_uavs}, {config.num_uavs})")
    if candidates.ndim != 2 or candidates.shape[1] != config.num_layers:
        raise ValueError(f"candidates must have shape (K, {config.num_layers})")

    senders = candidates[:, :-1]
    receivers = candidates[:, 1:]
    transition_mask = senders != receivers
    compute_speed = np.asarray(config.compute_speed, dtype=np.float64)
    computation_seconds = np.sum(
        config.compute_seconds_per_layer / compute_speed[candidates], axis=1
    )
    selected = np.empty((len(channels), config.num_layers), dtype=np.int64)
    started_at = time.perf_counter()

    for channel_index, channel in enumerate(channels):
        gains = np.asarray(channel, dtype=np.float64)[senders, receivers]
        snr = config.transmit_power * gains / config.noise_power
        spectral_efficiency = np.log2(1.0 + snr)

        drops = 1.0 - np.exp(
            -config.decoding_threshold * config.noise_power / (config.transmit_power * gains)
        )
        drops = np.where(transition_mask, drops, 0.0).astype(np.float32)

        coefficients = np.where(
            transition_mask,
            config.activation_size_mbit / spectral_efficiency,
            0.0,
        )
        root_sums = np.sqrt(coefficients).sum(axis=1)
        latency_seconds = computation_seconds + root_sums**2 / config.total_bandwidth_mhz

        best_index = -1
        best_reward = -float("inf")
        for start in range(0, len(candidates), candidate_batch_size):
            stop = min(start + candidate_batch_size, len(candidates))
            qualities = environment.quality_evaluator.evaluate(drops[start:stop])
            rewards = -(
                config.quality_weight * qualities
                + (1.0 - config.quality_weight)
                * latency_seconds[start:stop]
                / environment.latency_reference
            )
            local_index = int(np.argmax(rewards))
            local_reward = float(rewards[local_index])
            if local_reward > best_reward:
                best_reward = local_reward
                best_index = start + local_index

        if best_index < 0:
            raise RuntimeError("surrogate oracle found no candidate")
        selected[channel_index] = candidates[best_index]
        completed = channel_index + 1
        if progress_label and (completed == 1 or completed % 16 == 0 or completed == len(channels)):
            print(
                f"oracle={progress_label} channels={completed}/{len(channels)} "
                f"candidates={len(candidates)} elapsed={time.perf_counter() - started_at:.1f}s",
                flush=True,
            )
    return selected


def four_segment_surrogate_oracle(
    channels: np.ndarray,
    environment: DeploymentEnvironment,
    *,
    progress_label: str | None = None,
) -> np.ndarray:
    """Select the best capacity-filling path under the surrogate reward."""

    return surrogate_oracle_deployments(
        channels,
        environment,
        four_segment_candidates(environment.config),
        progress_label=progress_label,
    )


def full_surrogate_oracle(
    channels: np.ndarray,
    environment: DeploymentEnvironment,
    *,
    progress_label: str | None = None,
) -> np.ndarray:
    """Select the best valid continuous deployment under the surrogate reward."""

    return surrogate_oracle_deployments(
        channels,
        environment,
        continuous_deployment_candidates(environment.config),
        progress_label=progress_label,
    )


def reward_oracle_deployments(
    channels: np.ndarray,
    environment: DeploymentEnvironment,
    *,
    channel_batch_size: int = 32,
    quality_model: LocalFourSegmentQualityModel | None = None,
) -> np.ndarray:
    """Select the highest teacher-reward capacity-filling path per channel.

    The teacher keeps PPO's reward weights and latency term. Its quality term can
    come from the global surrogate or a local model fitted to matching actions.
    Batching bounds the temporary channel tensor while keeping evaluation overhead low.
    """

    candidates = four_segment_candidates(environment.config)
    if quality_model is None:
        return surrogate_oracle_deployments(channels, environment, candidates)

    selected: list[np.ndarray] = []
    for start in range(0, len(channels), channel_batch_size):
        channel_batch = channels[start : start + channel_batch_size]
        candidate_count = len(candidates)
        expanded_channels = np.repeat(channel_batch, candidate_count, axis=0)
        expanded_deployments = np.tile(candidates, (len(channel_batch), 1))
        rewards, details = environment.evaluate(expanded_channels, expanded_deployments)
        qualities = quality_model.predict(details["drop_probabilities"])
        rewards = -(
            environment.config.quality_weight * qualities
            + (1.0 - environment.config.quality_weight) * details["normalized_latency"]
        )
        rewards = rewards.reshape(len(channel_batch), candidate_count)
        selected.extend(candidates[rewards.argmax(axis=1)])
    return np.asarray(selected, dtype=np.int64)
