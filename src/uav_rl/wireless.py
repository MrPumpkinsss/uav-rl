"""Wireless channel, packet-drop, and collaborative latency models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uav_rl.config import SystemConfig
from uav_rl.deployment import deployment_boundaries, validate_deployment


@dataclass(frozen=True)
class LatencyResult:
    computation_seconds: float
    communication_seconds: float
    total_seconds: float


def sample_channel(rng: np.random.Generator, config: SystemConfig) -> np.ndarray:
    """Sample a symmetric channel-gain matrix with ideal self-links."""

    upper = rng.uniform(
        config.channel_gain_min,
        config.channel_gain_max,
        size=(config.num_uavs, config.num_uavs),
    )
    channel = (upper + upper.T) / 2.0
    np.fill_diagonal(channel, config.channel_gain_max)
    return channel


def packet_drop_probability(channel_gain: float, config: SystemConfig) -> float:
    """Evaluate the paper's Rayleigh-fading packet-drop expression."""

    exponent = -(
        config.decoding_threshold * config.noise_power / (config.transmit_power * channel_gain)
    )
    return float(1.0 - math.exp(exponent))


def boundary_drop_probabilities(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    """Map a deployment and channel to one activation-drop rate per layer boundary."""

    validate_deployment(deployment, config)
    probabilities = np.zeros(config.num_layers - 1, dtype=np.float32)
    for layer in deployment_boundaries(deployment):
        sender = int(deployment[layer])
        receiver = int(deployment[layer + 1])
        probabilities[layer] = packet_drop_probability(channel[sender, receiver], config)
    return probabilities


def collaborative_latency(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: SystemConfig,
) -> LatencyResult:
    """Compute layer execution plus optimally allocated transition bandwidth latency."""

    validate_deployment(deployment, config)
    computation = sum(
        config.compute_seconds_per_layer / config.compute_speed[int(uav)] for uav in deployment
    )

    coefficients: list[float] = []
    for layer in deployment_boundaries(deployment):
        sender = int(deployment[layer])
        receiver = int(deployment[layer + 1])
        snr = config.transmit_power * channel[sender, receiver] / config.noise_power
        spectral_efficiency = math.log2(1.0 + snr)
        coefficients.append(config.activation_size_mbit / spectral_efficiency)

    if not coefficients:
        communication = 0.0
    else:
        # Minimize sum(a_i / B_i), subject to sum(B_i)=Bmax: B_i proportional to sqrt(a_i).
        root_sum = sum(math.sqrt(value) for value in coefficients)
        communication = root_sum**2 / config.total_bandwidth_mhz

    return LatencyResult(
        computation_seconds=float(computation),
        communication_seconds=float(communication),
        total_seconds=float(computation + communication),
    )
