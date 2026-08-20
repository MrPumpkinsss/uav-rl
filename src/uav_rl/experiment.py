"""Shared construction helpers for reproducible deployment experiments."""

from __future__ import annotations

import numpy as np

from uav_rl.config import SystemConfig
from uav_rl.deployment import random_continuous_deployment
from uav_rl.wireless import collaborative_latency, sample_channel


def estimate_latency_reference(
    config: SystemConfig,
    seed: int,
    samples: int = 1024,
) -> float:
    """Estimate the deterministic normalization constant used by all rewards."""

    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    latencies = []
    for _ in range(samples):
        channel = sample_channel(rng, config)
        deployment = random_continuous_deployment(rng, config)
        latencies.append(collaborative_latency(deployment, channel, config).total_seconds)
    return float(np.mean(latencies))
