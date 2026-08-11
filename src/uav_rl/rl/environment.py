"""Fast surrogate-backed reward environment for PPO."""

from __future__ import annotations

import numpy as np
import torch

from uav_rl.config import SystemConfig
from uav_rl.surrogate import PPLSurrogate
from uav_rl.wireless import (
    boundary_drop_probabilities,
    collaborative_latency,
    sample_channel,
)


class DeploymentEnvironment:
    """Evaluate deployment quality without invoking the large language model."""

    def __init__(
        self,
        config: SystemConfig,
        surrogate: PPLSurrogate,
        latency_reference: float,
    ) -> None:
        self.config = config
        self.surrogate = surrogate.eval()
        self.latency_reference = latency_reference

    def normalize_channels(self, channels: np.ndarray) -> np.ndarray:
        scale = self.config.channel_gain_max - self.config.channel_gain_min
        return ((channels - self.config.channel_gain_min) / scale).astype(np.float32)

    def evaluate(
        self, channels: np.ndarray, deployments: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        qualities: list[float] = []
        latencies: list[float] = []
        drops: list[np.ndarray] = []
        for channel, deployment in zip(channels, deployments, strict=True):
            drop = boundary_drop_probabilities(deployment, channel, self.config)
            drops.append(drop)
            latencies.append(collaborative_latency(deployment, channel, self.config).total_seconds)
        surrogate_device = next(self.surrogate.parameters()).device
        drop_tensor = torch.from_numpy(np.stack(drops)).float().to(surrogate_device)
        with torch.no_grad():
            qualities = self.surrogate(drop_tensor).clamp_min(0.0).cpu().numpy().tolist()
        quality_array = np.asarray(qualities, dtype=np.float32)
        latency_array = np.asarray(latencies, dtype=np.float32)
        normalized_latency = latency_array / self.latency_reference
        rewards = -(
            self.config.quality_weight * quality_array
            + (1.0 - self.config.quality_weight) * normalized_latency
        )
        return rewards.astype(np.float32), {
            "log_ppl_ratio": quality_array,
            "latency_seconds": latency_array,
            "normalized_latency": normalized_latency,
            "drop_probabilities": np.stack(drops),
        }


def generate_channels(count: int, seed: int, config: SystemConfig) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([sample_channel(rng, config) for _ in range(count)]).astype(np.float32)
