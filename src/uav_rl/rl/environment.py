"""Reward environment for channel-conditioned UAV deployment."""

from __future__ import annotations

import numpy as np

from uav_rl.config import SystemConfig
from uav_rl.quality import QualityEvaluator, SurrogateQualityEvaluator
from uav_rl.surrogate import (
    PPLSurrogate,
    PPLSurrogateEnsemble,
    SurrogateModel,
    TailGatedSurrogate,
    TailResidualSurrogate,
)
from uav_rl.wireless import (
    boundary_drop_probabilities,
    collaborative_latency,
    sample_channel,
)


class DeploymentEnvironment:
    """Combine a pluggable quality backend with analytical deployment latency."""

    def __init__(
        self,
        config: SystemConfig,
        quality_evaluator: QualityEvaluator | SurrogateModel,
        latency_reference: float,
    ) -> None:
        self.config = config
        self.quality_evaluator: QualityEvaluator = (
            SurrogateQualityEvaluator(quality_evaluator)
            if isinstance(
                quality_evaluator,
                (PPLSurrogate, PPLSurrogateEnsemble, TailGatedSurrogate, TailResidualSurrogate),
            )
            else quality_evaluator
        )
        self.latency_reference = latency_reference

    def normalize_channels(self, channels: np.ndarray) -> np.ndarray:
        scale = self.config.channel_gain_max - self.config.channel_gain_min
        return ((channels - self.config.channel_gain_min) / scale).astype(np.float32)

    def evaluate(
        self,
        channels: np.ndarray,
        deployments: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        latencies: list[float] = []
        drops: list[np.ndarray] = []
        for channel, deployment in zip(channels, deployments, strict=True):
            drop = boundary_drop_probabilities(deployment, channel, self.config)
            drops.append(drop)
            latencies.append(collaborative_latency(deployment, channel, self.config).total_seconds)
        quality_array = self.quality_evaluator.evaluate(
            np.stack(drops),
            noise_seeds=noise_seeds,
        )
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
