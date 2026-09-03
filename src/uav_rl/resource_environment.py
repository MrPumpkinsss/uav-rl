"""Reward environment for the paper-style general layer assignment."""

from __future__ import annotations

import numpy as np

from uav_rl.quality import QualityEvaluator, SurrogateQualityEvaluator
from uav_rl.resource_assignment import (
    ResourceConstrainedConfig,
    layerwise_drop_probabilities,
    layerwise_latency,
)
from uav_rl.surrogate import (
    PPLSurrogate,
    PPLSurrogateEnsemble,
    SurrogateModel,
    TailGatedSurrogate,
    TailResidualSurrogate,
)
from uav_rl.wireless import sample_channel


class ResourceDeploymentEnvironment:
    """Evaluate arbitrary layer-to-UAV assignments with resource constraints."""

    def __init__(
        self,
        config: ResourceConstrainedConfig,
        quality_evaluator: QualityEvaluator | SurrogateModel,
        latency_reference: float,
    ) -> None:
        """初始化资源环境，并检查系统配置是否满足实验要求。"""
        if latency_reference <= 0.0:
            raise ValueError("latency_reference must be positive")
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

    @property
    def system(self):
        """返回当前实验使用的系统配置。"""
        return self.config.system

    def normalize_channels(self, channels: np.ndarray) -> np.ndarray:
        """将原始信道质量归一化为策略网络使用的状态范围。"""
        scale = self.system.channel_gain_max - self.system.channel_gain_min
        return ((np.asarray(channels) - self.system.channel_gain_min) / scale).astype(np.float32)

    def evaluate(
        self,
        channels: np.ndarray,
        deployments: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """评估一个 deployment 的资源约束、质量、时延和最终 reward。"""
        channels = np.asarray(channels, dtype=np.float32)
        deployments = np.asarray(deployments, dtype=np.int64)
        if channels.ndim != 3 or channels.shape[1:] != (self.system.num_uavs, self.system.num_uavs):
            raise ValueError("channels have the wrong shape")
        if deployments.shape != (len(channels), self.system.num_layers):
            raise ValueError("deployments have the wrong shape")
        drops: list[np.ndarray] = []
        latencies: list[float] = []
        invalid = np.zeros(len(channels), dtype=bool)
        for index, (channel, deployment) in enumerate(zip(channels, deployments, strict=True)):
            try:
                drops.append(layerwise_drop_probabilities(deployment, channel, self.config))
                latencies.append(layerwise_latency(deployment, channel, self.config).total_seconds)
            except ValueError:
                invalid[index] = True
                drops.append(np.zeros(self.system.num_layers - 1, dtype=np.float32))
                latencies.append(0.0)
        quality = np.full(len(channels), 100.0, dtype=np.float32)
        valid_indices = np.flatnonzero(~invalid)
        if len(valid_indices):
            quality[valid_indices] = self.quality_evaluator.evaluate(
                np.stack([drops[index] for index in valid_indices]),
                noise_seeds=noise_seeds,
            )
        latency_array = np.asarray(latencies, dtype=np.float32)
        normalized_latency = latency_array / self.latency_reference
        rewards = -(
            self.system.quality_weight * quality
            + (1.0 - self.system.quality_weight) * normalized_latency
        )
        rewards[invalid] = -100.0
        return rewards.astype(np.float32), {
            "log_ppl_ratio": quality,
            "latency_seconds": latency_array,
            "normalized_latency": normalized_latency,
            "drop_probabilities": np.stack(drops),
            "invalid": invalid,
        }


def generate_resource_channels(
    count: int, seed: int, config: ResourceConstrainedConfig
) -> np.ndarray:
    """Generate reproducible channel matrices for a general-assignment run."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return np.stack([sample_channel(rng, config.system) for _ in range(count)]).astype(np.float32)
