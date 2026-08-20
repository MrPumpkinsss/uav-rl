from __future__ import annotations

import numpy as np

from uav_rl.config import SystemConfig
from uav_rl.evaluation import evaluate_methods
from uav_rl.rl.environment import DeploymentEnvironment, generate_channels
from uav_rl.rl.policy import ContinuousDeploymentActorCritic


class RecordingQualityEvaluator:
    def __init__(self) -> None:
        self.received_seeds: list[np.ndarray] = []

    def evaluate(
        self,
        drop_probabilities: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        assert noise_seeds is not None
        self.received_seeds.append(np.asarray(noise_seeds).copy())
        return np.zeros(len(drop_probabilities), dtype=np.float32)


def test_methods_share_the_same_held_out_noise_seeds() -> None:
    config = SystemConfig(
        num_layers=4,
        num_uavs=2,
        max_layers_per_uav=2,
        compute_speed=(1.0, 1.2),
    )
    evaluator = RecordingQualityEvaluator()
    environment = DeploymentEnvironment(config, evaluator, latency_reference=1.0)
    policy = ContinuousDeploymentActorCritic(config, hidden_dim=16)
    channels = generate_channels(3, 10, config)
    held_out_seeds = np.asarray([1_500_000_001, 1_500_000_002], dtype=np.int64)

    evaluate_methods(
        environment,
        policy,
        channels,
        clean_perplexity=10.0,
        random_seed=11,
        method_names=("ppo", "random"),
        noise_seeds=held_out_seeds,
    )

    assert len(evaluator.received_seeds) == 2
    assert all(np.array_equal(seeds, held_out_seeds) for seeds in evaluator.received_seeds)
