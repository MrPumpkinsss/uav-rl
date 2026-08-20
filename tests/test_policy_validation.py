from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.policy_validation import evaluate_policy_candidates, load_policy_candidate
from uav_rl.rl.environment import DeploymentEnvironment
from uav_rl.rl.policy import ContinuousDeploymentActorCritic


class DeterministicQualityEvaluator:
    def evaluate(
        self,
        drop_probabilities: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        del noise_seeds
        return np.asarray(drop_probabilities, dtype=np.float32).sum(axis=1)


def _candidate(path: Path, system: SystemConfig, preferred_uav: int, episodes: int) -> None:
    policy = ContinuousDeploymentActorCritic(system, hidden_dim=16)
    for parameter in policy.parameters():
        parameter.data.zero_()
    policy.actor_head.bias.data[preferred_uav] = 1.0
    torch.save(
        {
            "model_state": policy.state_dict(),
            "ppo_config": asdict(PPOConfig(system=system, hidden_dim=16)),
            "episodes": episodes,
        },
        path,
    )


def test_true_validation_evaluates_frozen_candidates_on_common_inputs(tmp_path: Path) -> None:
    system = SystemConfig(
        num_layers=4,
        num_uavs=2,
        max_layers_per_uav=2,
        compute_speed=(1.0, 1.2),
    )
    first = tmp_path / "episode_000004.pth"
    second = tmp_path / "episode_000008.pth"
    _candidate(first, system, preferred_uav=0, episodes=4)
    _candidate(second, system, preferred_uav=1, episodes=8)
    environment = DeploymentEnvironment(system, DeterministicQualityEvaluator(), 1.0)
    channels = np.full((3, 2, 2), 4.0, dtype=np.float32)
    result = evaluate_policy_candidates(
        environment=environment,
        candidate_paths=[first, second],
        channels=channels,
        noise_seeds=np.array([1, 2], dtype=np.int64),
    )

    assert len(result["candidates"]) == 2
    assert result["selected"]["path"] in {str(first), str(second)}
    assert result["selected"]["metrics"]["reward_95_ci_low"] <= result["selected"]["metrics"]["reward_mean"]
    assert load_policy_candidate(first, system).sample(
        torch.zeros(1, system.num_uavs * system.num_uavs), deterministic=True
    ).actions.shape == (1, system.num_layers)
