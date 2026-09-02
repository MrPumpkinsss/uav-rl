"""Tests for independent variable-length segment PPO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.rl.environment import DeploymentEnvironment
from uav_rl.rl.segment_policy import SegmentActorCritic, decode_action, valid_action_mask
from uav_rl.rl.segment_ppo import SegmentPPOOptions, SegmentPPOTrainer


class DeterministicQuality:
    def evaluate(
        self, drop_probabilities: np.ndarray, *, noise_seeds: np.ndarray | None = None
    ) -> np.ndarray:
        values = np.asarray(drop_probabilities, dtype=np.float32).sum(axis=1)
        if noise_seeds is not None:
            values += np.asarray(noise_seeds).mean(axis=-1, dtype=np.float32) % 5 / 100.0
        return values


def _config(episodes: int) -> PPOConfig:
    system = SystemConfig(
        num_layers=4, num_uavs=2, max_layers_per_uav=2, compute_speed=(1.0, 1.2)
    )
    return PPOConfig(
        system=system,
        hidden_dim=16,
        rollout_size=2,
        training_episodes=episodes,
        update_epochs=1,
        minibatch_size=2,
        teacher_channels=0,
        behavior_cloning_epochs=0,
        validation_channels=2,
        validation_interval=1,
        test_channels=2,
    )


def _trainer(config: PPOConfig) -> SegmentPPOTrainer:
    return SegmentPPOTrainer(
        config, DeploymentEnvironment(config.system, DeterministicQuality(), latency_reference=1.0)
    )


def test_mask_only_permits_feasible_segments_and_policy_respects_it() -> None:
    config = _config(2).system
    mask = valid_action_mask(
        assigned_layers=0, used_uavs=np.zeros(config.num_uavs, dtype=bool), config=config
    )
    assert mask.sum() == 2
    policy = SegmentActorCritic(config, hidden_dim=16)
    state = torch.zeros((8, config.num_uavs**2 + 1 + 2 * config.num_uavs))
    sampled = policy.sample(state, torch.from_numpy(np.tile(mask, (8, 1))))
    for action in sampled.actions.tolist():
        assert mask[action]
        _, length = decode_action(action, config)
        assert length == 2


def test_fixed_capacity_conservative_ppo_keeps_only_full_segments(tmp_path: Path) -> None:
    config = _config(2)

    def teacher(channels: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0, 0, 1, 1], dtype=np.int64), (len(channels), 1))

    trainer = SegmentPPOTrainer(
        config,
        DeploymentEnvironment(config.system, DeterministicQuality(), latency_reference=1.0),
        teacher,
        SegmentPPOOptions(
            fixed_capacity_segments=True,
            positive_reference_improvement=True,
            reference_kl_coefficient=0.05,
        ),
    )
    history = trainer.train(
        tmp_path / "best.pth",
        state_path=tmp_path / "state.pth",
        resume=False,
        run_metadata={"test": "conservative"},
        candidate_directory=tmp_path / "candidates",
        candidate_interval_episodes=2,
    )
    assert history["mean_segments"] == [2.0]
    assert len(history["reference_reward"]) == 1
    assert 0.0 <= history["positive_improvement_fraction"][0] <= 1.0


def test_segment_ppo_exports_exact_candidates_and_resumes_losslessly(tmp_path: Path) -> None:
    metadata = {"test": "segment-resume"}
    full_state = tmp_path / "full-state.pth"
    full_candidates = tmp_path / "full-candidates"
    _trainer(_config(4)).train(
        tmp_path / "full-best.pth",
        state_path=full_state,
        resume=False,
        run_metadata=metadata,
        candidate_directory=full_candidates,
        candidate_interval_episodes=2,
    )
    resumed_state = tmp_path / "resumed-state.pth"
    resumed_candidates = tmp_path / "resumed-candidates"
    _trainer(_config(2)).train(
        tmp_path / "resumed-best.pth",
        state_path=resumed_state,
        resume=False,
        run_metadata=metadata,
        candidate_directory=resumed_candidates,
        candidate_interval_episodes=2,
    )
    _trainer(replace(_config(2), training_episodes=4)).train(
        tmp_path / "resumed-best.pth",
        state_path=resumed_state,
        resume=True,
        run_metadata=metadata,
        candidate_directory=resumed_candidates,
        candidate_interval_episodes=2,
    )
    full = torch.load(full_state, weights_only=False)
    resumed = torch.load(resumed_state, weights_only=False)
    assert full["history"] == resumed["history"]
    for name, value in full["model_state"].items():
        assert torch.equal(value, resumed["model_state"][name])
    assert sorted(path.name for path in resumed_candidates.glob("*.pth")) == [
        "episode_000000.pth",
        "episode_000002.pth",
        "episode_000004.pth",
    ]
