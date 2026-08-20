"""Regression tests for lossless rollout-boundary PPO continuation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.rl.environment import DeploymentEnvironment
from uav_rl.rl.ppo import PPOTrainer


class DeterministicQualityEvaluator:
    def evaluate(
        self,
        drop_probabilities: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        quality = np.asarray(drop_probabilities, dtype=np.float32).sum(axis=1)
        if noise_seeds is None:
            return quality
        seeds = np.asarray(noise_seeds)
        if seeds.ndim == 1:
            seed_effect = np.full(len(quality), np.mean(seeds % 17), dtype=np.float32)
        else:
            seed_effect = np.mean(seeds % 17, axis=1, dtype=np.float32)
        return quality + seed_effect / 100.0


def _config(training_episodes: int) -> PPOConfig:
    system = SystemConfig(
        num_layers=4,
        num_uavs=2,
        max_layers_per_uav=2,
        compute_speed=(1.0, 1.2),
    )
    return PPOConfig(
        system=system,
        hidden_dim=32,
        rollout_size=4,
        training_episodes=training_episodes,
        update_epochs=2,
        minibatch_size=2,
        teacher_channels=0,
        behavior_cloning_epochs=0,
        validation_channels=2,
        validation_interval=1,
        test_channels=2,
    )


def _trainer(config: PPOConfig) -> PPOTrainer:
    environment = DeploymentEnvironment(
        config.system,
        DeterministicQualityEvaluator(),
        latency_reference=1.0,
    )
    return PPOTrainer(config, environment)


def _structured_teacher(channels: np.ndarray) -> np.ndarray:
    return np.tile(np.array([0, 0, 1, 1], dtype=np.int64), (len(channels), 1))


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_resumed_training_matches_uninterrupted_training(tmp_path: Path) -> None:
    metadata = {"quality_backend": "deterministic-test"}
    uninterrupted_state = tmp_path / "uninterrupted-state.pth"
    uninterrupted = _trainer(_config(8))
    uninterrupted.train(
        tmp_path / "uninterrupted-best.pth",
        state_path=uninterrupted_state,
        run_metadata=metadata,
    )

    resumed_state = tmp_path / "resumed-state.pth"
    first_stage = _trainer(_config(4))
    first_stage.train(
        tmp_path / "resumed-best.pth",
        state_path=resumed_state,
        run_metadata=metadata,
    )
    second_stage = _trainer(replace(_config(4), training_episodes=8))
    second_stage.train(
        tmp_path / "resumed-best.pth",
        state_path=resumed_state,
        resume=True,
        run_metadata=metadata,
    )

    uninterrupted_payload = torch.load(uninterrupted_state, weights_only=False)
    resumed_payload = torch.load(resumed_state, weights_only=False)
    for key in (
        "model_state",
        "optimizer_state",
        "best_model_state",
        "best_validation_reward",
        "best_episodes",
        "completed_episodes",
        "rollout_index",
        "history",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "channel_rng_state",
        "noise_rng_state",
    ):
        _assert_nested_equal(uninterrupted_payload[key], resumed_payload[key])


def test_resume_rejects_changed_run_metadata(tmp_path: Path) -> None:
    state_path = tmp_path / "state.pth"
    trainer = _trainer(_config(4))
    trainer.train(
        tmp_path / "best.pth",
        state_path=state_path,
        run_metadata={"model": "first"},
    )

    resumed = _trainer(_config(8))
    try:
        resumed.train(
            tmp_path / "best.pth",
            state_path=state_path,
            resume=True,
            run_metadata={"model": "second"},
        )
    except ValueError as error:
        assert "metadata" in str(error)
    else:
        raise AssertionError("resume accepted incompatible run metadata")


def test_resume_rejects_changed_noise_configuration(tmp_path: Path) -> None:
    state_path = tmp_path / "state.pth"
    trainer = _trainer(_config(4))
    trainer.train(
        tmp_path / "best.pth",
        state_path=state_path,
    )

    changed = replace(_config(8), training_noise_samples=2)
    with np.testing.assert_raises_regex(ValueError, "configuration differs"):
        _trainer(changed).train(
            tmp_path / "best.pth",
            state_path=state_path,
            resume=True,
        )



def test_candidate_checkpoints_are_exported_for_external_validation(tmp_path: Path) -> None:
    trainer = _trainer(_config(8))
    candidate_directory = tmp_path / "candidates"
    trainer.train(
        tmp_path / "best.pth",
        state_path=tmp_path / "state.pth",
        candidate_checkpoint_directory=candidate_directory,
    )

    candidates = sorted(candidate_directory.glob("episode_*.pth"))
    assert [candidate.name for candidate in candidates] == [
        "episode_000004.pth",
        "episode_000008.pth",
    ]
    payload = torch.load(candidates[-1], weights_only=False)
    assert payload["purpose"] == "external_policy_validation_candidate"
    assert payload["episodes"] == 8


def test_candidate_checkpoints_can_follow_exact_episode_intervals(tmp_path: Path) -> None:
    trainer = _trainer(_config(8))
    candidate_directory = tmp_path / "candidates"
    trainer.train(
        tmp_path / "best.pth",
        state_path=tmp_path / "state.pth",
        candidate_checkpoint_directory=candidate_directory,
        candidate_checkpoint_interval_episodes=2,
    )

    candidates = sorted(candidate_directory.glob("episode_*.pth"))
    assert [candidate.name for candidate in candidates] == [
        "episode_000002.pth",
        "episode_000004.pth",
        "episode_000006.pth",
        "episode_000008.pth",
    ]


def test_behavior_cloning_uses_explicit_teacher_action_provider(tmp_path: Path) -> None:
    calls: list[np.ndarray] = []

    def provider(channels: np.ndarray) -> np.ndarray:
        calls.append(channels.copy())
        return np.tile(np.array([0, 0, 1, 1], dtype=np.int64), (len(channels), 1))

    config = replace(
        _config(4),
        teacher_channels=4,
        behavior_cloning_epochs=1,
    )
    environment = DeploymentEnvironment(
        config.system,
        DeterministicQualityEvaluator(),
        latency_reference=1.0,
    )
    trainer = PPOTrainer(config, environment, teacher_action_provider=provider)
    history = trainer.train(tmp_path / "best.pth")

    assert len(calls) == 1
    assert calls[0].shape == (4, config.system.num_uavs, config.system.num_uavs)
    assert len(history["behavior_cloning_loss"]) == 1


def test_teacher_anchored_ppo_records_relative_reward_and_online_bc(tmp_path: Path) -> None:
    config = replace(
        _config(4),
        teacher_relative_rewards=True,
        online_behavior_cloning_coefficient=0.1,
    )
    environment = DeploymentEnvironment(
        config.system,
        DeterministicQualityEvaluator(),
        latency_reference=1.0,
    )
    trainer = PPOTrainer(config, environment, teacher_action_provider=_structured_teacher)
    history = trainer.train(tmp_path / "best.pth")

    assert len(history["teacher_reward"]) == 1
    assert len(history["relative_reward"]) == 1
    assert len(history["online_behavior_cloning_loss"]) == 1
    assert history["online_behavior_cloning_loss"][0] > 0.0


def test_teacher_anchored_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    metadata = {"quality_backend": "deterministic-teacher-anchored-test"}
    full_config = replace(
        _config(8),
        teacher_relative_rewards=True,
        online_behavior_cloning_coefficient=0.1,
    )
    uninterrupted_state = tmp_path / "uninterrupted-anchored-state.pth"
    uninterrupted_environment = DeploymentEnvironment(
        full_config.system,
        DeterministicQualityEvaluator(),
        latency_reference=1.0,
    )
    PPOTrainer(
        full_config,
        uninterrupted_environment,
        teacher_action_provider=_structured_teacher,
    ).train(
        tmp_path / "uninterrupted-anchored-best.pth",
        state_path=uninterrupted_state,
        run_metadata=metadata,
    )

    resumed_state = tmp_path / "resumed-anchored-state.pth"
    first_config = replace(full_config, training_episodes=4)
    first_environment = DeploymentEnvironment(
        first_config.system,
        DeterministicQualityEvaluator(),
        latency_reference=1.0,
    )
    PPOTrainer(
        first_config,
        first_environment,
        teacher_action_provider=_structured_teacher,
    ).train(
        tmp_path / "resumed-anchored-best.pth",
        state_path=resumed_state,
        run_metadata=metadata,
    )

    resumed_environment = DeploymentEnvironment(
        full_config.system,
        DeterministicQualityEvaluator(),
        latency_reference=1.0,
    )
    PPOTrainer(
        full_config,
        resumed_environment,
        teacher_action_provider=_structured_teacher,
    ).train(
        tmp_path / "resumed-anchored-best.pth",
        state_path=resumed_state,
        resume=True,
        run_metadata=metadata,
    )

    uninterrupted_payload = torch.load(uninterrupted_state, weights_only=False)
    resumed_payload = torch.load(resumed_state, weights_only=False)
    _assert_nested_equal(uninterrupted_payload["model_state"], resumed_payload["model_state"])
    _assert_nested_equal(uninterrupted_payload["optimizer_state"], resumed_payload["optimizer_state"])
    _assert_nested_equal(uninterrupted_payload["history"], resumed_payload["history"])
