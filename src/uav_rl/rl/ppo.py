"""PPO optimization for one-step channel-to-deployment decisions."""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import PPOConfig
from uav_rl.noise_seeds import sample_training_noise_seeds, validation_noise_seeds
from uav_rl.rl.environment import DeploymentEnvironment, generate_channels
from uav_rl.rl.oracle import LocalFourSegmentQualityModel, reward_oracle_deployments
from uav_rl.rl.policy import ContinuousDeploymentActorCritic


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        environment: DeploymentEnvironment,
        teacher_quality_model: LocalFourSegmentQualityModel | None = None,
        require_ppo_checkpoint: bool = False,
        teacher_action_provider: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self.environment = environment
        self.teacher_quality_model = teacher_quality_model
        self.require_ppo_checkpoint = require_ppo_checkpoint
        self.teacher_action_provider = teacher_action_provider
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = ContinuousDeploymentActorCritic(config.system, config.hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.validation_noise_seeds = validation_noise_seeds(
            config.validation_noise_samples,
            config.validation_noise_seed,
        )

    def load_initial_policy(self, checkpoint_path: Path) -> None:
        """Resume optimization from a compatible actor-critic checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])

    def _states(self, channels: np.ndarray) -> torch.Tensor:
        normalized = self.environment.normalize_channels(channels)
        return torch.from_numpy(normalized.reshape(len(channels), -1))

    def _validation_reward(self, channels: np.ndarray) -> float:
        with torch.no_grad():
            actions = self.model.sample(self._states(channels), deterministic=True).actions
        rewards, _ = self.environment.evaluate(
            channels,
            actions.cpu().numpy(),
            noise_seeds=self.validation_noise_seeds,
        )
        return float(rewards.mean())

    def _teacher_deployments(self, channels: np.ndarray) -> np.ndarray:
        """Return validated teacher actions for behavior cloning or PPO anchoring."""

        if self.teacher_action_provider is None:
            deployments = reward_oracle_deployments(
                channels,
                self.environment,
                quality_model=self.teacher_quality_model,
            )
        else:
            deployments = np.asarray(self.teacher_action_provider(channels), dtype=np.int64)
        expected_shape = (len(channels), self.config.system.num_layers)
        if deployments.shape != expected_shape:
            raise ValueError(
                "teacher action provider returned shape "
                f"{deployments.shape}, expected {expected_shape}"
            )
        return deployments

    def _behavior_clone(self) -> list[float]:
        if self.config.teacher_channels == 0 or self.config.behavior_cloning_epochs == 0:
            return []
        channels = generate_channels(
            self.config.teacher_channels, self.config.teacher_seed, self.config.system
        )
        states = self._states(channels)
        deployments = self._teacher_deployments(channels)
        teacher_actions = torch.from_numpy(deployments)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.behavior_cloning_learning_rate
        )
        losses: list[float] = []
        for epoch in range(self.config.behavior_cloning_epochs):
            order = torch.randperm(len(channels))
            total_loss = 0.0
            for start in range(0, len(channels), self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                output = self.model.evaluate(states[indices], teacher_actions[indices])
                loss = -output.log_probabilities.mean() / self.config.system.num_layers
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                total_loss += float(loss.item()) * len(indices)
            mean_loss = total_loss / len(channels)
            losses.append(mean_loss)
            print(
                f"behavior_cloning_epoch={epoch + 1:3d} loss={mean_loss:.6f}",
                flush=True,
            )
        return losses

    def _save_checkpoint(self, output_path: Path, validation_reward: float, episodes: int) -> None:
        self._atomic_save(
            {
                "model_state": self.model.state_dict(),
                "ppo_config": asdict(self.config),
                "best_validation_reward": validation_reward,
                "episodes_at_best": episodes,
            },
            output_path,
        )

    def _save_candidate_checkpoint(
        self,
        directory: Path,
        *,
        monitor_reward: float,
        episodes: int,
    ) -> Path:
        """Persist a frozen policy candidate for an external validation oracle.

        The monitor reward comes from the training environment. It is metadata
        only: callers using a surrogate environment must not treat it as the
        real-model policy-selection metric.
        """

        path = directory / f"episode_{episodes:06d}.pth"
        self._atomic_save(
            {
                "format_version": 1,
                "purpose": "external_policy_validation_candidate",
                "model_state": self.model.state_dict(),
                "ppo_config": asdict(self.config),
                "episodes": episodes,
                "training_monitor_reward": monitor_reward,
            },
            path,
        )
        return path

    @staticmethod
    def _atomic_save(payload: dict[str, Any], output_path: Path) -> None:
        """Replace a checkpoint only after its complete temporary file is durable."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        with temporary_path.open("ab") as checkpoint_file:
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        temporary_path.replace(output_path)

    def _save_training_state(
        self,
        state_path: Path,
        *,
        completed: int,
        rollout_index: int,
        best_validation_reward: float,
        best_episodes: int,
        best_model_state: dict[str, torch.Tensor],
        checkpoint_available: bool,
        history: dict[str, list[float]],
        channel_rng: np.random.Generator,
        noise_rng: np.random.Generator,
        run_metadata: dict[str, Any],
    ) -> None:
        self._atomic_save(
            {
                "format_version": 2,
                "ppo_config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_model_state": best_model_state,
                "best_validation_reward": best_validation_reward,
                "best_episodes": best_episodes,
                "checkpoint_available": checkpoint_available,
                "completed_episodes": completed,
                "rollout_index": rollout_index,
                "history": history,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_states": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "channel_rng_state": channel_rng.bit_generator.state,
                "noise_rng_state": noise_rng.bit_generator.state,
                "run_metadata": run_metadata,
            },
            state_path,
        )

    def _load_training_state(
        self,
        state_path: Path,
        expected_run_metadata: dict[str, Any],
    ) -> tuple[
        int,
        int,
        float,
        int,
        dict[str, torch.Tensor],
        bool,
        dict[str, list[float]],
        np.random.Generator,
        np.random.Generator,
    ]:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 2:
            raise ValueError("unsupported PPO training-state format")
        if state.get("run_metadata") != expected_run_metadata:
            raise ValueError("resume run metadata differs from the saved experiment")
        saved_config = dict(state["ppo_config"])
        current_config = asdict(self.config)
        saved_target = saved_config.pop("training_episodes")
        current_target = current_config.pop("training_episodes")
        if saved_config != current_config:
            raise ValueError(
                "resume configuration differs from the saved run; only training_episodes may change"
            )
        completed = int(state["completed_episodes"])
        if current_target < completed:
            raise ValueError(
                f"training_episodes={current_target} is below completed_episodes={completed}"
            )
        if current_target < saved_target:
            raise ValueError("training_episodes cannot be reduced when resuming")

        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and state["cuda_random_states"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_random_states"])
        channel_rng = np.random.default_rng()
        channel_rng.bit_generator.state = state["channel_rng_state"]
        noise_rng = np.random.default_rng()
        noise_rng.bit_generator.state = state["noise_rng_state"]
        return (
            completed,
            int(state["rollout_index"]),
            float(state["best_validation_reward"]),
            int(state["best_episodes"]),
            state["best_model_state"],
            bool(state["checkpoint_available"]),
            state["history"],
            channel_rng,
            noise_rng,
        )

    def train(
        self,
        output_path: Path,
        *,
        state_path: Path | None = None,
        resume: bool = False,
        run_metadata: dict[str, Any] | None = None,
        candidate_checkpoint_directory: Path | None = None,
        candidate_checkpoint_interval_episodes: int | None = None,
    ) -> dict[str, list[float]]:
        """Train PPO and optionally persist a lossless rollout-boundary state."""

        if resume and state_path is None:
            raise ValueError("state_path is required when resume=True")
        run_metadata = {} if run_metadata is None else run_metadata
        uses_teacher_anchor = (
            self.config.teacher_relative_rewards
            or self.config.online_behavior_cloning_coefficient > 0.0
        )
        if uses_teacher_anchor and self.teacher_action_provider is None:
            raise ValueError(
                "teacher-relative PPO requires an explicit deterministic teacher action provider"
            )
        if candidate_checkpoint_interval_episodes is not None:
            if candidate_checkpoint_interval_episodes < 1:
                raise ValueError("candidate checkpoint interval must be positive")
            if candidate_checkpoint_directory is None:
                raise ValueError("candidate checkpoint directory is required for a checkpoint interval")
        if candidate_checkpoint_directory is not None:
            candidate_checkpoint_directory.mkdir(parents=True, exist_ok=True)
        validation_channels = generate_channels(
            self.config.validation_channels,
            self.config.validation_seed,
            self.config.system,
        )
        if resume:
            assert state_path is not None
            (
                completed,
                rollout_index,
                best_validation_reward,
                best_episodes,
                best_model_state,
                checkpoint_available,
                history,
                rng,
                noise_rng,
            ) = self._load_training_state(state_path, run_metadata)
            print(
                f"resumed_episodes={completed} target_episodes={self.config.training_episodes} "
                f"best_validation_reward={best_validation_reward:.6f}",
                flush=True,
            )
        else:
            history = {
                "behavior_cloning_loss": self._behavior_clone(),
                "reward": [],
                "latency": [],
                "log_ppl_ratio": [],
                "validation_reward": [],
            }
            if uses_teacher_anchor:
                history.update(
                    {
                        "teacher_reward": [],
                        "relative_reward": [],
                        "online_behavior_cloning_loss": [],
                    }
                )
            best_validation_reward = self._validation_reward(validation_channels)
            best_episodes = 0
            best_model_state = deepcopy(self.model.state_dict())
            checkpoint_available = not self.require_ppo_checkpoint
            if checkpoint_available:
                self._save_checkpoint(output_path, best_validation_reward, episodes=0)
            print(f"initial_validation_reward={best_validation_reward:.6f}", flush=True)
            completed = 0
            rollout_index = 0
            rng = np.random.default_rng(self.config.seed)
            noise_rng = np.random.default_rng(self.config.training_noise_seed)
            if state_path is not None:
                self._save_training_state(
                    state_path,
                    completed=completed,
                    rollout_index=rollout_index,
                    best_validation_reward=best_validation_reward,
                    best_episodes=best_episodes,
                    best_model_state=best_model_state,
                    checkpoint_available=checkpoint_available,
                    history=history,
                    channel_rng=rng,
                    noise_rng=noise_rng,
                    run_metadata=run_metadata,
                )

        while completed < self.config.training_episodes:
            rollout_size = min(self.config.rollout_size, self.config.training_episodes - completed)
            if candidate_checkpoint_interval_episodes is not None:
                next_candidate = (
                    (completed // candidate_checkpoint_interval_episodes) + 1
                ) * candidate_checkpoint_interval_episodes
                rollout_size = min(rollout_size, next_candidate - completed)
            channel_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            channels = generate_channels(rollout_size, channel_seed, self.config.system)
            states = self._states(channels)
            with torch.no_grad():
                old_output = self.model.sample(states)
            deployments = old_output.actions.cpu().numpy()
            training_noise_seeds = sample_training_noise_seeds(
                noise_rng,
                rollout_size,
                self.config.training_noise_samples,
            )
            rewards_np, details = self.environment.evaluate(
                channels,
                deployments,
                noise_seeds=training_noise_seeds,
            )
            rewards = torch.from_numpy(rewards_np)
            teacher_actions: torch.Tensor | None = None
            teacher_rewards_np: np.ndarray | None = None
            if uses_teacher_anchor:
                teacher_actions = torch.from_numpy(self._teacher_deployments(channels))
                if self.config.teacher_relative_rewards:
                    teacher_rewards_np, _ = self.environment.evaluate(
                        channels,
                        teacher_actions.numpy(),
                        noise_seeds=training_noise_seeds,
                    )
            returns = (
                rewards
                if teacher_rewards_np is None
                else rewards - torch.from_numpy(teacher_rewards_np)
            )
            advantages = returns - old_output.values
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            online_bc_total = 0.0
            online_bc_samples = 0
            for _ in range(self.config.update_epochs):
                order = torch.randperm(rollout_size)
                for start in range(0, rollout_size, self.config.minibatch_size):
                    indices = order[start : start + self.config.minibatch_size]
                    output = self.model.evaluate(states[indices], old_output.actions[indices])
                    ratio = torch.exp(
                        output.log_probabilities - old_output.log_probabilities[indices]
                    )
                    advantage = normalized_advantages[indices]
                    unclipped = ratio * advantage
                    clipped = (
                        torch.clamp(
                            ratio,
                            1.0 - self.config.clip_epsilon,
                            1.0 + self.config.clip_epsilon,
                        )
                        * advantage
                    )
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = torch.mean((output.values - returns[indices]) ** 2)
                    entropy_bonus = output.entropy.mean() / self.config.system.num_layers
                    loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy_bonus
                    )
                    if self.config.online_behavior_cloning_coefficient > 0.0:
                        assert teacher_actions is not None
                        teacher_output = self.model.evaluate(
                            states[indices], teacher_actions[indices]
                        )
                        online_bc_loss = (
                            -teacher_output.log_probabilities.mean()
                            / self.config.system.num_layers
                        )
                        loss = (
                            loss
                            + self.config.online_behavior_cloning_coefficient * online_bc_loss
                        )
                        online_bc_total += float(online_bc_loss.item()) * len(indices)
                        online_bc_samples += len(indices)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

            completed += rollout_size
            rollout_index += 1
            history["reward"].append(float(rewards_np.mean()))
            history["latency"].append(float(details["latency_seconds"].mean()))
            history["log_ppl_ratio"].append(float(details["log_ppl_ratio"].mean()))
            if uses_teacher_anchor:
                assert teacher_actions is not None
                history["online_behavior_cloning_loss"].append(
                    online_bc_total / online_bc_samples if online_bc_samples else 0.0
                )
                if teacher_rewards_np is not None:
                    history["teacher_reward"].append(float(teacher_rewards_np.mean()))
                    history["relative_reward"].append(
                        float((rewards_np - teacher_rewards_np).mean())
                    )
            print(
                f"episodes={completed:5d} reward={rewards_np.mean():.4f} "
                f"latency={details['latency_seconds'].mean():.4f}s "
                f"log_ppl_ratio={details['log_ppl_ratio'].mean():.4f}",
                flush=True,
            )
            if teacher_rewards_np is not None:
                print(
                    f"teacher_reward={teacher_rewards_np.mean():.4f} "
                    f"relative_reward={(rewards_np - teacher_rewards_np).mean():.4f} "
                    f"online_bc_loss={online_bc_total / online_bc_samples:.6f}",
                    flush=True,
                )

            candidate_due = (
                candidate_checkpoint_interval_episodes is not None
                and completed % candidate_checkpoint_interval_episodes == 0
            )
            if (
                rollout_index % self.config.validation_interval == 0
                or completed == self.config.training_episodes
                or candidate_due
            ):
                validation_reward = self._validation_reward(validation_channels)
                history["validation_reward"].append(validation_reward)
                print(
                    f"validation_reward={validation_reward:.6f} best={best_validation_reward:.6f}",
                    flush=True,
                )
                should_export_candidate = (
                    candidate_checkpoint_directory is not None
                    and completed > 0
                    and (
                        candidate_checkpoint_interval_episodes is None
                        or candidate_due
                    )
                )
                if should_export_candidate:
                    candidate_path = self._save_candidate_checkpoint(
                        candidate_checkpoint_directory,
                        monitor_reward=validation_reward,
                        episodes=completed,
                    )
                    print(f"candidate_checkpoint={candidate_path}", flush=True)
                if not checkpoint_available or validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    best_episodes = completed
                    best_model_state = deepcopy(self.model.state_dict())
                    self._save_checkpoint(output_path, best_validation_reward, episodes=completed)
                    checkpoint_available = True

            if state_path is not None:
                self._save_training_state(
                    state_path,
                    completed=completed,
                    rollout_index=rollout_index,
                    best_validation_reward=best_validation_reward,
                    best_episodes=best_episodes,
                    best_model_state=best_model_state,
                    checkpoint_available=checkpoint_available,
                    history=history,
                    channel_rng=rng,
                    noise_rng=noise_rng,
                    run_metadata=run_metadata,
                )

        if not checkpoint_available:
            raise RuntimeError("no PPO-updated checkpoint was evaluated")
        self.model.load_state_dict(best_model_state)
        self._save_checkpoint(output_path, best_validation_reward, episodes=best_episodes)
        return history
