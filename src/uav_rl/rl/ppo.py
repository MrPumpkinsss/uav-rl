"""PPO optimization for one-step channel-to-deployment decisions."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from uav_rl.config import PPOConfig
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
    ) -> None:
        self.config = config
        self.environment = environment
        self.teacher_quality_model = teacher_quality_model
        self.require_ppo_checkpoint = require_ppo_checkpoint
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = ContinuousDeploymentActorCritic(config.system, config.hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

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
        rewards, _ = self.environment.evaluate(channels, actions.cpu().numpy())
        return float(rewards.mean())

    def _behavior_clone(self) -> list[float]:
        if self.config.teacher_channels == 0 or self.config.behavior_cloning_epochs == 0:
            return []
        channels = generate_channels(
            self.config.teacher_channels, self.config.teacher_seed, self.config.system
        )
        states = self._states(channels)
        teacher_actions = torch.from_numpy(
            reward_oracle_deployments(
                channels,
                self.environment,
                quality_model=self.teacher_quality_model,
            )
        )
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                optimizer.step()
                total_loss += float(loss.item()) * len(indices)
            mean_loss = total_loss / len(channels)
            losses.append(mean_loss)
            print(
                f"behavior_cloning_epoch={epoch + 1:3d} loss={mean_loss:.6f}",
                flush=True,
            )
        return losses

    def _save_checkpoint(
        self, output_path: Path, validation_reward: float, episodes: int
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "ppo_config": asdict(self.config),
                "best_validation_reward": validation_reward,
                "episodes_at_best": episodes,
            },
            output_path,
        )

    def train(self, output_path: Path) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "behavior_cloning_loss": self._behavior_clone(),
            "reward": [],
            "latency": [],
            "log_ppl_ratio": [],
            "validation_reward": [],
        }
        validation_channels = generate_channels(
            self.config.validation_channels,
            self.config.validation_seed,
            self.config.system,
        )
        best_validation_reward = self._validation_reward(validation_channels)
        checkpoint_available = not self.require_ppo_checkpoint
        if checkpoint_available:
            self._save_checkpoint(output_path, best_validation_reward, episodes=0)
        print(
            f"initial_validation_reward={best_validation_reward:.6f}", flush=True
        )
        completed = 0
        rollout_index = 0
        rng = np.random.default_rng(self.config.seed)
        while completed < self.config.training_episodes:
            rollout_size = min(
                self.config.rollout_size, self.config.training_episodes - completed
            )
            channel_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            channels = generate_channels(rollout_size, channel_seed, self.config.system)
            states = self._states(channels)
            with torch.no_grad():
                old_output = self.model.sample(states)
            deployments = old_output.actions.cpu().numpy()
            rewards_np, details = self.environment.evaluate(channels, deployments)
            rewards = torch.from_numpy(rewards_np)
            advantages = rewards - old_output.values
            returns = rewards
            normalized_advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

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
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * advantage
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = torch.mean((output.values - returns[indices]) ** 2)
                    entropy_bonus = output.entropy.mean() / self.config.system.num_layers
                    loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy_bonus
                    )
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
            print(
                f"episodes={completed:5d} reward={rewards_np.mean():.4f} "
                f"latency={details['latency_seconds'].mean():.4f}s "
                f"log_ppl_ratio={details['log_ppl_ratio'].mean():.4f}",
                flush=True,
            )

            if (
                rollout_index % self.config.validation_interval == 0
                or completed == self.config.training_episodes
            ):
                validation_reward = self._validation_reward(validation_channels)
                history["validation_reward"].append(validation_reward)
                print(
                    f"validation_reward={validation_reward:.6f} "
                    f"best={best_validation_reward:.6f}",
                    flush=True,
                )
                if not checkpoint_available or validation_reward > best_validation_reward:
                    best_validation_reward = validation_reward
                    self._save_checkpoint(
                        output_path, best_validation_reward, episodes=completed
                    )
                    checkpoint_available = True

        if not checkpoint_available:
            raise RuntimeError("no PPO-updated checkpoint was evaluated")
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        return history
