"""Synchronous A2C baseline for resource-constrained UAV layer assignment.

The environment exposes only a terminal deployment reward.  With the default
``gamma=1`` every layer action receives the same Monte-Carlo return, matching
the return convention used by the repository's layerwise PPO implementation.
This keeps the PPO-vs-A2C comparison focused on the update rule rather than on
reward shaping.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.algorithms.config import A2CConfig
from uav_rl.rl.layerwise_episode import LayerwiseEpisode
from uav_rl.rl.layerwise_policy import LayerwiseActorCritic
from uav_rl.wireless import sample_channel


class LayerwiseA2CTrainer:
    """Train a masked actor-critic with one synchronous update per rollout."""

    algorithm = "a2c"
    policy_type = "layerwise_general_assignment_a2c"

    def __init__(
        self,
        config: A2CConfig,
        resource_config: ResourceConstrainedConfig,
        environment: ResourceDeploymentEnvironment,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.resource_config = resource_config
        self.environment = environment
        self.device = torch.device(device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = LayerwiseActorCritic(resource_config, config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _rollout_one(
        self, normalized_channel: np.ndarray, *, deterministic: bool
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], np.ndarray]:
        episode = LayerwiseEpisode(
            normalized_channel,
            self.resource_config,
            max_boundaries=self.config.max_boundaries,
        )
        states: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        while not episode.done:
            state, mask = episode.observation_and_mask()
            with torch.no_grad():
                output = self.model.sample(
                    torch.from_numpy(state[None]).to(self.device),
                    torch.from_numpy(mask[None]).to(self.device),
                    deterministic=deterministic,
                )
            action = int(output.actions.item())
            states.append(state)
            masks.append(mask)
            actions.append(action)
            episode.step(action)
        return states, masks, actions, episode.completed_deployment()

    def deployments(self, channels: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        """Generate one feasible assignment per channel matrix."""

        normalized = self.environment.normalize_channels(channels)
        return np.stack(
            [self._rollout_one(channel, deterministic=deterministic)[-1] for channel in normalized]
        )

    def evaluate(self, channels: np.ndarray) -> dict[str, float]:
        """Evaluate the deterministic policy with the configured surrogate reward."""

        deployments = self.deployments(channels, deterministic=True)
        rewards, details = self.environment.evaluate(channels, deployments)
        return {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
            "latency_mean_seconds": float(details["latency_seconds"].mean()),
            "invalid_fraction": float(details["invalid"].mean()),
        }

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        with temporary.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def train(self, output_path: Path) -> dict[str, list[float]]:
        """Train A2C and save the best deterministic validation checkpoint."""

        channel_rng = np.random.default_rng(self.config.seed)
        validation_channels = generate_resource_channels(
            self.config.validation_channels,
            self.config.validation_seed,
            self.resource_config,
        )
        best_reward = float("-inf")
        best_episodes = 0
        best_state = deepcopy(self.model.state_dict())
        history: dict[str, list[float]] = {
            "episodes": [],
            "training_reward": [],
            "validation_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "invalid_fraction": [],
        }
        completed = 0
        rollout_index = 0
        while completed < self.config.training_episodes:
            rollout_size = min(
                self.config.rollout_size, self.config.training_episodes - completed
            )
            channels = np.stack(
                [sample_channel(channel_rng, self.resource_config.system) for _ in range(rollout_size)]
            ).astype(np.float32)
            normalized = self.environment.normalize_channels(channels)
            all_states: list[np.ndarray] = []
            all_masks: list[np.ndarray] = []
            all_actions: list[int] = []
            deployments: list[np.ndarray] = []
            for channel in normalized:
                states, masks, actions, deployment = self._rollout_one(
                    channel, deterministic=False
                )
                all_states.extend(states)
                all_masks.extend(masks)
                all_actions.extend(actions)
                deployments.append(deployment)
            rewards, details = self.environment.evaluate(channels, np.stack(deployments))

            states_tensor = torch.from_numpy(np.stack(all_states)).to(self.device)
            masks_tensor = torch.from_numpy(np.stack(all_masks)).to(self.device)
            actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=self.device)
            output = self.model.evaluate(states_tensor, masks_tensor, actions_tensor)

            layers = self.resource_config.system.num_layers
            # The terminal reward is discounted backward only when gamma < 1.
            discounts = torch.tensor(
                [self.config.gamma ** (layers - 1 - step) for step in range(layers)],
                dtype=torch.float32,
                device=self.device,
            )
            returns = (
                torch.from_numpy(rewards).to(self.device).repeat_interleave(layers)
                * discounts.repeat(rollout_size)
            )
            advantages = returns - output.values.detach()
            normalized_advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
            policy_loss = -(output.log_probabilities * normalized_advantages).mean()
            value_loss = torch.mean((output.values - returns) ** 2)
            entropy = output.entropy.mean()
            loss = (
                policy_loss
                + self.config.value_coefficient * value_loss
                - self.config.entropy_coefficient * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            completed += rollout_size
            rollout_index += 1
            validate_now = (
                rollout_index % self.config.validation_interval_rollouts == 0
                or completed == self.config.training_episodes
            )
            validation_reward = float("nan")
            if validate_now:
                validation_reward = self.evaluate(validation_channels)["reward_mean"]
                if validation_reward > best_reward:
                    best_reward = validation_reward
                    best_episodes = completed
                    best_state = deepcopy(self.model.state_dict())
            history["episodes"].append(float(completed))
            history["training_reward"].append(float(rewards.mean()))
            history["validation_reward"].append(validation_reward)
            history["policy_loss"].append(float(policy_loss.item()))
            history["value_loss"].append(float(value_loss.item()))
            history["entropy"].append(float(entropy.item()))
            history["invalid_fraction"].append(float(details["invalid"].mean()))
            print(
                f"algorithm=a2c episodes={completed} reward={rewards.mean():.5f} "
                f"validation={validation_reward:.5f}",
                flush=True,
            )

        self.model.load_state_dict(best_state)
        self._atomic_save(
            {
                "format_version": 1,
                "algorithm": self.algorithm,
                "policy_type": self.policy_type,
                "algorithm_config": asdict(self.config),
                "resource_config": self.resource_config.to_dict(),
                "model_state": best_state,
                "best_validation_reward": best_reward,
                "best_episodes": best_episodes,
                "history": history,
            },
            output_path,
        )
        return history
