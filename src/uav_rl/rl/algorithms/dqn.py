"""Masked Double-DQN baseline for resource-constrained UAV layer assignment.

DQN is included because reviewers commonly expect a value-based discrete-action
baseline.  The code uses Double-DQN targets and action masking; vanilla one-shot
DQN would be inappropriate because the complete assignment space contains
``num_uavs ** num_layers`` actions.  We therefore use the same 32-step MDP and
feasible UAV masks as the layerwise PPO/A2C policies.
"""

from __future__ import annotations

import os
import random
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.algorithms.config import DQNConfig
from uav_rl.rl.algorithms.networks import LayerwiseQNetwork
from uav_rl.rl.layerwise_episode import LayerwiseEpisode
from uav_rl.wireless import sample_channel


@dataclass(frozen=True)
class ReplayTransition:
    """One masked transition; only the final layer receives environment reward."""

    state: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class LayerwiseDQNTrainer:
    """Train a replay-based Double-DQN on the sequential assignment MDP."""

    algorithm = "dqn"
    policy_type = "layerwise_general_assignment_dqn"

    def __init__(
        self,
        config: DQNConfig,
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
        self.online = LayerwiseQNetwork(resource_config, config.hidden_dim).to(self.device)
        self.target = LayerwiseQNetwork(resource_config, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: deque[ReplayTransition] = deque(maxlen=config.replay_capacity)
        self.gradient_steps = 0

    def epsilon(self, completed_episodes: int) -> float:
        """Linear exploration schedule measured in complete deployments."""

        fraction = min(1.0, completed_episodes / self.config.epsilon_decay_episodes)
        return float(
            self.config.epsilon_start
            + fraction * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def _select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        valid = np.flatnonzero(mask)
        if rng.random() < epsilon:
            return int(rng.choice(valid))
        with torch.no_grad():
            values = self.online.masked_q_values(
                torch.from_numpy(state[None]).to(self.device),
                torch.from_numpy(mask[None]).to(self.device),
            )
        return int(values.argmax(dim=1).item())

    def _episode(
        self,
        normalized_channel: np.ndarray,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, bool]]]:
        episode = LayerwiseEpisode(
            normalized_channel,
            self.resource_config,
            max_boundaries=self.config.max_boundaries,
        )
        raw: list[tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, bool]] = []
        while not episode.done:
            state, mask = episode.observation_and_mask()
            action = self._select_action(state, mask, epsilon=epsilon, rng=rng)
            episode.step(action)
            done = episode.done
            if done:
                next_state = np.zeros_like(state)
                next_mask = np.zeros_like(mask)
            else:
                next_state, next_mask = episode.observation_and_mask()
            raw.append((state, mask, action, next_state, next_mask, done))
        return episode.completed_deployment(), raw

    def deployments(self, channels: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        """Generate greedy assignments; ``deterministic=False`` uses current epsilon."""

        normalized = self.environment.normalize_channels(channels)
        rng = np.random.default_rng(self.config.seed + 999_983)
        epsilon = 0.0 if deterministic else self.config.epsilon_end
        return np.stack(
            [self._episode(channel, epsilon=epsilon, rng=rng)[0] for channel in normalized]
        )

    def evaluate(self, channels: np.ndarray) -> dict[str, float]:
        deployments = self.deployments(channels, deterministic=True)
        rewards, details = self.environment.evaluate(channels, deployments)
        return {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
            "latency_mean_seconds": float(details["latency_seconds"].mean()),
            "invalid_fraction": float(details["invalid"].mean()),
        }

    def _learn_once(self, rng: np.random.Generator) -> float | None:
        if len(self.replay) < max(self.config.replay_warmup_transitions, self.config.batch_size):
            return None
        indices = rng.choice(len(self.replay), size=self.config.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        states = torch.from_numpy(np.stack([item.state for item in batch])).to(self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([item.next_state for item in batch])).to(self.device)
        next_masks = torch.from_numpy(np.stack([item.next_mask for item in batch])).to(self.device)
        done = torch.tensor([item.done for item in batch], dtype=torch.bool, device=self.device)

        predicted = self.online(states).gather(1, actions[:, None]).squeeze(1)
        targets = rewards.clone()
        nonterminal = ~done
        if nonterminal.any():
            # Double-DQN: online network selects, target network evaluates.
            online_next = self.online.masked_q_values(
                next_states[nonterminal], next_masks[nonterminal]
            )
            next_actions = online_next.argmax(dim=1)
            with torch.no_grad():
                target_next = self.target(next_states[nonterminal])
                next_values = target_next.gather(1, next_actions[:, None]).squeeze(1)
            targets[nonterminal] += self.config.gamma * next_values
        loss = F.smooth_l1_loss(predicted, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

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
        """Train Double-DQN and save the best deterministic validation network."""

        channel_rng = np.random.default_rng(self.config.seed)
        action_rng = np.random.default_rng(self.config.seed + 1)
        replay_rng = np.random.default_rng(self.config.seed + 2)
        validation_channels = generate_resource_channels(
            self.config.validation_channels,
            self.config.validation_seed,
            self.resource_config,
        )
        best_reward = float("-inf")
        best_episodes = 0
        best_state = deepcopy(self.online.state_dict())
        history: dict[str, list[float]] = {
            "episodes": [],
            "training_reward": [],
            "validation_reward": [],
            "epsilon": [],
            "loss": [],
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
            epsilon = self.epsilon(completed)
            deployments: list[np.ndarray] = []
            episode_transitions = []
            for channel in normalized:
                deployment, transitions = self._episode(
                    channel, epsilon=epsilon, rng=action_rng
                )
                deployments.append(deployment)
                episode_transitions.append(transitions)
            rewards, details = self.environment.evaluate(channels, np.stack(deployments))
            for terminal_reward, transitions in zip(rewards, episode_transitions, strict=True):
                for state, mask, action, next_state, next_mask, done in transitions:
                    self.replay.append(
                        ReplayTransition(
                            state=state,
                            mask=mask,
                            action=action,
                            reward=float(terminal_reward) if done else 0.0,
                            next_state=next_state,
                            next_mask=next_mask,
                            done=done,
                        )
                    )
            losses = [
                loss
                for _ in range(self.config.gradient_steps_per_rollout)
                if (loss := self._learn_once(replay_rng)) is not None
            ]
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
                    best_state = deepcopy(self.online.state_dict())
            history["episodes"].append(float(completed))
            history["training_reward"].append(float(rewards.mean()))
            history["validation_reward"].append(validation_reward)
            history["epsilon"].append(epsilon)
            history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
            history["invalid_fraction"].append(float(details["invalid"].mean()))
            print(
                f"algorithm=dqn episodes={completed} reward={rewards.mean():.5f} "
                f"epsilon={epsilon:.3f} validation={validation_reward:.5f}",
                flush=True,
            )

        self.online.load_state_dict(best_state)
        self.target.load_state_dict(best_state)
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
