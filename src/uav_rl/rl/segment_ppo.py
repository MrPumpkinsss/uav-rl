"""Independent variable-length segment PPO for UAV layer deployment."""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import PPOConfig
from uav_rl.noise_seeds import sample_training_noise_seeds, validation_noise_seeds
from uav_rl.rl.environment import DeploymentEnvironment, generate_channels
from uav_rl.rl.segment_policy import (
    SegmentActorCritic,
    decode_action,
    deployment_segments,
    partial_state,
    valid_action_mask,
)
from uav_rl.wireless import sample_channel


@dataclass(frozen=True)
class SegmentPPOOptions:
    """Objective and action-space choices for segment PPO.

    `fixed_capacity_segments` matches the current 32-layer, 8-layer-per-UAV
    physical task: the policy selects an ordered four-UAV path, not redundant
    segment lengths. Positive-reference PPO updates only on sampled deployments
    that beat the training-only structured reference under identical noise seeds.
    """

    fixed_capacity_segments: bool = False
    positive_reference_improvement: bool = False
    improvement_margin: float = 0.0
    reference_kl_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.improvement_margin < 0.0:
            raise ValueError("improvement margin cannot be negative")
        if self.reference_kl_coefficient < 0.0:
            raise ValueError("reference KL coefficient cannot be negative")


class SegmentPPOTrainer:
    """Train a policy that ends an episode as soon as all layers are assigned.

    A structured teacher is optional and used only for behavior-cloning warm start.
    Policy inference never calls the teacher.
    """

    policy_type = "segment_multistep"

    def __init__(
        self,
        config: PPOConfig,
        environment: DeploymentEnvironment,
        teacher_action_provider: Callable[[np.ndarray], np.ndarray] | None = None,
        options: SegmentPPOOptions = SegmentPPOOptions(),
    ) -> None:
        self.config = config
        self.environment = environment
        self.teacher_action_provider = teacher_action_provider
        self.options = options
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = SegmentActorCritic(config.system, config.hidden_dim)
        self.reference_model = SegmentActorCritic(config.system, config.hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.validation_noise_seeds = validation_noise_seeds(
            config.validation_noise_samples, config.validation_noise_seed
        )

    def _action_mask(self, *, assigned_layers: int, used_uavs: np.ndarray) -> np.ndarray:
        """Return the legal action mask, optionally fixing each segment to capacity."""

        mask = valid_action_mask(
            assigned_layers=assigned_layers, used_uavs=used_uavs, config=self.config.system
        )
        if not self.options.fixed_capacity_segments:
            return mask
        capacity_indexes = np.arange(self.config.system.num_uavs) * self.config.system.max_layers_per_uav
        capacity_indexes += self.config.system.max_layers_per_uav - 1
        fixed_mask = np.zeros_like(mask)
        fixed_mask[capacity_indexes] = mask[capacity_indexes]
        if not fixed_mask.any():
            raise RuntimeError("fixed-capacity segment policy has no feasible action")
        return fixed_mask

    def _deployment(self, normalized_channel: np.ndarray, *, deterministic: bool) -> np.ndarray:
        """Roll out one legal deployment without consulting a teacher."""

        assigned = 0
        used = np.zeros(self.config.system.num_uavs, dtype=bool)
        previous: int | None = None
        deployment: list[int] = []
        while assigned < self.config.system.num_layers:
            state = partial_state(
                normalized_channel,
                assigned_layers=assigned,
                used_uavs=used,
                previous_uav=previous,
                config=self.config.system,
            )
            mask = self._action_mask(assigned_layers=assigned, used_uavs=used)
            with torch.no_grad():
                output = self.model.sample(
                    torch.from_numpy(state[None, :]),
                    torch.from_numpy(mask[None, :]),
                    deterministic=deterministic,
                )
            uav, length = decode_action(int(output.actions.item()), self.config.system)
            deployment.extend([uav] * length)
            assigned += length
            used[uav] = True
            previous = uav
        return np.asarray(deployment, dtype=np.int64)

    def deployments(self, channels: np.ndarray, *, deterministic: bool) -> np.ndarray:
        """Map channel matrices to deployments using only the PPO policy."""

        normalized = self.environment.normalize_channels(channels)
        return np.stack(
            [self._deployment(channel, deterministic=deterministic) for channel in normalized]
        )

    def _validation_reward(self, channels: np.ndarray) -> float:
        deployments = self.deployments(channels, deterministic=True)
        rewards, _ = self.environment.evaluate(
            channels, deployments, noise_seeds=self.validation_noise_seeds
        )
        return float(rewards.mean())

    def _teacher_transitions(
        self, channels: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert complete teacher deployments to legal segment decisions."""

        if self.teacher_action_provider is None:
            raise ValueError("behavior cloning requires an explicit teacher action provider")
        deployments = np.asarray(self.teacher_action_provider(channels), dtype=np.int64)
        expected = (len(channels), self.config.system.num_layers)
        if deployments.shape != expected:
            raise ValueError(f"teacher action provider returned {deployments.shape}, expected {expected}")
        normalized = self.environment.normalize_channels(channels)
        states: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        for channel, deployment in zip(normalized, deployments, strict=True):
            assigned = 0
            used = np.zeros(self.config.system.num_uavs, dtype=bool)
            previous: int | None = None
            for uav, length in deployment_segments(deployment, self.config.system):
                states.append(
                    partial_state(
                        channel,
                        assigned_layers=assigned,
                        used_uavs=used,
                        previous_uav=previous,
                        config=self.config.system,
                    )
                )
                masks.append(self._action_mask(assigned_layers=assigned, used_uavs=used))
                actions.append(uav * self.config.system.max_layers_per_uav + length - 1)
                assigned += length
                used[uav] = True
                previous = uav
        return (
            torch.from_numpy(np.stack(states)),
            torch.from_numpy(np.stack(masks)),
            torch.tensor(actions, dtype=torch.long),
        )

    def _behavior_clone(self) -> list[float]:
        if self.config.teacher_channels == 0 or self.config.behavior_cloning_epochs == 0:
            return []
        channels = generate_channels(
            self.config.teacher_channels, self.config.teacher_seed, self.config.system
        )
        states, masks, actions = self._teacher_transitions(channels)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.behavior_cloning_learning_rate
        )
        losses: list[float] = []
        for epoch in range(self.config.behavior_cloning_epochs):
            order = torch.randperm(len(actions))
            total = 0.0
            for start in range(0, len(order), self.config.minibatch_size):
                index = order[start : start + self.config.minibatch_size]
                output = self.model.evaluate(states[index], masks[index], actions[index])
                loss = -output.log_probabilities.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                total += float(loss.item()) * len(index)
            mean_loss = total / len(actions)
            losses.append(mean_loss)
            print(f"segment_behavior_cloning_epoch={epoch + 1:3d} loss={mean_loss:.6f}", flush=True)
        return losses

    def _rollout(
        self, channels: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample variable-length legal trajectories and return complete deployments."""

        count = len(channels)
        normalized = self.environment.normalize_channels(channels)
        assigned = np.zeros(count, dtype=np.int64)
        used = np.zeros((count, self.config.system.num_uavs), dtype=bool)
        previous = np.full(count, -1, dtype=np.int64)
        deployments: list[list[int]] = [[] for _ in range(count)]
        states: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        while np.any(assigned < self.config.system.num_layers):
            active = np.flatnonzero(assigned < self.config.system.num_layers)
            step_states = np.stack(
                [
                    partial_state(
                        normalized[index],
                        assigned_layers=int(assigned[index]),
                        used_uavs=used[index],
                        previous_uav=None if previous[index] < 0 else int(previous[index]),
                        config=self.config.system,
                    )
                    for index in active
                ]
            )
            step_masks = np.stack(
                [
                    self._action_mask(
                        assigned_layers=int(assigned[index]), used_uavs=used[index]
                    )
                    for index in active
                ]
            )
            state_tensor = torch.from_numpy(step_states)
            mask_tensor = torch.from_numpy(step_masks)
            with torch.no_grad():
                output = self.model.sample(state_tensor, mask_tensor)
            states.append(state_tensor)
            masks.append(mask_tensor)
            actions.append(output.actions)
            log_probabilities.append(output.log_probabilities)
            values.append(output.values)
            for local, index in enumerate(active):
                uav, length = decode_action(int(output.actions[local].item()), self.config.system)
                deployments[index].extend([uav] * length)
                assigned[index] += length
                used[index, uav] = True
                previous[index] = uav
        return (
            torch.cat(states),
            torch.cat(masks),
            torch.cat(actions),
            torch.cat(log_probabilities),
            torch.cat(values),
            np.asarray(deployments, dtype=np.int64),
        )

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        with temporary.open("ab") as file:
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)

    def _candidate_payload(self, episodes: int, monitor_reward: float) -> dict[str, Any]:
        return {
            "format_version": 1,
            "purpose": "external_policy_validation_candidate",
            "policy_type": self.policy_type,
            "model_state": self.model.state_dict(),
            "ppo_config": asdict(self.config),
            "segment_ppo_options": asdict(self.options),
            "episodes": episodes,
            "training_monitor_reward": monitor_reward,
            "inference": "channel_to_variable_length_segments_without_teacher",
        }

    def _save_candidate(self, directory: Path, episodes: int, monitor_reward: float) -> Path:
        path = directory / f"episode_{episodes:06d}.pth"
        self._atomic_save(self._candidate_payload(episodes, monitor_reward), path)
        return path

    def _save_state(
        self,
        path: Path,
        *,
        completed: int,
        rollout_index: int,
        best_reward: float,
        best_episodes: int,
        best_model_state: dict[str, torch.Tensor],
        history: dict[str, list[float]],
        channel_rng: np.random.Generator,
        noise_rng: np.random.Generator,
        run_metadata: dict[str, Any],
    ) -> None:
        self._atomic_save(
            {
                "format_version": 1,
                "policy_type": self.policy_type,
                "ppo_config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "reference_model_state": self.reference_model.state_dict(),
                "segment_ppo_options": asdict(self.options),
                "best_model_state": best_model_state,
                "best_validation_reward": best_reward,
                "best_episodes": best_episodes,
                "completed_episodes": completed,
                "rollout_index": rollout_index,
                "history": history,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "channel_rng_state": channel_rng.bit_generator.state,
                "noise_rng_state": noise_rng.bit_generator.state,
                "run_metadata": run_metadata,
            },
            path,
        )

    def _load_state(
        self, path: Path, run_metadata: dict[str, Any]
    ) -> tuple[
        int,
        int,
        float,
        int,
        dict[str, torch.Tensor],
        dict[str, list[float]],
        np.random.Generator,
        np.random.Generator,
    ]:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1 or state.get("policy_type") != self.policy_type:
            raise ValueError("unsupported segment PPO training-state format")
        if state.get("segment_ppo_options") != asdict(self.options):
            raise ValueError("resume segment PPO options differ from the saved experiment")
        if state.get("run_metadata") != run_metadata:
            raise ValueError("resume run metadata differs from the saved experiment")
        saved = dict(state["ppo_config"])
        current = asdict(self.config)
        saved_target = saved.pop("training_episodes")
        current_target = current.pop("training_episodes")
        if saved != current:
            raise ValueError("resume configuration differs; only training_episodes may increase")
        completed = int(state["completed_episodes"])
        if current_target < saved_target or current_target < completed:
            raise ValueError("training_episodes cannot be reduced when resuming")
        self.model.load_state_dict(state["model_state"])
        self.reference_model.load_state_dict(state["reference_model_state"])
        self.reference_model.eval()
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
            state["history"],
            channel_rng,
            noise_rng,
        )

    def train(
        self,
        output_path: Path,
        *,
        state_path: Path,
        resume: bool,
        run_metadata: dict[str, Any],
        candidate_directory: Path,
        candidate_interval_episodes: int,
    ) -> dict[str, list[float]]:
        """Run PPO with exact periodic candidates and lossless boundary resume."""

        if candidate_interval_episodes < 1:
            raise ValueError("candidate interval must be positive")
        candidate_directory.mkdir(parents=True, exist_ok=True)
        validation_channels = generate_channels(
            self.config.validation_channels, self.config.validation_seed, self.config.system
        )
        if resume:
            (
                completed,
                rollout_index,
                best_reward,
                best_episodes,
                best_model_state,
                history,
                channel_rng,
                noise_rng,
            ) = self._load_state(state_path, run_metadata)
            print(f"segment_resumed_episodes={completed} target={self.config.training_episodes}", flush=True)
        else:
            history = {
                "behavior_cloning_loss": self._behavior_clone(),
                "reward": [],
                "latency": [],
                "log_ppl_ratio": [],
                "mean_segments": [],
                "validation_reward": [],
            }
            if self.options.positive_reference_improvement:
                history.update(
                    {
                        "reference_reward": [],
                        "reference_improvement": [],
                        "positive_improvement_fraction": [],
                        "reference_kl": [],
                    }
                )
            if self.options.positive_reference_improvement and self.teacher_action_provider is None:
                raise ValueError("positive-reference PPO requires a training-only teacher provider")
            self.reference_model.load_state_dict(self.model.state_dict())
            self.reference_model.eval()
            completed = 0
            rollout_index = 0
            channel_rng = np.random.default_rng(self.config.seed)
            noise_rng = np.random.default_rng(self.config.training_noise_seed)
            best_reward = self._validation_reward(validation_channels)
            best_episodes = 0
            best_model_state = deepcopy(self.model.state_dict())
            self._atomic_save(self._candidate_payload(0, best_reward), output_path)
            self._save_candidate(candidate_directory, 0, best_reward)
            self._save_state(
                state_path,
                completed=completed,
                rollout_index=rollout_index,
                best_reward=best_reward,
                best_episodes=best_episodes,
                best_model_state=best_model_state,
                history=history,
                channel_rng=channel_rng,
                noise_rng=noise_rng,
                run_metadata=run_metadata,
            )
            print(f"segment_initial_validation_reward={best_reward:.6f}", flush=True)

        while completed < self.config.training_episodes:
            until_candidate = candidate_interval_episodes - (completed % candidate_interval_episodes)
            rollout_size = min(
                self.config.rollout_size,
                until_candidate,
                self.config.training_episodes - completed,
            )
            channels = np.stack(
                [sample_channel(channel_rng, self.config.system) for _ in range(rollout_size)]
            ).astype(np.float32)
            states, masks, actions, old_log_probabilities, old_values, deployments = self._rollout(channels)
            noise_seeds = sample_training_noise_seeds(
                noise_rng, rollout_size, self.config.training_noise_samples
            )
            rewards_np, details = self.environment.evaluate(
                channels, deployments, noise_seeds=noise_seeds
            )
            transition_counts = np.asarray(
                [len(deployment_segments(row, self.config.system)) for row in deployments]
            )
            episode_ids = np.repeat(np.arange(rollout_size), transition_counts)
            reference_rewards_np: np.ndarray | None = None
            if self.options.positive_reference_improvement:
                assert self.teacher_action_provider is not None
                reference_deployments = np.asarray(
                    self.teacher_action_provider(channels), dtype=np.int64
                )
                reference_rewards_np, _ = self.environment.evaluate(
                    channels, reference_deployments, noise_seeds=noise_seeds
                )
                improvements = rewards_np - reference_rewards_np
                positive = improvements > self.options.improvement_margin
                positive_values = improvements[positive] - self.options.improvement_margin
                episode_advantages = np.zeros_like(improvements)
                if len(positive_values):
                    scale = max(
                        float(positive_values.std()), float(positive_values.mean()), 1e-8
                    )
                    episode_advantages[positive] = positive_values / scale
                returns = torch.from_numpy(improvements[episode_ids])
                normalized_advantages = torch.from_numpy(episode_advantages[episode_ids])
            else:
                returns = torch.from_numpy(rewards_np[episode_ids])
                advantages = returns - old_values
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            transitions = len(actions)
            kl_total = 0.0
            kl_samples = 0
            for _ in range(self.config.update_epochs):
                order = torch.randperm(transitions)
                for start in range(0, transitions, self.config.minibatch_size):
                    index = order[start : start + self.config.minibatch_size]
                    output = self.model.evaluate(states[index], masks[index], actions[index])
                    ratio = torch.exp(output.log_probabilities - old_log_probabilities[index])
                    advantage = normalized_advantages[index]
                    policy_loss = -torch.minimum(
                        ratio * advantage,
                        torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                        * advantage,
                    ).mean()
                    value_loss = torch.mean((output.values - returns[index]) ** 2)
                    loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * output.entropy.mean()
                    )
                    if self.options.reference_kl_coefficient > 0.0:
                        with torch.no_grad():
                            reference_distribution = self.reference_model._distribution(
                                states[index], masks[index]
                            )
                        current_distribution = self.model._distribution(states[index], masks[index])
                        reference_kl = torch.distributions.kl_divergence(
                            reference_distribution, current_distribution
                        ).mean()
                        loss = loss + self.options.reference_kl_coefficient * reference_kl
                        kl_total += float(reference_kl.item()) * len(index)
                        kl_samples += len(index)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
            completed += rollout_size
            rollout_index += 1
            history["reward"].append(float(rewards_np.mean()))
            history["latency"].append(float(details["latency_seconds"].mean()))
            history["log_ppl_ratio"].append(float(details["log_ppl_ratio"].mean()))
            history["mean_segments"].append(float(transition_counts.mean()))
            if reference_rewards_np is not None:
                history["reference_reward"].append(float(reference_rewards_np.mean()))
                history["reference_improvement"].append(float((rewards_np - reference_rewards_np).mean()))
                history["positive_improvement_fraction"].append(float(positive.mean()))
                history["reference_kl"].append(kl_total / kl_samples if kl_samples else 0.0)
            candidate_due = completed % candidate_interval_episodes == 0
            should_monitor = (
                candidate_due
                or completed == self.config.training_episodes
                or rollout_index % self.config.validation_interval == 0
            )
            monitor_reward = best_reward
            if should_monitor:
                monitor_reward = self._validation_reward(validation_channels)
                history["validation_reward"].append(monitor_reward)
                if monitor_reward > best_reward:
                    best_reward = monitor_reward
                    best_episodes = completed
                    best_model_state = deepcopy(self.model.state_dict())
                    self._atomic_save(self._candidate_payload(completed, best_reward), output_path)
            if candidate_due:
                self._save_candidate(candidate_directory, completed, monitor_reward)
            self._save_state(
                state_path,
                completed=completed,
                rollout_index=rollout_index,
                best_reward=best_reward,
                best_episodes=best_episodes,
                best_model_state=best_model_state,
                history=history,
                channel_rng=channel_rng,
                noise_rng=noise_rng,
                run_metadata=run_metadata,
            )
            print(
                f"segment_episodes={completed:5d} reward={rewards_np.mean():.4f} "
                f"segments={transition_counts.mean():.2f} monitor={monitor_reward:.4f}",
                flush=True,
            )
            if reference_rewards_np is not None:
                print(
                    f"reference_reward={reference_rewards_np.mean():.4f} "
                    f"improvement={(rewards_np - reference_rewards_np).mean():.5f} "
                    f"positive_fraction={positive.mean():.3f} "
                    f"reference_kl={kl_total / kl_samples if kl_samples else 0.0:.6f}",
                    flush=True,
                )
        self.model.load_state_dict(best_model_state)
        self._atomic_save(self._candidate_payload(best_episodes, best_reward), output_path)
        return history
