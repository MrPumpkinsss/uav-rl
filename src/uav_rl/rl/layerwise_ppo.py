"""面向论文实验的任意 layer-to-UAV assignment PPO 训练器。"""

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
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.layerwise_policy import (
    LayerwiseActorCritic,
    layer_state,
    valid_layer_action_mask,
)


class LayerwisePPOTrainer:
    """训练一个按 layer 顺序自回归决策的部署策略。"""

    policy_type = "layerwise_general_assignment"

    def __init__(
        self,
        config: PPOConfig,
        resource_config: ResourceConstrainedConfig,
        environment: ResourceDeploymentEnvironment,
        teacher_action_provider: Callable[[np.ndarray], np.ndarray] | None = None,
        max_policy_boundaries: int = 4,
    ) -> None:
        """初始化 layerwise PPO 训练器及其策略、优化器和运行状态。"""
        self.config = config
        self.resource_config = resource_config
        self.environment = environment
        self.teacher_action_provider = teacher_action_provider
        if max_policy_boundaries < 1 or max_policy_boundaries >= resource_config.system.num_layers:
            raise ValueError('max_policy_boundaries is out of range')
        self.max_policy_boundaries = max_policy_boundaries
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.model = LayerwiseActorCritic(resource_config, config.hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.validation_noise_seeds = validation_noise_seeds(
            config.validation_noise_samples, config.validation_noise_seed
        )

    def _rollout_one(
        self,
        normalized_channel: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[float], list[float], np.ndarray]:
        """执行一条 PPO rollout，采样 action 并计算对应 reward。"""
        layers = self.resource_config.system.num_layers
        uavs = self.resource_config.system.num_uavs
        memory_used = np.zeros(uavs, dtype=np.float64)
        energy_used = np.zeros(uavs, dtype=np.float64)
        previous: int | None = None
        boundary_count = 0
        states: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        log_probs: list[float] = []
        values: list[float] = []
        deployment = np.zeros(layers, dtype=np.int64)
        speeds = np.asarray(self.resource_config.system.compute_speed, dtype=np.float64)
        for layer_index in range(layers):
            # 每一步只决定当前 layer 的 UAV；state 同时包含信道、资源余量和上一层 UAV。
            state = layer_state(
                normalized_channel,
                layer_index=layer_index,
                memory_used=memory_used,
                energy_used=energy_used,
                previous_uav=previous,
                config=self.resource_config,
            )
            mask = valid_layer_action_mask(
                layer_index=layer_index,
                memory_used=memory_used,
                energy_used=energy_used,
                config=self.resource_config,
            )
            # 达到 boundary freeze threshold 后，若上一台 UAV 仍可行，则保持连续部署。
            # 这不是硬性的 boundary 上限；上一台 UAV 不可行时仍允许切换。
            if previous is not None and boundary_count >= self.max_policy_boundaries and mask[previous]:
                mask[:] = False
                mask[previous] = True
            with torch.no_grad():
                output = self.model.sample(
                    torch.from_numpy(state[None, :]),
                    torch.from_numpy(mask[None, :]),
                    deterministic=deterministic,
                )
            # 先保存 transition，随后才更新资源累计量，避免当前 action 看到未来资源。
            action = int(output.actions.item())
            states.append(state)
            masks.append(mask)
            actions.append(action)
            log_probs.append(float(output.log_probabilities.item()))
            values.append(float(output.values.item()))
            deployment[layer_index] = action
            # 相邻 layer 的 UAV 不同时产生一次跨 UAV activation boundary。
            if previous is not None and action != previous:
                boundary_count += 1
            memory_used[action] += self.resource_config.layer_memory_units[layer_index]
            energy_used[action] += float(
                self.resource_config.compute_energy_coefficient
                * speeds[action] ** 2
                * self.resource_config.layer_compute_seconds_at_unit_speed[layer_index]
            )
            previous = action
        return states, masks, actions, log_probs, values, deployment

    def deployments(self, channels: np.ndarray, *, deterministic: bool) -> np.ndarray:
        """根据策略生成多个可行 deployment，供 deterministic 或 Top-K 评估使用。"""
        normalized = self.environment.normalize_channels(channels)
        return np.stack(
            [self._rollout_one(channel, deterministic=deterministic)[-1] for channel in normalized]
        )

    def top_k_deployments(
        self,
        channels: np.ndarray,
        *,
        k: int,
        samples_per_channel: int | None = None,
    ):
        """生成候选 deployment 并按 surrogate reward 排序，返回前 k 个候选。"""
        if k < 1:
            raise ValueError('k must be positive')
        channels = np.asarray(channels, dtype=np.float32)
        normalized = self.environment.normalize_channels(channels)
        candidates = []
        rewards = []
        # beam_width 至少为 k；默认保留 4k 条 beam，避免搜索空间随层数指数爆炸。
        beam_width = max(k, samples_per_channel or (4 * k))
        speeds = np.asarray(self.resource_config.system.compute_speed, dtype=np.float64)
        for channel, normalized_channel in zip(channels, normalized, strict=True):
            # beam 依次保存累计 log-prob、资源占用、上一 UAV、boundary 数和部分 deployment。
            beams = [(0.0, np.zeros(self.resource_config.system.num_uavs), np.zeros(self.resource_config.system.num_uavs), None, 0, [])]
            for layer_index in range(self.resource_config.system.num_layers):
                # 当前层扩展所有 beam，再按 policy log-prob 截断，而不是枚举完整 UAV^layer 空间。
                expanded = []
                for logp, memory_used, energy_used, previous, boundary_count, deployment in beams:
                    state = layer_state(normalized_channel, layer_index=layer_index, memory_used=memory_used, energy_used=energy_used, previous_uav=previous, config=self.resource_config)
                    mask = valid_layer_action_mask(layer_index=layer_index, memory_used=memory_used, energy_used=energy_used, config=self.resource_config)
                    if previous is not None and boundary_count >= self.max_policy_boundaries and mask[previous]:
                        mask[:] = False
                        mask[previous] = True
                    with torch.no_grad():
                        distribution = self.model._distribution(torch.from_numpy(state[None, :]), torch.from_numpy(mask[None, :]))
                        action_scores = [(int(action), float(distribution.log_prob(torch.tensor([int(action)])).item())) for action in np.flatnonzero(mask)]
                    # 只扩展当前 policy 最可能的 beam_width 个合法动作。
                    action_scores.sort(key=lambda item: item[1], reverse=True)
                    for action, action_logp in action_scores[:beam_width]:
                        next_memory = memory_used.copy()
                        next_energy = energy_used.copy()
                        next_memory[action] += self.resource_config.layer_memory_units[layer_index]
                        next_energy[action] += self.resource_config.compute_energy_coefficient * speeds[action] ** 2 * self.resource_config.layer_compute_seconds_at_unit_speed[layer_index]
                        expanded.append((logp + action_logp, next_memory, next_energy, action, boundary_count + int(previous is not None and action != previous), deployment + [action]))
                # 保留累计 log-prob 最高的路径，控制 Top-K 候选生成的计算量。
                expanded.sort(key=lambda item: item[0], reverse=True)
                beams = expanded[:beam_width]
            # 不同 beam 可能得到相同 deployment；去重后再进行 surrogate 评分。
            unique = {}
            for _, _, _, _, _, deployment in beams:
                array = np.asarray(deployment, dtype=np.int64)
                unique.setdefault(array.tobytes(), array)
            candidate_array = np.stack(list(unique.values()))
            repeated = np.repeat(channel[None, :, :], len(candidate_array), axis=0)
            # beam 概率只负责产生候选，最终 Top-K 排名必须使用 surrogate reward。
            values, _ = self.environment.evaluate(repeated, candidate_array)
            order = np.argsort(values)[::-1][:k]
            if len(order) < k:
                order = np.pad(order, (0, k - len(order)), mode='edge')
            candidates.append(candidate_array[order])
            rewards.append(values[order])
        return np.stack(candidates), np.stack(rewards)

    def _validation_reward(self, channels: np.ndarray) -> float:
        """在 validation context 上评估当前策略的平均 reward。"""
        deployments = self.deployments(channels, deterministic=True)
        rewards, _ = self.environment.evaluate(
            channels, deployments, noise_seeds=self.validation_noise_seeds
        )
        return float(rewards.mean())

    def _behavior_clone(self) -> list[float]:
        """用已有高质量 deployment 对策略进行行为克隆预热。"""
        if self.config.teacher_channels == 0 or self.config.behavior_cloning_epochs == 0:
            return []
        if self.teacher_action_provider is None:
            raise ValueError("layerwise behavior cloning requires an explicit teacher provider")
        channels = generate_resource_channels(
            self.config.teacher_channels, self.config.teacher_seed, self.resource_config
        )
        deployments = np.asarray(self.teacher_action_provider(channels), dtype=np.int64)
        expected = (len(channels), self.resource_config.system.num_layers)
        if deployments.shape != expected:
            raise ValueError(f"teacher provider returned {deployments.shape}, expected {expected}")
        normalized = self.environment.normalize_channels(channels)
        states: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        speeds = np.asarray(self.resource_config.system.compute_speed, dtype=np.float64)
        for channel, deployment in zip(normalized, deployments, strict=True):
            memory_used = np.zeros(self.resource_config.system.num_uavs, dtype=np.float64)
            energy_used = np.zeros(self.resource_config.system.num_uavs, dtype=np.float64)
            previous: int | None = None
            for layer_index, action in enumerate(deployment):
                mask = valid_layer_action_mask(
                    layer_index=layer_index,
                    memory_used=memory_used,
                    energy_used=energy_used,
                    config=self.resource_config,
                )
                action = int(action)
                if not mask[action]:
                    raise ValueError("teacher deployment is infeasible under resource constraints")
                states.append(
                    layer_state(
                        channel,
                        layer_index=layer_index,
                        memory_used=memory_used,
                        energy_used=energy_used,
                        previous_uav=previous,
                        config=self.resource_config,
                    )
                )
                masks.append(mask)
                actions.append(action)
                memory_used[action] += self.resource_config.layer_memory_units[layer_index]
                energy_used[action] += float(
                    self.resource_config.compute_energy_coefficient
                    * speeds[action] ** 2
                    * self.resource_config.layer_compute_seconds_at_unit_speed[layer_index]
                )
                previous = action
        state_tensor = torch.from_numpy(np.stack(states))
        mask_tensor = torch.from_numpy(np.stack(masks))
        action_tensor = torch.tensor(actions, dtype=torch.long)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.behavior_cloning_learning_rate
        )
        losses: list[float] = []
        for epoch in range(self.config.behavior_cloning_epochs):
            order = torch.randperm(len(actions))
            total = 0.0
            for start in range(0, len(order), self.config.minibatch_size):
                index = order[start : start + self.config.minibatch_size]
                output = self.model.evaluate(state_tensor[index], mask_tensor[index], action_tensor[index])
                loss = -output.log_probabilities.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                total += float(loss.item()) * len(index)
            mean_loss = total / len(actions)
            losses.append(mean_loss)
            print(f"layerwise_behavior_cloning_epoch={epoch + 1:3d} loss={mean_loss:.6f}", flush=True)
        return losses

    def _atomic_save(self, payload: dict[str, Any], path: Path) -> None:
        """执行 _atomic_save，完成本模块中的对应数据处理或实验步骤。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        with temporary.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def _candidate_payload(self, episodes: int, monitor_reward: float) -> dict[str, Any]:
        """执行 _candidate_payload，完成本模块中的对应数据处理或实验步骤。"""
        return {
            "format_version": 1,
            "purpose": "external_policy_validation_candidate",
            "policy_type": self.policy_type,
            "model_state": self.model.state_dict(),
            "ppo_config": asdict(self.config),
            "resource_config": self.resource_config.to_dict(),
            "max_policy_boundaries": self.max_policy_boundaries,
            "episodes": episodes,
            "training_monitor_reward": monitor_reward,
            "inference": "channel_to_layerwise_assignment_without_teacher",
        }

    def train(
        self,
        output_path: Path,
        *,
        state_path: Path,
        run_metadata: dict[str, Any],
        candidate_directory: Path,
        candidate_interval_episodes: int,
        resume: bool = False,
    ) -> dict[str, list[float]]:
        """训练 layerwise PPO，并在训练过程中保存可恢复状态和候选策略。

        每轮先用当前策略采样多个 layer-to-UAV deployment，再由资源环境计算
        reward、PPL 相对退化和时延；随后按照 PPO clipped objective 更新
        Actor-Critic 网络。validation reward 用来选择最优模型，state_path
        保存完整随机状态、优化器状态和训练历史，因此可以安全 resume。
        """

        # candidate_interval_episodes 决定多久导出一次候选策略，同时让保存点
        # 落在完整 rollout 之后，避免中断时留下半个 rollout 的不一致状态。
        if candidate_interval_episodes < 1:
            raise ValueError("candidate interval must be positive")
        candidate_directory.mkdir(parents=True, exist_ok=True)
        # validation channel 在整个训练期间固定，只用于比较不同 checkpoint 的策略质量。
        validation_channels = generate_resource_channels(
            self.config.validation_channels, self.config.validation_seed, self.resource_config
        )
        if resume:
            # resume 时恢复模型、优化器和三套随机状态，保证续训不会改变实验轨迹。
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state.get("run_metadata") != run_metadata or state.get("policy_type") != self.policy_type:
                raise ValueError("layerwise resume metadata is incompatible")
            if state.get('resource_config') != self.resource_config.to_dict():
                raise ValueError('layerwise resource configuration is incompatible')
            saved_config = dict(state["ppo_config"])
            current_config = asdict(self.config)
            saved_target = saved_config.pop("training_episodes")
            current_target = current_config.pop("training_episodes")
            if saved_config != current_config or current_target < saved_target:
                raise ValueError("only training_episodes may increase when resuming")
            self.model.load_state_dict(state["model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            random.setstate(state["python_random_state"])
            np.random.set_state(state["numpy_random_state"])
            torch.set_rng_state(state["torch_random_state"])
            if torch.cuda.is_available() and state.get('torch_cuda_random_states') is not None:
                torch.cuda.set_rng_state_all(state['torch_cuda_random_states'])
            completed = int(state["completed_episodes"])
            rollout_index = int(state["rollout_index"])
            best_reward = float(state["best_validation_reward"])
            best_episodes = int(state["best_episodes"])
            best_model_state = state["best_model_state"]
            history = state["history"]
            channel_rng = np.random.default_rng()
            channel_rng.bit_generator.state = state["channel_rng_state"]
            noise_rng = np.random.default_rng()
            noise_rng.bit_generator.state = state["noise_rng_state"]
        else:
            # 新训练先进行 behavior cloning 预热，再建立独立的 channel/noise 随机流。
            history = {
                "behavior_cloning_loss": self._behavior_clone(),
                "reward": [],
                "latency": [],
                "log_ppl_ratio": [],
                "invalid_fraction": [],
                "validation_reward": [],
            }
            completed = 0
            rollout_index = 0
            channel_rng = np.random.default_rng(self.config.seed)
            noise_rng = np.random.default_rng(self.config.training_noise_seed)
            best_reward = self._validation_reward(validation_channels)
            best_episodes = 0
            best_model_state = deepcopy(self.model.state_dict())

        # 外层循环以 rollout 为单位推进；每次 rollout 后才增加 completed episode 数。
        while completed < self.config.training_episodes:
            # 不跨越 candidate interval，确保候选策略按精确 episode 间隔导出。
            rollout_size = min(
                self.config.rollout_size,
                candidate_interval_episodes - completed % candidate_interval_episodes,
                self.config.training_episodes - completed,
            )
            channels = generate_resource_channels(rollout_size, int(channel_rng.integers(2**31)), self.resource_config)
            normalized = self.environment.normalize_channels(channels)
            all_states: list[np.ndarray] = []
            all_masks: list[np.ndarray] = []
            all_actions: list[int] = []
            all_old_log_probs: list[float] = []
            all_old_values: list[float] = []
            deployments: list[np.ndarray] = []
            # 对每个 channel 自回归地为所有 layer 选择 UAV，记录 PPO 更新所需的旧策略量。
            for channel in normalized:
                states, masks, actions, log_probs, values, deployment = self._rollout_one(
                    channel, deterministic=False
                )
                all_states.extend(states)
                all_masks.extend(masks)
                all_actions.extend(actions)
                all_old_log_probs.extend(log_probs)
                all_old_values.extend(values)
                deployments.append(deployment)
            deployments_array = np.stack(deployments)
            noise_seeds = sample_training_noise_seeds(
                noise_rng, rollout_size, self.config.training_noise_samples
            )
            # reward 在 rollout 结束后统一计算；同一个 deployment 的所有 layer step
            # 共享该 episode reward，形成 Monte-Carlo return 的训练目标。
            rewards_np, details = self.environment.evaluate(
                channels, deployments_array, noise_seeds=noise_seeds
            )
            states_tensor = torch.from_numpy(np.stack(all_states))
            masks_tensor = torch.from_numpy(np.stack(all_masks))
            actions_tensor = torch.tensor(all_actions, dtype=torch.long)
            old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32)
            old_values = torch.tensor(all_old_values, dtype=torch.float32)
            # 每个 episode reward 复制到对应的 layer steps，再用 V(s) 得到 advantage。
            returns = torch.from_numpy(np.repeat(rewards_np, self.resource_config.system.num_layers))
            advantages = returns - old_values
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # PPO 对同一批 on-policy 数据进行多轮 minibatch 更新。
            for _ in range(self.config.update_epochs):
                order = torch.randperm(len(all_actions))
                for start in range(0, len(order), self.config.minibatch_size):
                    index = order[start : start + self.config.minibatch_size]
                    # 用更新后的策略重新计算 log-probability/value，和 rollout 时的旧值比较。
                    output = self.model.evaluate(states_tensor[index], masks_tensor[index], actions_tensor[index])
                    ratio = torch.exp(output.log_probabilities - old_log_probs[index])
                    advantage = normalized_advantages[index]
                    # clipped objective 限制新旧策略比率，防止一次更新幅度过大。
                    policy_loss = -torch.minimum(
                        ratio * advantage,
                        torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                        * advantage,
                    ).mean()
                    # Critic 拟合 return；entropy bonus 鼓励 Actor 保留必要探索。
                    value_loss = torch.mean((output.values - returns[index]) ** 2)
                    loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * output.entropy.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
            # 只有 PPO 更新完成后才提交本轮进度，保证 checkpoint 位于一致边界。
            completed += rollout_size
            rollout_index += 1
            history["reward"].append(float(rewards_np.mean()))
            history["latency"].append(float(details["latency_seconds"].mean()))
            history["log_ppl_ratio"].append(float(details["log_ppl_ratio"].mean()))
            history["invalid_fraction"].append(float(details["invalid"].mean()))
            candidate_due = completed % candidate_interval_episodes == 0
            monitor = best_reward
            # 定期在固定 validation channel 上评估；只有更好的 reward 才覆盖 best model。
            if candidate_due or completed == self.config.training_episodes or rollout_index % self.config.validation_interval == 0:
                monitor = self._validation_reward(validation_channels)
                history["validation_reward"].append(monitor)
                if monitor > best_reward:
                    best_reward = monitor
                    best_episodes = completed
                    best_model_state = deepcopy(self.model.state_dict())
            if candidate_due:
                self._atomic_save(self._candidate_payload(completed, monitor), candidate_directory / f"episode_{completed:06d}.pth")
            # 每轮都保存可恢复状态，包含随机数状态和历史指标，支持中断后精确续训。
            self._atomic_save(
                {
                    "format_version": 1,
                    "policy_type": self.policy_type,
                    "ppo_config": asdict(self.config),
                    "resource_config": self.resource_config.to_dict(),
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_model_state": best_model_state,
                    "best_validation_reward": best_reward,
                    "best_episodes": best_episodes,
                    "completed_episodes": completed,
                    "rollout_index": rollout_index,
                    "history": history,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    'torch_cuda_random_states': (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    ),
                    "channel_rng_state": channel_rng.bit_generator.state,
                    "noise_rng_state": noise_rng.bit_generator.state,
                    "run_metadata": run_metadata,
                },
                state_path,
            )
            print(
                f"layerwise_episodes={completed:5d} reward={rewards_np.mean():.4f} "
                f"invalid={details['invalid'].mean():.3f} monitor={monitor:.4f}",
                flush=True,
            )
        self.model.load_state_dict(best_model_state)
        self._atomic_save(self._candidate_payload(best_episodes, best_reward), output_path)
        return history
