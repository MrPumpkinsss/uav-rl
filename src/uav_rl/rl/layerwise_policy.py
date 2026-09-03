"""Autoregressive policy for arbitrary layer-to-UAV assignments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from uav_rl.resource_assignment import ResourceConstrainedConfig


def state_dimension(config: ResourceConstrainedConfig) -> int:
    """Channel plus prefix progress, resource usage, and previous-UAV features."""
    """UAV 之间的 channel quality 矩阵  当前处理进度，例如已经处理到第几层   每台 UAV 的资源使用情况和负载情况      上一个 layer 被分配到哪台 UAV 的 one-hot 向量"""
    uavs = config.system.num_uavs
    return uavs * uavs + 1 + 2 * uavs + uavs


def valid_layer_action_mask(
    *,
    layer_index: int,
    memory_used: np.ndarray,
    energy_used: np.ndarray,
    config: ResourceConstrainedConfig,
) -> np.ndarray:
    """Mask UAVs that preserve optimistic memory/energy completion feasibility."""

    layers = config.system.num_layers
    uavs = config.system.num_uavs
    if not 0 <= layer_index < layers:
        raise ValueError("layer_index is out of range")
    memory = np.asarray(memory_used, dtype=np.float64)
    energy = np.asarray(energy_used, dtype=np.float64)
    if memory.shape != (uavs,) or energy.shape != (uavs,):
        raise ValueError("resource usage vectors have the wrong shape")
    layer_memory = float(config.layer_memory_units[layer_index])
    compute_speeds = np.asarray(config.system.compute_speed, dtype=np.float64)
    capacities = np.asarray(config.uav_memory_capacity_units, dtype=np.float64)
    budgets = np.asarray(config.uav_energy_budget_joule, dtype=np.float64)
    hover = np.asarray(config.uav_hover_energy_joule, dtype=np.float64)
    mask = np.zeros(uavs, dtype=bool)
    remaining_memory = float(np.sum(config.layer_memory_units[layer_index + 1 :]))
    remaining_energy = float(
        np.sum(config.layer_compute_seconds_at_unit_speed[layer_index + 1 :])
        * config.compute_energy_coefficient
        * np.min(np.asarray(config.system.compute_speed, dtype=np.float64)) ** 2
    )
    for uav in range(uavs):
        new_memory = memory.copy()
        new_energy = energy.copy()
        new_memory[uav] += layer_memory
        new_energy[uav] += float(
            config.compute_energy_coefficient
            * compute_speeds[uav] ** 2
            * config.layer_compute_seconds_at_unit_speed[layer_index]
        )
        if new_memory[uav] > capacities[uav] + 1e-9:
            continue
        if new_energy[uav] + hover[uav] > budgets[uav] + 1e-9:
            continue
        if remaining_memory > float(np.maximum(capacities - new_memory, 0.0).sum()) + 1e-9:
            continue
        if remaining_energy > float(
            np.maximum(budgets - hover - new_energy, 0.0).sum()
        ) + 1e-9:
            continue
        mask[uav] = True
    if not mask.any():
        raise RuntimeError("no feasible UAV action remains for this layer prefix")
    return mask


def layer_state(
    normalized_channel: np.ndarray,
    *,
    layer_index: int,
    memory_used: np.ndarray,
    energy_used: np.ndarray,
    previous_uav: int | None,
    config: ResourceConstrainedConfig,
) -> np.ndarray:
    """Construct one autoregressive Markov state."""

    channel = np.asarray(normalized_channel, dtype=np.float32)
    uavs = config.system.num_uavs
    if channel.shape != (uavs, uavs):
        raise ValueError("normalized channel has the wrong shape")
    memory = np.asarray(memory_used, dtype=np.float32) / np.asarray(
        config.uav_memory_capacity_units, dtype=np.float32
    )
    energy = np.asarray(energy_used, dtype=np.float32) / np.asarray(
        config.uav_energy_budget_joule, dtype=np.float32
    )
    previous = np.zeros(uavs, dtype=np.float32)
    if previous_uav is not None:
        previous[previous_uav] = 1.0
    return np.concatenate(
        [
            channel.reshape(-1),
            np.asarray([layer_index / config.system.num_layers], dtype=np.float32),
            memory,
            energy,
            previous,
        ]
    )


@dataclass(frozen=True)
class LayerPolicyOutput:
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


class LayerwiseActorCritic(nn.Module):
    """Independent actor-critic that assigns one layer at a time."""

    def __init__(self, config: ResourceConstrainedConfig, hidden_dim: int = 256) -> None:
        """初始化 layerwise policy 网络，用于逐层选择 UAV。"""
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(         #状态编码器 输入的是当前资源分配状态：当前处理到了哪一层，每台 UAV 已经分配了多少计算资源，当前信道质量。。。
            nn.Linear(state_dimension(config), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, config.system.num_uavs)  #Actor 分支，负责选择 UAV
        self.value_head = nn.Linear(hidden_dim, 1)    #Critic 分支，输出一个标量，用来估计当前状态未来能够获得多大的累计 reward

    def _distribution(self, states: torch.Tensor, masks: torch.Tensor) -> Categorical:
        """执行 _distribution，完成本模块中的对应数据处理或实验步骤。"""
        if states.ndim != 2 or states.shape[1] != state_dimension(self.config):
            raise ValueError("layer states have the wrong shape")
        if masks.shape != (len(states), self.config.system.num_uavs):
            raise ValueError("layer action masks have the wrong shape")
        if not torch.all(masks.any(dim=1)):
            raise ValueError("each layer state needs at least one valid UAV")
        logits = self.actor_head(self.encoder(states))
        return Categorical(logits=logits.masked_fill(~masks, torch.finfo(logits.dtype).min))

    def sample(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> LayerPolicyOutput:
        """执行 sample，完成本模块中的对应数据处理或实验步骤。"""
        distribution = self._distribution(states, masks)
        actions = (
            torch.argmax(distribution.logits, dim=1)
            if deterministic
            else distribution.sample()
        )
        encoded = self.encoder(states)
        return LayerPolicyOutput(
            actions=actions,
            log_probabilities=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            values=self.value_head(encoded).squeeze(1),
        )

    def evaluate(
        self, states: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor
    ) -> LayerPolicyOutput:
        """评估一个 deployment 的资源约束、质量、时延和最终 reward。"""
        distribution = self._distribution(states, masks)
        if not torch.all(masks.gather(1, actions[:, None])):
            raise ValueError("provided layer action is invalid for its state")
        encoded = self.encoder(states)
        return LayerPolicyOutput(
            actions=actions,
            log_probabilities=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            values=self.value_head(encoded).squeeze(1),
        )
