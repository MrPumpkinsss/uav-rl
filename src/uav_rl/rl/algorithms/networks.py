"""Neural-network primitives used by non-PPO RL baselines."""

from __future__ import annotations

import torch
from torch import nn

from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.rl.layerwise_policy import state_dimension


class LayerwiseQNetwork(nn.Module):
    """Estimate one Q-value per UAV for the sequential layer assignment MDP."""

    def __init__(self, config: ResourceConstrainedConfig, hidden_dim: int = 256) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(state_dimension(config), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.system.num_uavs),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.shape[1] != state_dimension(self.config):
            raise ValueError("states have the wrong shape")
        return self.network(states)

    def masked_q_values(self, states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Set infeasible UAV actions to negative infinity before action selection."""

        values = self(states)
        if masks.shape != values.shape:
            raise ValueError("action masks have the wrong shape")
        if not torch.all(masks.any(dim=1)):
            raise ValueError("every state must expose at least one feasible action")
        return values.masked_fill(~masks, torch.finfo(values.dtype).min)
