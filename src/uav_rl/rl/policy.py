"""Autoregressive actor enforcing contiguous per-UAV layer allocation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from uav_rl.config import SystemConfig


@dataclass(frozen=True)
class PolicyOutput:
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


class ContinuousDeploymentActorCritic(nn.Module):
    """Choose one UAV per layer while preventing disjoint intervals per UAV."""

    def __init__(self, config: SystemConfig, hidden_dim: int = 256) -> None:
        super().__init__()
        self.config = config
        state_dim = config.num_uavs * config.num_uavs
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_embedding = nn.Embedding(config.num_uavs + 1, hidden_dim // 4)
        self.decoder = nn.GRUCell(hidden_dim // 4 + 1, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, config.num_uavs)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _valid_mask(
        self,
        used: torch.Tensor,
        current: torch.Tensor,
        run_length: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        batch_size = used.size(0)
        mask = torch.zeros(batch_size, self.config.num_uavs, dtype=torch.bool, device=used.device)
        remaining = self.config.num_layers - layer - 1
        for batch in range(batch_size):
            for candidate in range(self.config.num_uavs):
                is_first = current[batch] < 0
                is_current = candidate == int(current[batch])
                is_unused = not bool(used[batch, candidate])
                if not is_first and not is_current and not is_unused:
                    continue
                new_run = int(run_length[batch]) + 1 if is_current else 1
                if new_run > self.config.max_layers_per_uav:
                    continue
                new_used = used[batch].clone()
                new_used[candidate] = True
                unused_count = int((~new_used).sum())
                available = self.config.max_layers_per_uav - new_run
                available += unused_count * self.config.max_layers_per_uav
                if available >= remaining:
                    mask[batch, candidate] = True
        if not torch.all(mask.any(dim=1)):
            raise RuntimeError("no feasible UAV remains for a deployment prefix")
        return mask

    def _decode(
        self,
        states: torch.Tensor,
        provided_actions: torch.Tensor | None,
        deterministic: bool,
    ) -> PolicyOutput:
        batch_size = states.size(0)
        hidden = self.state_encoder(states)
        values = self.value_head(hidden).squeeze(-1)
        used = torch.zeros(batch_size, self.config.num_uavs, dtype=torch.bool, device=states.device)
        current = torch.full((batch_size,), -1, dtype=torch.long, device=states.device)
        run_length = torch.zeros(batch_size, dtype=torch.long, device=states.device)
        previous = torch.full(
            (batch_size,), self.config.num_uavs, dtype=torch.long, device=states.device
        )
        actions: list[torch.Tensor] = []
        log_probabilities = torch.zeros(batch_size, device=states.device)
        entropy = torch.zeros(batch_size, device=states.device)

        for layer in range(self.config.num_layers):
            layer_position = torch.full(
                (batch_size, 1), layer / max(1, self.config.num_layers - 1), device=states.device
            )
            decoder_input = torch.cat((self.action_embedding(previous), layer_position), dim=1)
            hidden = self.decoder(decoder_input, hidden)
            logits = self.actor_head(hidden)
            valid_mask = self._valid_mask(used, current, run_length, layer)
            distribution = Categorical(logits=logits.masked_fill(~valid_mask, -torch.inf))
            if provided_actions is not None:
                action = provided_actions[:, layer]
                if not torch.all(valid_mask.gather(1, action[:, None])):
                    raise ValueError("provided action violates deployment continuity or capacity")
            elif deterministic:
                action = torch.argmax(distribution.logits, dim=1)
            else:
                action = distribution.sample()

            log_probabilities += distribution.log_prob(action)
            entropy += distribution.entropy()
            actions.append(action)
            same_uav = action == current
            run_length = torch.where(same_uav, run_length + 1, torch.ones_like(run_length))
            used.scatter_(1, action[:, None], True)
            current = action
            previous = action

        return PolicyOutput(
            actions=torch.stack(actions, dim=1),
            log_probabilities=log_probabilities,
            entropy=entropy,
            values=values,
        )

    def sample(self, states: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        return self._decode(states, provided_actions=None, deterministic=deterministic)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> PolicyOutput:
        return self._decode(states, provided_actions=actions, deterministic=False)
