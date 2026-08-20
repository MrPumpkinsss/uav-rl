"""Segment-level policy primitives for independent multi-step deployment PPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from uav_rl.config import SystemConfig


def state_dimension(config: SystemConfig) -> int:
    """Return the flat channel-plus-partial-deployment state width."""

    return config.num_uavs * config.num_uavs + 1 + 2 * config.num_uavs


def action_dimension(config: SystemConfig) -> int:
    """Return the number of flattened ``(uav, segment_length)`` actions."""

    return config.num_uavs * config.max_layers_per_uav


def valid_action_mask(
    *,
    assigned_layers: int,
    used_uavs: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    """Mask actions that would violate continuity, capacity, or completion feasibility."""

    if not 0 <= assigned_layers < config.num_layers:
        raise ValueError("assigned layers must identify an unfinished deployment")
    used = np.asarray(used_uavs, dtype=bool)
    if used.shape != (config.num_uavs,):
        raise ValueError("used UAV mask has the wrong shape")

    remaining = config.num_layers - assigned_layers
    mask = np.zeros(action_dimension(config), dtype=bool)
    for uav in range(config.num_uavs):
        if used[uav]:
            continue
        available_after = config.num_uavs - int(used.sum()) - 1
        for length in range(1, config.max_layers_per_uav + 1):
            layers_after = remaining - length
            if layers_after < 0:
                continue
            if layers_after > available_after * config.max_layers_per_uav:
                continue
            mask[uav * config.max_layers_per_uav + length - 1] = True
    if not mask.any():
        raise RuntimeError("partial deployment has no feasible next segment")
    return mask


def decode_action(action: int, config: SystemConfig) -> tuple[int, int]:
    """Decode a flattened categorical action into ``(uav, segment_length)``."""

    if not 0 <= action < action_dimension(config):
        raise ValueError("segment action is out of range")
    return divmod(action, config.max_layers_per_uav)[0], action % config.max_layers_per_uav + 1


def deployment_segments(deployment: np.ndarray, config: SystemConfig) -> list[tuple[int, int]]:
    """Encode a valid continuous deployment as its ordered non-repeated segments."""

    values = np.asarray(deployment, dtype=np.int64)
    if values.shape != (config.num_layers,):
        raise ValueError("deployment has the wrong number of layers")
    if np.any(values < 0) or np.any(values >= config.num_uavs):
        raise ValueError("deployment contains an invalid UAV index")

    segments: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        segments.append((int(values[start]), end - start))
        start = end
    if any(length > config.max_layers_per_uav for _, length in segments):
        raise ValueError("deployment segment exceeds UAV capacity")
    uavs = [uav for uav, _ in segments]
    if len(uavs) != len(set(uavs)):
        raise ValueError("deployment assigns disjoint segments to one UAV")
    return segments


def partial_state(
    normalized_channel: np.ndarray,
    *,
    assigned_layers: int,
    used_uavs: np.ndarray,
    previous_uav: int | None,
    config: SystemConfig,
) -> np.ndarray:
    """Build the Markov state for one partial segment deployment."""

    channel = np.asarray(normalized_channel, dtype=np.float32)
    if channel.shape != (config.num_uavs, config.num_uavs):
        raise ValueError("normalized channel has the wrong shape")
    used = np.asarray(used_uavs, dtype=np.float32)
    previous = np.zeros(config.num_uavs, dtype=np.float32)
    if previous_uav is not None:
        previous[previous_uav] = 1.0
    return np.concatenate(
        [
            channel.reshape(-1),
            np.asarray([assigned_layers / config.num_layers], dtype=np.float32),
            used,
            previous,
        ]
    )


@dataclass(frozen=True)
class SegmentPolicyOutput:
    """Categorical action, log probability, entropy, and state value."""

    actions: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor


class SegmentActorCritic(nn.Module):
    """Independent actor-critic that chooses one legal deployment segment per step."""

    def __init__(self, config: SystemConfig, hidden_dim: int = 256) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(state_dimension(config), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dimension(config))
        self.value_head = nn.Linear(hidden_dim, 1)

    def _distribution(self, states: torch.Tensor, action_masks: torch.Tensor) -> Categorical:
        if states.ndim != 2 or states.shape[1] != state_dimension(self.config):
            raise ValueError("segment states have the wrong shape")
        if action_masks.shape != (len(states), action_dimension(self.config)):
            raise ValueError("segment action masks have the wrong shape")
        if not torch.all(action_masks.any(dim=1)):
            raise ValueError("each segment state needs at least one valid action")
        logits = self.actor_head(self.encoder(states))
        return Categorical(logits=logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min))

    def sample(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> SegmentPolicyOutput:
        distribution = self._distribution(states, action_masks)
        actions = (
            torch.argmax(distribution.logits, dim=1)
            if deterministic
            else distribution.sample()
        )
        return SegmentPolicyOutput(
            actions=actions,
            log_probabilities=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            values=self.value_head(self.encoder(states)).squeeze(1),
        )

    def evaluate(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> SegmentPolicyOutput:
        distribution = self._distribution(states, action_masks)
        if not torch.all(action_masks.gather(1, actions[:, None])):
            raise ValueError("provided segment action is invalid for its state")
        return SegmentPolicyOutput(
            actions=actions,
            log_probabilities=distribution.log_prob(actions),
            entropy=distribution.entropy(),
            values=self.value_head(self.encoder(states)).squeeze(1),
        )
