"""Shared finite-horizon MDP state for layer-to-UAV RL algorithms.

One episode assigns all model layers for one fixed channel matrix.  The action at
step ``t`` is the UAV that executes layer ``t``.  Memory, compute energy and a
configurable boundary cap are enforced before an action is exposed to a policy.
The expensive PPL/latency reward is evaluated once after the final assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.rl.layerwise_policy import layer_state, valid_layer_action_mask

def apply_boundary_budget_mask(
    mask: np.ndarray,
    *,
    layer_index: int,
    memory_used: np.ndarray,
    energy_used: np.ndarray,
    previous_uav: int | None,
    boundary_count: int,
    max_boundaries: int,
    config: ResourceConstrainedConfig,
) -> np.ndarray:
    """Apply the repository's feasibility-preserving boundary freeze rule.

    After ``max_boundaries`` switches, the policy stays on the current UAV when
    that UAV is still resource-feasible.  If it is no longer feasible, the base
    resource mask is retained so the deployment can finish instead of entering
    a dead end.  Consequently this parameter is a *freeze threshold*, not a
    mathematically strict cap; evaluation must report the realized boundary
    count and threshold-exceedance fraction.
    """

    del layer_index, memory_used, energy_used, config
    limited = np.asarray(mask, dtype=bool).copy()
    if (
        previous_uav is not None
        and boundary_count >= max_boundaries
        and limited[previous_uav]
    ):
        limited[:] = False
        limited[previous_uav] = True
    return limited

@dataclass
class LayerwiseEpisode:
    """Mutable prefix state shared by PPO, A2C and DQN comparisons.

    Parameters
    ----------
    normalized_channel:
        Channel matrix scaled to the interval used by the policy encoder.
    config:
        Resource and system constraints for the layer assignment problem.
    max_boundaries:
        Boundary freeze threshold shared by all algorithms. It is not a strict
        cap when resource feasibility forces another switch.
    """

    normalized_channel: np.ndarray
    config: ResourceConstrainedConfig
    max_boundaries: int = 4
    layer_index: int = 0
    previous_uav: int | None = None
    boundary_count: int = 0
    memory_used: np.ndarray = field(init=False)
    energy_used: np.ndarray = field(init=False)
    deployment: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        channel = np.asarray(self.normalized_channel, dtype=np.float32)
        uavs = self.config.system.num_uavs
        if channel.shape != (uavs, uavs):
            raise ValueError("normalized_channel has the wrong shape")
        if not 0 <= self.max_boundaries < self.config.system.num_layers:
            raise ValueError("max_boundaries is out of range")
        self.normalized_channel = channel
        self.memory_used = np.zeros(uavs, dtype=np.float64)
        self.energy_used = np.zeros(uavs, dtype=np.float64)
        self.deployment = np.full(self.config.system.num_layers, -1, dtype=np.int64)

    @property
    def done(self) -> bool:
        """Whether every model layer has been assigned."""

        return self.layer_index == self.config.system.num_layers

    def observation_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the Markov observation and the currently feasible UAV mask."""

        if self.done:
            raise RuntimeError("the layerwise episode is already complete")
        observation = layer_state(
            self.normalized_channel,
            layer_index=self.layer_index,
            memory_used=self.memory_used,
            energy_used=self.energy_used,
            previous_uav=self.previous_uav,
            config=self.config,
        )
        mask = valid_layer_action_mask(
            layer_index=self.layer_index,
            memory_used=self.memory_used,
            energy_used=self.energy_used,
            config=self.config,
        )
        mask = apply_boundary_budget_mask(
            mask,
            layer_index=self.layer_index,
            memory_used=self.memory_used,
            energy_used=self.energy_used,
            previous_uav=self.previous_uav,
            boundary_count=self.boundary_count,
            max_boundaries=self.max_boundaries,
            config=self.config,
        )
        return observation, mask

    def step(self, action: int) -> None:
        """Append one feasible UAV action and update prefix resource usage."""

        _, mask = self.observation_and_mask()
        if not 0 <= action < self.config.system.num_uavs or not bool(mask[action]):
            raise ValueError(f"UAV action {action} is infeasible at layer {self.layer_index}")
        layer = self.layer_index
        if self.previous_uav is not None and action != self.previous_uav:
            self.boundary_count += 1
        self.deployment[layer] = action
        self.memory_used[action] += float(self.config.layer_memory_units[layer])
        speed = float(self.config.system.compute_speed[action])
        self.energy_used[action] += float(
            self.config.compute_energy_coefficient
            * speed**2
            * self.config.layer_compute_seconds_at_unit_speed[layer]
        )
        self.previous_uav = action
        self.layer_index += 1

    def completed_deployment(self) -> np.ndarray:
        """Return a copy of the full assignment, raising if the episode is incomplete."""

        if not self.done:
            raise RuntimeError("cannot read deployment before all layers are assigned")
        return self.deployment.copy()
