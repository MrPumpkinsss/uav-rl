"""Exact reward oracle over grouped layer assignments.

Full arbitrary enumeration is exponential in 32 layers. The diagnostic groups
adjacent layers into a small number of super-layers, expands every grouped
assignment back to the original 32-layer action, filters exact resource
constraints, and evaluates the same surrogate or true-LLM environment. Thus the
reported optimum is exact only inside the explicitly stated grouped action set.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from uav_rl.resource_assignment import validate_layerwise_deployment
from uav_rl.resource_environment import ResourceDeploymentEnvironment


@dataclass(frozen=True)
class ExactOracleResult:
    """Best grouped deployment and enumeration accounting."""

    deployment: np.ndarray
    reward: float
    feasible_assignments: int
    total_assignments: int
    num_groups: int


def _expand_group_assignment(assignment: tuple[int, ...], num_layers: int) -> np.ndarray:
    groups = len(assignment)
    cuts = np.linspace(0, num_layers, groups + 1, dtype=np.int64)
    deployment = np.empty(num_layers, dtype=np.int64)
    for index, uav in enumerate(assignment):
        deployment[cuts[index] : cuts[index + 1]] = uav
    return deployment


def exact_grouped_reward_oracle(
    channel: np.ndarray,
    environment: ResourceDeploymentEnvironment,
    *,
    num_groups: int = 8,
    batch_size: int = 4096,
    max_assignments: int = 1_000_000,
) -> ExactOracleResult:
    """Enumerate and score every feasible grouped assignment exactly.

    Repeated UAV identifiers are allowed, matching the repository's general
    layer-assignment model. The function is intentionally guarded against an
    accidental full ``U^32`` run.
    """

    system = environment.config.system
    if not 1 <= num_groups <= system.num_layers:
        raise ValueError("num_groups must be between 1 and num_layers")
    if batch_size < 1 or max_assignments < 1:
        raise ValueError("batch_size and max_assignments must be positive")
    total = system.num_uavs**num_groups
    if total > max_assignments:
        raise ValueError(
            f"grouped oracle requires {total} assignments, above max_assignments={max_assignments}"
        )
    values = np.asarray(channel, dtype=np.float32)
    expected = (system.num_uavs, system.num_uavs)
    if values.shape != expected:
        raise ValueError(f"channel must have shape {expected}")

    best_reward = -np.inf
    best_deployment: np.ndarray | None = None
    feasible = 0
    pending: list[np.ndarray] = []

    def score_pending() -> None:
        nonlocal best_reward, best_deployment
        if not pending:
            return
        deployments = np.stack(pending)
        channels = np.repeat(values[None, ...], len(deployments), axis=0)
        rewards, _ = environment.evaluate(channels, deployments)
        index = int(np.argmax(rewards))
        reward = float(rewards[index])
        if reward > best_reward:
            best_reward = reward
            best_deployment = deployments[index].copy()
        pending.clear()

    for assignment in itertools.product(range(system.num_uavs), repeat=num_groups):
        deployment = _expand_group_assignment(assignment, system.num_layers)
        try:
            validate_layerwise_deployment(deployment, environment.config, channel=values)
        except ValueError:
            continue
        feasible += 1
        pending.append(deployment)
        if len(pending) >= batch_size:
            score_pending()
    score_pending()
    if best_deployment is None:
        raise RuntimeError("grouped oracle found no feasible assignment")
    return ExactOracleResult(
        deployment=best_deployment,
        reward=best_reward,
        feasible_assignments=feasible,
        total_assignments=total,
        num_groups=num_groups,
    )
