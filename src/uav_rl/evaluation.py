"""Common-channel evaluation for PPO and deterministic baselines."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch

from uav_rl.baselines import (
    compute_greedy_baseline,
    dynamic_programming_baseline,
    random_baseline,
    strong_link_baseline,
)
from uav_rl.rl.environment import DeploymentEnvironment
from uav_rl.rl.oracle import four_segment_surrogate_oracle, full_surrogate_oracle
from uav_rl.rl.policy import ContinuousDeploymentActorCritic

MethodName = Literal[
    "ppo",
    "random",
    "compute_greedy",
    "strong_link",
    "dynamic_programming",
    "four_segment_surrogate_oracle",
    "full_surrogate_oracle",
]
STANDARD_METHODS: tuple[MethodName, ...] = (
    "ppo",
    "random",
    "compute_greedy",
    "strong_link",
    "dynamic_programming",
)
ALL_METHODS: tuple[MethodName, ...] = STANDARD_METHODS + (
    "four_segment_surrogate_oracle",
    "full_surrogate_oracle",
)


def _summarize(
    rewards: np.ndarray,
    details: dict[str, np.ndarray],
    clean_perplexity: float,
) -> dict[str, float]:
    perplexity = clean_perplexity * np.exp(details["log_ppl_ratio"])
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
        "latency_std_seconds": float(details["latency_seconds"].std()),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "ppl_mean": float(perplexity.mean()),
        "ppl_std": float(perplexity.std()),
    }


def evaluate_methods(
    environment: DeploymentEnvironment,
    policy: ContinuousDeploymentActorCritic,
    channels: np.ndarray,
    clean_perplexity: float,
    random_seed: int,
    method_names: tuple[MethodName, ...] = ALL_METHODS,
    noise_seeds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate methods on common channels and common activation-noise seeds."""

    if unknown_methods := set(method_names) - set(ALL_METHODS):
        raise ValueError(f"unknown evaluation methods: {sorted(unknown_methods)}")
    if "ppo" not in method_names:
        raise ValueError("method_names must include ppo for comparison reporting")

    states = torch.from_numpy(environment.normalize_channels(channels).reshape(len(channels), -1))
    with torch.no_grad():
        ppo_deployments = policy.sample(states, deterministic=True).actions.cpu().numpy()
    builders: dict[MethodName, Callable[[], np.ndarray]] = {
        "ppo": lambda: ppo_deployments,
        "random": lambda: random_baseline(channels, random_seed, environment.config),
        "compute_greedy": lambda: np.repeat(
            compute_greedy_baseline(channels, environment.config)[None, :],
            len(channels),
            axis=0,
        ),
        "strong_link": lambda: strong_link_baseline(channels, environment.config),
        "dynamic_programming": lambda: dynamic_programming_baseline(
            channels,
            environment.config,
            environment.latency_reference,
        ),
        "four_segment_surrogate_oracle": lambda: four_segment_surrogate_oracle(
            channels, environment
        ),
        "full_surrogate_oracle": lambda: full_surrogate_oracle(channels, environment),
    }
    deployments = {name: builders[name]() for name in method_names}
    results: dict[str, Any] = {}
    for name, method_deployments in deployments.items():
        rewards, details = environment.evaluate(
            channels,
            method_deployments,
            noise_seeds=noise_seeds,
        )
        results[name] = _summarize(rewards, details, clean_perplexity)

    ppo_reward = results["ppo"]["reward_mean"]
    for baseline in method_names:
        if baseline == "ppo":
            continue
        denominator = abs(results[baseline]["reward_mean"])
        results["ppo"][f"reward_improvement_vs_{baseline}_percent"] = (
            100.0 * (ppo_reward - results[baseline]["reward_mean"]) / denominator
            if not math.isclose(denominator, 0.0)
            else 0.0
        )
    return results
