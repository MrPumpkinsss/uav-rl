"""Common-channel evaluation for PPO and deterministic baselines."""

from __future__ import annotations

import math
from typing import Any

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


def _summarize(
    rewards: np.ndarray,
    details: dict[str, np.ndarray],
    clean_perplexity: float,
) -> dict[str, float]:
    predicted_ppl = clean_perplexity * np.exp(details["log_ppl_ratio"])
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
        "latency_std_seconds": float(details["latency_seconds"].std()),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "predicted_ppl_mean": float(predicted_ppl.mean()),
        "predicted_ppl_std": float(predicted_ppl.std()),
    }


def evaluate_methods(
    environment: DeploymentEnvironment,
    policy: ContinuousDeploymentActorCritic,
    channels: np.ndarray,
    clean_perplexity: float,
    random_seed: int,
) -> dict[str, Any]:
    """Evaluate every method on exactly the same channel tensor."""

    states = torch.from_numpy(environment.normalize_channels(channels).reshape(len(channels), -1))
    with torch.no_grad():
        ppo_deployments = policy.sample(states, deterministic=True).actions.cpu().numpy()
    deployments = {
        "ppo": ppo_deployments,
        "random": random_baseline(channels, random_seed, environment.config),
        "compute_greedy": np.repeat(
            compute_greedy_baseline(channels, environment.config)[None, :],
            len(channels),
            axis=0,
        ),
        "strong_link": strong_link_baseline(channels, environment.config),
        "dynamic_programming": dynamic_programming_baseline(
            channels,
            environment.config,
            environment.latency_reference,
        ),
        "four_segment_surrogate_oracle": four_segment_surrogate_oracle(
            channels,
            environment,
        ),
        "full_surrogate_oracle": full_surrogate_oracle(
            channels,
            environment,
        ),
    }
    results: dict[str, Any] = {}
    for name, method_deployments in deployments.items():
        rewards, details = environment.evaluate(channels, method_deployments)
        results[name] = _summarize(rewards, details, clean_perplexity)

    ppo_reward = results["ppo"]["reward_mean"]
    for baseline in (
        "random",
        "compute_greedy",
        "strong_link",
        "dynamic_programming",
        "four_segment_surrogate_oracle",
        "full_surrogate_oracle",
    ):
        denominator = abs(results[baseline]["reward_mean"])
        results["ppo"][f"reward_improvement_vs_{baseline}_percent"] = (
            100.0 * (ppo_reward - results[baseline]["reward_mean"]) / denominator
            if not math.isclose(denominator, 0.0)
            else 0.0
        )
    return results
