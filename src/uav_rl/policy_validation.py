"""True-model validation of frozen surrogate-PPO policy candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.rl.environment import DeploymentEnvironment
from uav_rl.rl.policy import ContinuousDeploymentActorCritic


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy_candidate(
    path: Path,
    system: SystemConfig,
) -> ContinuousDeploymentActorCritic:
    """Load one frozen PPO policy candidate with its recorded hidden width."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in payload or "ppo_config" not in payload:
        raise ValueError(f"{path} is not a PPO policy checkpoint")
    hidden_dim = int(payload["ppo_config"]["hidden_dim"])
    policy = ContinuousDeploymentActorCritic(system, hidden_dim)
    policy.load_state_dict(payload["model_state"])
    return policy.eval()


def _summary(
    rewards: np.ndarray,
    details: dict[str, np.ndarray],
) -> dict[str, float]:
    mean = float(rewards.mean())
    standard_deviation = float(rewards.std(ddof=0))
    standard_error = standard_deviation / np.sqrt(len(rewards))
    return {
        "reward_mean": mean,
        "reward_std": standard_deviation,
        "reward_standard_error": float(standard_error),
        "reward_95_ci_low": float(mean - 1.96 * standard_error),
        "reward_95_ci_high": float(mean + 1.96 * standard_error),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
    }


def evaluate_policy_candidates(
    *,
    environment: DeploymentEnvironment,
    candidate_paths: list[Path],
    channels: np.ndarray,
    noise_seeds: np.ndarray,
) -> dict[str, Any]:
    """Evaluate candidates on identical true-model channels and seed samples."""

    if not candidate_paths:
        raise ValueError("at least one candidate checkpoint is required")
    states = torch.from_numpy(
        environment.normalize_channels(channels).reshape(len(channels), -1)
    )
    candidates: list[dict[str, Any]] = []
    reward_vectors: list[np.ndarray] = []
    for path in candidate_paths:
        policy = load_policy_candidate(path, environment.config)
        with torch.no_grad():
            deployments = policy.sample(states, deterministic=True).actions.cpu().numpy()
        rewards, details = environment.evaluate(
            channels, deployments, noise_seeds=noise_seeds
        )
        candidates.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "episodes": int(
                    torch.load(path, map_location="cpu", weights_only=False).get(
                        "episodes", -1
                    )
                ),
                "metrics": _summary(rewards, details),
            }
        )
        reward_vectors.append(rewards.astype(np.float64, copy=False))
    best_index = max(
        range(len(candidates)),
        key=lambda index: candidates[index]["metrics"]["reward_mean"],
    )
    selected = candidates[best_index]
    for index, candidate in enumerate(candidates):
        delta = reward_vectors[index] - reward_vectors[best_index]
        candidate["paired_delta_to_selected"] = {
            "reward_mean": float(delta.mean()),
            "reward_standard_error": float(delta.std(ddof=0) / np.sqrt(len(delta))),
        }
    return {
        "candidates": candidates,
        "selected_index": best_index,
        "selected": selected,
    }


def checkpoint_ppo_config(path: Path) -> PPOConfig:
    """Read a candidate's PPO configuration for validation provenance checks."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return PPOConfig(**payload["ppo_config"])
