"""Diverse Top-K inference built on the existing layerwise PPO candidate pool."""

from __future__ import annotations

import numpy as np

from uav_rl.rl.candidate_selection import mean_pairwise_hamming, select_diverse_candidates


def diverse_top_k_deployments(
    trainer,
    channels: np.ndarray,
    *,
    k: int,
    candidate_pool_size: int | None = None,
    diversity_weight: float = 0.15,
    min_hamming_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate a larger PPO pool and select a diverse reward-ranked Top-K set."""

    if k < 1:
        raise ValueError("k must be positive")
    pool_size = candidate_pool_size or max(4 * k, k)
    if pool_size < k:
        raise ValueError("candidate_pool_size must be at least k")
    channels = np.asarray(channels, dtype=np.float32)
    pool, pool_rewards = trainer.top_k_deployments(
        channels, k=pool_size, samples_per_channel=pool_size
    )
    selected: list[np.ndarray] = []
    selected_rewards: list[np.ndarray] = []
    pool_diversity: list[float] = []
    selected_diversity: list[float] = []
    for channel_pool, channel_rewards in zip(pool, pool_rewards, strict=True):
        chosen, chosen_rewards, _ = select_diverse_candidates(
            channel_pool,
            channel_rewards,
            k=k,
            diversity_weight=diversity_weight,
            min_hamming_fraction=min_hamming_fraction,
        )
        selected.append(chosen)
        selected_rewards.append(chosen_rewards)
        pool_diversity.append(mean_pairwise_hamming(channel_pool))
        selected_diversity.append(mean_pairwise_hamming(chosen))
    diagnostics = {
        "candidate_pool_size": float(pool_size),
        "pool_mean_pairwise_hamming": float(np.mean(pool_diversity)),
        "selected_mean_pairwise_hamming": float(np.mean(selected_diversity)),
        "selected_surrogate_reward_mean": float(np.mean(selected_rewards)),
    }
    return np.stack(selected), np.stack(selected_rewards), diagnostics


__all__ = ["diverse_top_k_deployments"]
