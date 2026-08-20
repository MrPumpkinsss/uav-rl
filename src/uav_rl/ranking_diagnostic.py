"""Local surrogate-ranking diagnostics around structured baseline deployments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from uav_rl.baselines import strong_link_baseline
from uav_rl.config import SystemConfig


@dataclass(frozen=True)
class NeighborhoodCandidates:
    """Fixed candidate labels and one valid deployment per channel/label pair."""

    labels: tuple[str, ...]
    deployments: np.ndarray


def strong_link_neighborhood(channels: np.ndarray, config: SystemConfig) -> NeighborhoodCandidates:
    """Build strong-link plus adjacent swaps and one-UAV replacements.

    The present capacity constraint requires four full eight-layer segments. Each
    candidate therefore differs only in the ordered four-UAV path, isolating the
    decision surface where conservative PPO is expected to find improvements.
    """

    required = int(np.ceil(config.num_layers / config.max_layers_per_uav))
    if config.num_layers != required * config.max_layers_per_uav:
        raise ValueError("diagnostic currently requires full equal-capacity segments")
    if required != config.num_uavs - 1:
        raise ValueError("diagnostic currently expects exactly one unused UAV")
    baseline = strong_link_baseline(channels, config)
    labels = ("strong_link", "swap_01", "swap_12", "swap_23", "replace_0", "replace_1", "replace_2", "replace_3")
    all_candidates: list[np.ndarray] = []
    for deployment in baseline:
        order = deployment[:: config.max_layers_per_uav].copy()
        unused = int(next(iter(set(range(config.num_uavs)) - set(order.tolist()))))
        orders = [order]
        for left in range(required - 1):
            swapped = order.copy()
            swapped[left], swapped[left + 1] = swapped[left + 1], swapped[left]
            orders.append(swapped)
        for position in range(required):
            replaced = order.copy()
            replaced[position] = unused
            orders.append(replaced)
        all_candidates.append(
            np.stack([np.repeat(candidate, config.max_layers_per_uav) for candidate in orders])
        )
    return NeighborhoodCandidates(labels=labels, deployments=np.stack(all_candidates).astype(np.int64))


def summarize_local_ranking(
    labels: tuple[str, ...],
    surrogate_rewards: np.ndarray,
    true_rewards: np.ndarray,
) -> dict[str, object]:
    """Measure local ranking agreement and baseline-relative regret.

    Rewards use the common convention that higher is better. The first candidate
    must be the strong-link baseline.
    """

    surrogate = np.asarray(surrogate_rewards, dtype=np.float64)
    truth = np.asarray(true_rewards, dtype=np.float64)
    if surrogate.shape != truth.shape or surrogate.ndim != 2:
        raise ValueError("surrogate and true rewards must have identical (channel, candidate) shape")
    if surrogate.shape[1] != len(labels) or labels[0] != "strong_link":
        raise ValueError("candidate labels must begin with strong_link")
    top_surrogate = surrogate.argmax(axis=1)
    top_true = truth.argmax(axis=1)
    agreements: list[bool] = []
    for left, right in combinations(range(len(labels)), 2):
        surrogate_delta = surrogate[:, left] - surrogate[:, right]
        true_delta = truth[:, left] - truth[:, right]
        non_tied = (surrogate_delta != 0.0) & (true_delta != 0.0)
        agreements.extend((np.sign(surrogate_delta[non_tied]) == np.sign(true_delta[non_tied])).tolist())
    true_best = truth.max(axis=1)
    source_metrics: dict[str, dict[str, float]] = {}
    for index, label in enumerate(labels):
        source_metrics[label] = {
            "surrogate_reward_mean": float(surrogate[:, index].mean()),
            "true_reward_mean": float(truth[:, index].mean()),
            "surrogate_improvement_to_strong_link_mean": float(
                (surrogate[:, index] - surrogate[:, 0]).mean()
            ),
            "true_improvement_to_strong_link_mean": float((truth[:, index] - truth[:, 0]).mean()),
        }
    return {
        "channel_count": int(len(surrogate)),
        "candidate_labels": list(labels),
        "pairwise_comparison_count": len(agreements),
        "pairwise_win_rate_agreement": float(np.mean(agreements)) if agreements else None,
        "top1_agreement": float(np.mean(top_surrogate == top_true)),
        "surrogate_selected_true_regret_mean": float(
            (true_best - truth[np.arange(len(truth)), top_surrogate]).mean()
        ),
        "surrogate_selected_true_regret_p90": float(
            np.quantile(true_best - truth[np.arange(len(truth)), top_surrogate], 0.9)
        ),
        "strong_link_true_regret_mean": float((true_best - truth[:, 0]).mean()),
        "strong_link_true_regret_p90": float(np.quantile(true_best - truth[:, 0], 0.9)),
        "per_candidate": source_metrics,
    }
