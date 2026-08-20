"""Candidate-pool selection utilities for layerwise policy inference."""

from __future__ import annotations

import numpy as np


def _normalized_relevance(rewards: np.ndarray) -> np.ndarray:
    minimum = float(np.min(rewards))
    maximum = float(np.max(rewards))
    span = maximum - minimum
    if span <= 1e-12:
        return np.ones_like(rewards, dtype=np.float64)
    return (rewards.astype(np.float64) - minimum) / span


def _hamming_distance(candidate: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Return layer-level assignment distance in [0, 1]."""

    return np.mean(candidate != selected[None, :], axis=1, dtype=np.float64)


def select_diverse_candidates(
    candidates: np.ndarray,
    rewards: np.ndarray,
    *,
    k: int,
    diversity_weight: float = 0.15,
    min_hamming_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select a reward-ranked but diverse Top-K subset with MMR.

    The highest-reward candidate is always selected first.  Subsequent
    candidates maximize a mixture of normalized surrogate relevance and their
    minimum Hamming distance to the already selected assignments.  The return
    order is sorted by surrogate reward so existing callers keep the meaning
    of rank 0 as the surrogate-selected Top-1 candidate.
    """

    candidate_array = np.asarray(candidates, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float64)
    if candidate_array.ndim != 2:
        raise ValueError("candidates must have shape (N, layers)")
    if reward_array.shape != (len(candidate_array),):
        raise ValueError("rewards must have shape (N,)")
    if not 1 <= k:
        raise ValueError("k must be positive")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between zero and one")
    if not 0.0 <= min_hamming_fraction <= 1.0:
        raise ValueError("min_hamming_fraction must be between zero and one")
    if len(candidate_array) == 0:
        raise ValueError("candidate pool cannot be empty")

    relevance = _normalized_relevance(reward_array)
    first = int(np.argmax(reward_array))
    selected = [first]
    remaining = np.ones(len(candidate_array), dtype=bool)
    remaining[first] = False
    target_count = min(k, len(candidate_array))
    while len(selected) < target_count:
        remaining_indices = np.flatnonzero(remaining)
        distances = np.ones(len(remaining_indices), dtype=np.float64)
        for selected_index in selected:
            distances = np.minimum(
                distances,
                _hamming_distance(
                    candidate_array[remaining_indices], candidate_array[selected_index]
                ),
            )
        eligible = distances >= min_hamming_fraction - 1e-12
        if not np.any(eligible):
            eligible = np.ones_like(distances, dtype=bool)
        scores = (1.0 - diversity_weight) * relevance[remaining_indices] + (
            diversity_weight * distances
        )
        scores = np.where(eligible, scores, -np.inf)
        best_position = int(np.argmax(scores))
        best_index = int(remaining_indices[best_position])
        selected.append(best_index)
        remaining[best_index] = False

    # Preserve the established rank contract: rank 0 is always the best
    # surrogate reward, while the selected set itself is diversity-aware.
    selected.sort(key=lambda index: (-reward_array[index], index))
    if len(selected) < k:
        selected.extend([selected[-1]] * (k - len(selected)))
    selected_indices = np.asarray(selected, dtype=np.int64)
    return (
        candidate_array[selected_indices],
        reward_array[selected_indices].astype(np.float32),
        selected_indices,
    )


def mean_pairwise_hamming(candidates: np.ndarray) -> float:
    """Return mean pairwise layer-assignment distance for diagnostics."""

    values = np.asarray(candidates, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("candidates must have shape (N, layers)")
    if len(values) < 2:
        return 0.0
    distances = [
        float(np.mean(values[first] != values[second]))
        for first in range(len(values))
        for second in range(first + 1, len(values))
    ]
    return float(np.mean(distances))


__all__ = ["mean_pairwise_hamming", "select_diverse_candidates"]
