"""Tests for reward-ranked diverse Top-K candidate selection."""

import numpy as np
import pytest

from uav_rl.rl.candidate_selection import (
    mean_pairwise_hamming,
    select_diverse_candidates,
)


def test_zero_diversity_matches_reward_order() -> None:
    candidates = np.asarray([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int64)
    rewards = np.asarray([-0.4, -0.2, -0.3], dtype=np.float32)
    selected, selected_rewards, indices = select_diverse_candidates(
        candidates, rewards, k=2, diversity_weight=0.0
    )
    np.testing.assert_array_equal(indices, np.asarray([1, 2]))
    np.testing.assert_array_equal(selected, candidates[[1, 2]])
    np.testing.assert_allclose(selected_rewards, rewards[[1, 2]])


def test_diversity_can_choose_lower_reward_different_assignment() -> None:
    candidates = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    rewards = np.asarray([-0.10, -0.11, -0.30], dtype=np.float32)
    selected, _, indices = select_diverse_candidates(
        candidates, rewards, k=2, diversity_weight=0.8
    )
    assert int(indices[0]) == 0
    assert int(indices[1]) == 2
    np.testing.assert_array_equal(selected[1], candidates[2])


def test_min_hamming_fraction_is_soft_when_pool_cannot_satisfy_it() -> None:
    candidates = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
    rewards = np.asarray([-0.1, -0.2], dtype=np.float32)
    selected, _, _ = select_diverse_candidates(
        candidates, rewards, k=3, min_hamming_fraction=1.0
    )
    assert selected.shape == (3, 2)
    np.testing.assert_array_equal(selected[0], candidates[0])


def test_invalid_selector_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="diversity_weight"):
        select_diverse_candidates(np.zeros((1, 2)), np.zeros(1), k=1, diversity_weight=1.1)
    with pytest.raises(ValueError, match="candidate pool"):
        select_diverse_candidates(np.empty((0, 2)), np.empty(0), k=1)


def test_mean_pairwise_hamming() -> None:
    candidates = np.asarray([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    assert mean_pairwise_hamming(candidates) == pytest.approx(2.0 / 3.0)
