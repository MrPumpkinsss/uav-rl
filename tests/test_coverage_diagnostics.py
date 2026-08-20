from __future__ import annotations

import numpy as np
import pytest

from uav_rl.coverage_diagnostics import (
    build_feature_matrix,
    nearest_neighbor_profile,
    normalize_features,
    spearman_correlation,
)


def test_feature_matrix_matches_boundary_plus_engineered_layout() -> None:
    drops = np.array([[0.0, 0.25], [0.5, 0.0]], dtype=np.float32)
    features = build_feature_matrix(drops)

    assert features.shape == (2, 7)
    assert np.allclose(features[:, :2], drops)
    assert np.allclose(features[:, 2], [0.25, 0.5])
    assert np.allclose(features[:, 3], [0.25, 0.5])
    assert np.allclose(features[:, 5], [0.5, 0.5])


def test_nearest_neighbor_profile_excludes_self_and_is_batched() -> None:
    features = np.array([[0.0], [1.0], [3.0], [10.0]], dtype=np.float64)
    indices, distances = nearest_neighbor_profile(
        features,
        features,
        neighbors=2,
        exclude_matching_index=True,
        batch_size=2,
    )

    assert indices.shape == distances.shape == (4, 2)
    assert np.all(indices[:, 0] != np.arange(4))
    assert np.allclose(distances[:, 0], [1.0, 1.0, 2.0, 7.0])


def test_normalization_uses_train_statistics_and_scale_floor() -> None:
    train = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float64)
    query = np.array([[2.0, 5.1]], dtype=np.float64)
    normalized_train, normalized_query = normalize_features(train, query)

    assert np.allclose(normalized_train.mean(axis=0), 0.0)
    assert normalized_query[0, 1] == pytest.approx(5.0)


def test_spearman_correlation_is_rank_based() -> None:
    assert spearman_correlation(np.array([1, 2, 3]), np.array([10, 20, 30])) == pytest.approx(1.0)
    assert spearman_correlation(np.array([1, 2, 3]), np.array([30, 20, 10])) == pytest.approx(-1.0)
