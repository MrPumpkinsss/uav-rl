from __future__ import annotations

import numpy as np
import pytest

from uav_rl.config import SystemConfig
from uav_rl.ranking_diagnostic import strong_link_neighborhood, summarize_local_ranking


def test_strong_link_neighborhood_contains_eight_valid_four_segment_candidates() -> None:
    config = SystemConfig()
    channels = np.full((2, config.num_uavs, config.num_uavs), 5.0, dtype=np.float32)
    candidates = strong_link_neighborhood(channels, config)

    assert candidates.labels[0] == "strong_link"
    assert candidates.deployments.shape == (2, 8, config.num_layers)
    for deployment in candidates.deployments.reshape(-1, config.num_layers):
        segments = deployment.reshape(4, config.max_layers_per_uav)
        assert np.all(segments == segments[:, :1])
        assert len(np.unique(segments[:, 0])) == 4


def test_local_ranking_metrics_report_top1_and_regret() -> None:
    labels = ("strong_link", "swap_01", "swap_12")
    surrogate = np.array([[0.0, 0.4, 0.1], [0.0, 0.1, 0.4]])
    truth = np.array([[0.0, 0.2, 0.1], [0.0, 0.3, 0.2]])

    metrics = summarize_local_ranking(labels, surrogate, truth)

    assert metrics["top1_agreement"] == 0.5
    assert metrics["surrogate_selected_true_regret_mean"] == pytest.approx(0.05)
    assert metrics["strong_link_true_regret_mean"] == pytest.approx(0.25)
