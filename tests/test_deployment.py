import numpy as np
import pytest

from uav_rl.config import SystemConfig
from uav_rl.deployment import random_continuous_deployment, validate_deployment
from uav_rl.wireless import boundary_drop_probabilities, collaborative_latency


def test_random_deployments_are_contiguous_and_within_capacity() -> None:
    config = SystemConfig()
    rng = np.random.default_rng(123)
    for _ in range(100):
        validate_deployment(random_continuous_deployment(rng, config), config)


def test_disjoint_uav_interval_is_rejected() -> None:
    config = SystemConfig(num_layers=4, num_uavs=3, max_layers_per_uav=2, compute_speed=(1, 1, 1))
    with pytest.raises(ValueError, match="not contiguous"):
        validate_deployment(np.array([0, 1, 0, 2]), config)


def test_same_uav_boundary_has_no_drop_or_communication() -> None:
    config = SystemConfig(num_layers=4, max_layers_per_uav=4)
    deployment = np.zeros(4, dtype=np.int64)
    channel = np.full((5, 5), 10.0)
    assert np.all(boundary_drop_probabilities(deployment, channel, config) == 0)
    assert collaborative_latency(deployment, channel, config).communication_seconds == 0
