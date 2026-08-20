from __future__ import annotations

import numpy as np

from uav_rl.noise_seeds import (
    DIAGNOSTIC_NOISE_SEED_RANGE,
    TEST_NOISE_SEED_RANGE,
    TRAIN_NOISE_SEED_RANGE,
    VALIDATION_NOISE_SEED_RANGE,
    diagnostic_noise_seeds,
    sample_training_noise_seeds,
    test_noise_seeds as make_test_noise_seeds,
    validation_noise_seeds,
)


def test_training_validation_test_and_diagnostic_seed_ranges_are_disjoint() -> None:
    training = sample_training_noise_seeds(np.random.default_rng(1), 12, 4)
    validation = validation_noise_seeds(16, 2)
    test = make_test_noise_seeds(16, 3)
    diagnostic = diagnostic_noise_seeds(16, 4)

    assert training.shape == (12, 4)
    assert len(np.unique(training)) == training.size
    assert np.all((training >= TRAIN_NOISE_SEED_RANGE.start) & (training < TRAIN_NOISE_SEED_RANGE.stop))
    assert np.all(
        (validation >= VALIDATION_NOISE_SEED_RANGE.start)
        & (validation < VALIDATION_NOISE_SEED_RANGE.stop)
    )
    assert np.all((test >= TEST_NOISE_SEED_RANGE.start) & (test < TEST_NOISE_SEED_RANGE.stop))
    assert np.all(
        (diagnostic >= DIAGNOSTIC_NOISE_SEED_RANGE.start)
        & (diagnostic < DIAGNOSTIC_NOISE_SEED_RANGE.stop)
    )
    assert not (set(training.ravel()) | set(validation) | set(test)) & set(diagnostic)


def test_fixed_held_out_seed_sets_are_reproducible() -> None:
    assert np.array_equal(validation_noise_seeds(8, 12), validation_noise_seeds(8, 12))
    assert np.array_equal(make_test_noise_seeds(8, 13), make_test_noise_seeds(8, 13))
    assert np.array_equal(diagnostic_noise_seeds(8, 14), diagnostic_noise_seeds(8, 14))
