"""Disjoint activation-noise seed streams for training and evaluation."""

from __future__ import annotations

import numpy as np

TRAIN_NOISE_SEED_RANGE = range(0, 1_000_000_000)
VALIDATION_NOISE_SEED_RANGE = range(1_000_000_000, 1_500_000_000)
TEST_NOISE_SEED_RANGE = range(1_500_000_000, 2_000_000_000)
DIAGNOSTIC_NOISE_SEED_RANGE = range(2_000_000_000, 2_500_000_000)


def _sample_unique(
    rng: np.random.Generator,
    seed_range: range,
    count: int,
) -> np.ndarray:
    if count < 1:
        raise ValueError("noise sample count must be positive")
    if count > len(seed_range):
        raise ValueError("noise sample count exceeds the available seed range")
    offsets = rng.choice(len(seed_range), size=count, replace=False)
    return np.asarray(offsets + seed_range.start, dtype=np.int64)


def sample_training_noise_seeds(
    rng: np.random.Generator,
    action_count: int,
    samples_per_action: int,
) -> np.ndarray:
    """Draw a unique training seed for each action/sample pair."""

    if action_count < 1:
        raise ValueError("action_count must be positive")
    seeds = _sample_unique(
        rng,
        TRAIN_NOISE_SEED_RANGE,
        action_count * samples_per_action,
    )
    return seeds.reshape(action_count, samples_per_action)


def validation_noise_seeds(count: int, generator_seed: int) -> np.ndarray:
    """Return the reproducible validation seed set."""

    return _sample_unique(
        np.random.default_rng(generator_seed),
        VALIDATION_NOISE_SEED_RANGE,
        count,
    )


def test_noise_seeds(count: int, generator_seed: int) -> np.ndarray:
    """Return the reproducible test seed set, disjoint from training and validation."""

    return _sample_unique(
        np.random.default_rng(generator_seed),
        TEST_NOISE_SEED_RANGE,
        count,
    )


def diagnostic_noise_seeds(count: int, generator_seed: int) -> np.ndarray:
    """Return a diagnostic-only stream disjoint from train/validation/test seeds."""

    return _sample_unique(
        np.random.default_rng(generator_seed),
        DIAGNOSTIC_NOISE_SEED_RANGE,
        count,
    )
