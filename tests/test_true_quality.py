from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_rl.config import DataGenerationConfig
from uav_rl.true_quality import TruePPLQualityEvaluator


def _lightweight_evaluator() -> TruePPLQualityEvaluator:
    evaluator = TruePPLQualityEvaluator.__new__(TruePPLQualityEvaluator)
    evaluator.generation = replace(DataGenerationConfig(), noise_seed=7)
    evaluator.num_boundaries = 2
    evaluator._evaluate_one = lambda probabilities, noise_seed: (  # type: ignore[method-assign]
        float(np.sum(probabilities)) + noise_seed
    )
    return evaluator


def test_cache_key_includes_noise_seed() -> None:
    probabilities = np.asarray([0.1, 0.2], dtype=np.float32)

    assert TruePPLQualityEvaluator._key(probabilities, 1) != TruePPLQualityEvaluator._key(
        probabilities, 2
    )


def test_shared_and_per_action_noise_seed_averaging() -> None:
    evaluator = _lightweight_evaluator()
    probabilities = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    shared = evaluator.evaluate(probabilities, noise_seeds=np.asarray([10, 20]))
    per_action = evaluator.evaluate(
        probabilities,
        noise_seeds=np.asarray([[1, 3], [5, 9]]),
    )

    assert shared == pytest.approx([15.3, 15.7])
    assert per_action == pytest.approx([2.3, 7.7])


@pytest.mark.parametrize(
    "noise_seeds",
    [np.empty((0,), dtype=np.int64), np.ones((3, 2), dtype=np.int64), np.asarray([1.5])],
)
def test_invalid_noise_seed_shapes_and_types_are_rejected(noise_seeds: np.ndarray) -> None:
    evaluator = _lightweight_evaluator()
    probabilities = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    with pytest.raises(ValueError, match="noise_seeds"):
        evaluator.evaluate(probabilities, noise_seeds=noise_seeds)


def test_version_one_cache_metadata_is_rejected(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    cache_path.touch()
    cache_path.with_suffix(".jsonl.meta.json").write_text(
        '{"format_version": 1, "generation": {}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata"):
        TruePPLQualityEvaluator._validate_cache_metadata(cache_path, DataGenerationConfig())


def test_cuda_allocator_cleanup_only_runs_for_cuda_evaluators(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = TruePPLQualityEvaluator.__new__(TruePPLQualityEvaluator)
    calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(True))

    evaluator.device = torch.device("cpu")
    evaluator._release_unused_cuda_memory()
    evaluator.device = torch.device("cuda")
    evaluator._release_unused_cuda_memory()

    assert calls == [True]
