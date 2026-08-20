"""Tests for resumable multi-seed surrogate collection and split isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from uav_rl.config import SystemConfig
from uav_rl.data.surrogate_dataset import (
    SurrogateAction,
    _aggregate_fresh_actions,
    _load_ppo_training_context,
    _validate_action_plan,
    _validate_imported_context_isolation,
    collect_surrogate_samples,
)
from uav_rl.wireless import boundary_drop_probabilities


class FakeEvaluator:
    def __init__(self) -> None:
        self.clean_perplexity = 10.0
        self.model_forwards = 0

    def evaluate(
        self, probabilities: np.ndarray, *, noise_seeds: np.ndarray | None = None
    ) -> np.ndarray:
        assert noise_seeds is not None
        self.model_forwards += probabilities.shape[0]
        return probabilities.sum(axis=1) + noise_seeds.astype(np.float32) * 1e-4


def _action(action_id: str, split: str, value: float, seeds: list[int]) -> dict[str, object]:
    return {
        "action_id": action_id,
        "split": split,
        "source": "fake",
        "group_id": f"group-{action_id}",
        "channel": (np.eye(2) + value).tolist(),
        "deployment": [0, 1],
        "drop_probabilities": [value],
        "latency_seconds": 1.0,
        "noise_seeds": seeds,
    }


def test_collection_resumes_without_duplicate_evaluation(tmp_path: Path) -> None:
    plan = {
        "config_fingerprint": "fingerprint",
        "actions": [_action("a", "train", 0.1, [1, 2])],
    }
    cache = tmp_path / "samples.jsonl"
    evaluator = FakeEvaluator()

    first = collect_surrogate_samples(
        plan=plan, evaluator=evaluator, sample_cache_path=cache, progress_interval=0
    )
    second = collect_surrogate_samples(
        plan=plan, evaluator=evaluator, sample_cache_path=cache, progress_interval=0
    )

    assert first["new_samples"] == 2
    assert second["new_samples"] == 0
    assert evaluator.model_forwards == 2
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2


def test_collection_rejects_incompatible_fingerprint(tmp_path: Path) -> None:
    cache = tmp_path / "samples.jsonl"
    metadata = cache.with_suffix(".jsonl.meta.json")
    metadata.write_text(
        json.dumps({"format_version": 2, "config_fingerprint": "old"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible"):
        collect_surrogate_samples(
            plan={"config_fingerprint": "new", "actions": []},
            evaluator=FakeEvaluator(),
            sample_cache_path=cache,
        )


def test_multiseed_aggregation_uses_mean_std_and_count() -> None:
    plan = {"actions": [_action("a", "train", 0.1, [1, 2, 3])]}
    records = [
        {"action_id": "a", "noise_seed": seed, "log_ppl_ratio": value}
        for seed, value in ((1, 1.0), (2, 2.0), (3, 3.0))
    ]

    row = _aggregate_fresh_actions(plan, records)["train"][0]

    assert np.mean(row["qualities"]) == pytest.approx(2.0)
    assert np.std(row["qualities"]) == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert len(row["noise_seeds"]) == 3


def test_action_plan_rejects_cross_split_seed_or_drop_overlap() -> None:
    def make(split: str, value: float, seed: int) -> SurrogateAction:
        return SurrogateAction(
            action_id=split,
            split=split,  # type: ignore[arg-type]
            source="fake",
            group_id=split,
            channel=[[1.0, value], [value, 1.0]],
            deployment=[0, 1],
            drop_probabilities=[value],
            latency_seconds=1.0,
            noise_seeds=[seed],
        )

    actions = [make("train", 0.1, 1), make("validation", 0.2, 2), make("test", 0.3, 3)]
    _validate_action_plan(actions)
    with pytest.raises(ValueError, match="overlapping noise seeds"):
        _validate_action_plan([actions[0], make("validation", 0.2, 1), actions[2]])


def test_imported_ppo_context_is_checked_against_held_out_pairs() -> None:
    imported = [_action("ppo", "train", 0.1, [1, 2, 3, 4])]
    held_out = [_action("eval", "test", 0.2, [5, 6])]
    _validate_imported_context_isolation(imported, held_out)

    collision = dict(held_out[0])
    collision["channel"] = imported[0]["channel"]
    collision["deployment"] = imported[0]["deployment"]
    with pytest.raises(ValueError, match="channels"):
        _validate_imported_context_isolation(imported, [collision])

    seed_collision = dict(held_out[0])
    seed_collision["noise_seeds"] = [4, 5]
    with pytest.raises(ValueError, match="overlapping noise seeds"):
        _validate_imported_context_isolation(imported, [seed_collision])


def test_ppo_context_requires_matching_replay_metadata_hashes(tmp_path: Path) -> None:
    system = SystemConfig(
        num_layers=2,
        num_uavs=2,
        max_layers_per_uav=1,
        compute_speed=(1.0, 1.0),
    )
    gains = np.linspace(2.0, 20.0, 1000, dtype=np.float32)
    channels = np.repeat(np.eye(2, dtype=np.float32)[None, :, :] * 20.0, 1000, axis=0)
    channels[:, 0, 1] = gains
    channels[:, 1, 0] = gains
    deployments = np.repeat(np.asarray([[0, 1]], dtype=np.int64), 1000, axis=0)
    drops = np.stack(
        [
            boundary_drop_probabilities(deployment, channel, system)
            for channel, deployment in zip(channels, deployments, strict=True)
        ]
    )
    context_path = tmp_path / "context.npz"
    np.savez_compressed(
        context_path,
        channels=channels,
        deployments=deployments,
        drop_probabilities=drops,
    )
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text("", encoding="utf-8")
    metadata = {
        "format_version": 1,
        "actions": 1000,
        "unique_drop_vectors": 1000,
        "history_exact": True,
        "drop_set_exact": True,
        "context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "ppl_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
    }
    metadata_path = context_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = _load_ppo_training_context(context_path, cache_path, system)
    assert len(loaded) == 1000

    metadata["context_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="context SHA256"):
        _load_ppo_training_context(context_path, cache_path, system)
