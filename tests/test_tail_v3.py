"""Tests for tail-v3 data sharding, weighted training, and validation gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_rl.config import SystemConfig
from uav_rl.data.tail_dataset import (
    TailSeedExtensionConfig,
    _audit_fields,
    _combine_summary_splits,
    _field_sets_from_npz,
    aggregate_tail_seed_extension_dataset,
    build_tail_seed_extension_plan,
)
from uav_rl.data.surrogate_dataset import canonical_json_hash
from uav_rl.surrogate import PPLSurrogateEnsemble, load_surrogate
from uav_rl.surrogate_training import (
    EnsembleTrainingConfig,
    _sample_weights,
    train_and_validate_ensemble,
)
from uav_rl.tail_training import TailValidationCriteria, assess_tail_validation


def _write_split(
    path: Path,
    *,
    start: float,
    sources: list[str],
    seeds_per_action: int,
) -> None:
    count = len(sources)
    drops = np.stack(
        [
            np.asarray([start + index * 0.01, 0.1, 0.2], dtype=np.float32)
            for index in range(count)
        ]
    )
    target = drops.sum(axis=1).astype(np.float32)
    channels = np.stack(
        [
            np.asarray(
                [[10.0 + start + index, start], [start, 10.0 + start + index]],
                dtype=np.float32,
            )
            for index in range(count)
        ]
    )
    deployments = np.stack(
        [np.asarray([index, index + 1], dtype=np.int64) for index in range(count)]
    )
    noise = np.arange(
        int(start * 10000),
        int(start * 10000) + count * seeds_per_action,
        dtype=np.int64,
    ).reshape(count, seeds_per_action)
    np.savez_compressed(
        path,
        action_ids=np.asarray([f"action-{start}-{index}" for index in range(count)]),
        sample_source=np.asarray(sources),
        group_ids=np.asarray([f"group-{index // 2}" for index in range(count)]),
        channels=channels,
        deployments=deployments,
        drop_probabilities=drops,
        latency_seconds=np.linspace(0.8, 1.2, count, dtype=np.float32),
        noise_seeds=noise,
        log_ppl_ratio=target,
        log_ppl_ratio_std=np.linspace(0.1, 0.4, count, dtype=np.float32),
        noise_seed_count=np.full(count, seeds_per_action, dtype=np.int16),
        has_context=np.ones(count, dtype=np.bool_),
    )


def test_combined_summary_supports_four_and_eight_seed_shards(tmp_path: Path) -> None:
    base = tmp_path / "base_train.npz"
    delta = tmp_path / "tail_delta.npz"
    combined = tmp_path / "combined.npz"
    _write_split(base, start=0.01, sources=["random", "tail"], seeds_per_action=4)
    _write_split(
        delta,
        start=0.51,
        sources=["tail_hazard", "tail_disagreement"],
        seeds_per_action=8,
    )

    metadata = _combine_summary_splits([base, delta], combined)

    assert metadata["actions"] == 4
    assert metadata["noise_seeds_per_action_min"] == 4
    assert metadata["noise_seeds_per_action_max"] == 8
    with np.load(combined, allow_pickle=False) as data:
        assert "noise_seeds" not in data.files
        assert data["noise_seed_count"].tolist() == [4, 4, 8, 8]


def test_tail_seed_extension_reuses_actions_and_aggregates_sixteen_seeds(
    tmp_path: Path,
) -> None:
    base_paths = {
        "train": tmp_path / "base_train.npz",
        "validation": tmp_path / "base_validation.npz",
        "test": tmp_path / "base_test.npz",
    }
    _write_split(base_paths["train"], start=0.01, sources=["random"], seeds_per_action=4)
    _write_split(
        base_paths["validation"],
        start=0.41,
        sources=["random"],
        seeds_per_action=4,
    )
    _write_split(base_paths["test"], start=0.81, sources=["random"], seeds_per_action=4)
    train_action = {
        "action_id": "train-v3-tail_hazard-0000",
        "split": "train",
        "source": "tail_hazard",
        "group_id": "train-v3-tail_hazard-0000",
        "channel": [[20.0, 1.0], [1.0, 20.0]],
        "deployment": [0, 1],
        "drop_probabilities": [0.15, 0.05, 0.0],
        "latency_seconds": 1.1,
        "noise_seeds": list(range(100_000, 100_008)),
    }
    validation_action = {
        "action_id": "validation-v3-tail_hazard-0000",
        "split": "validation",
        "source": "tail_hazard",
        "group_id": "validation-v3-tail_hazard-0000",
        "channel": [[21.0, 2.0], [2.0, 21.0]],
        "deployment": [1, 0],
        "drop_probabilities": [0.25, 0.05, 0.0],
        "latency_seconds": 1.2,
        "noise_seeds": list(range(200_000, 200_016)),
    }
    original_plan = {
        "format_version": 3,
        "stage": "development",
        "actions": [train_action, validation_action],
    }
    development_plan_path = tmp_path / "development_plan.json"
    development_plan_path.write_text(json.dumps(original_plan), encoding="utf-8")
    train_delta = tmp_path / "train_delta.npz"
    validation_delta = tmp_path / "validation_delta.npz"
    combined_validation = tmp_path / "combined_validation.npz"
    _write_split(train_delta, start=0.21, sources=["tail_hazard"], seeds_per_action=8)
    _write_split(
        validation_delta,
        start=0.61,
        sources=["tail_hazard"],
        seeds_per_action=16,
    )
    _combine_summary_splits(
        [base_paths["validation"], validation_delta], combined_validation
    )
    development = {
        "format_version": 3,
        "stage": "development",
        "generation": {},
        "system": {},
        "tail_config": {},
        "quality_evaluator": {},
        "plan": {
            "path": str(development_plan_path),
            "sha256": canonical_json_hash(original_plan),
        },
        "base_v2": {
            "splits": {
                split: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for split, path in base_paths.items()
            }
        },
        "selector_checkpoint": {},
        "delta_splits": {
            "train": {"path": str(train_delta), "actions": 1},
            "validation": {"path": str(validation_delta), "actions": 1},
        },
        "splits": {
            "train": {"path": str(train_delta), "actions": 2},
            "validation": {
                "path": str(combined_validation),
                "actions": 2,
                "sha256": hashlib.sha256(combined_validation.read_bytes()).hexdigest(),
            },
        },
        "diagnostic_test": {"path": str(base_paths["test"])},
        "isolation_audit": {"passed": True},
        "dataset_fingerprint": "development-test-fingerprint",
    }
    development_manifest_path = tmp_path / "development_manifest.json"
    development_manifest_path.write_text(json.dumps(development), encoding="utf-8")
    extension_plan_path = tmp_path / "extension_plan.json"
    extension_plan = build_tail_seed_extension_plan(
        development_manifest_path=development_manifest_path,
        development_plan_path=development_plan_path,
        plan_path=extension_plan_path,
        config=TailSeedExtensionConfig(),
    )

    assert extension_plan["extension_audit"]["passed"]
    assert extension_plan["extension_audit"]["additional_samples"] == 8
    assert extension_plan["actions"][0]["drop_probabilities"] == train_action[
        "drop_probabilities"
    ]
    assert not set(extension_plan["actions"][0]["noise_seeds"]) & set(
        train_action["noise_seeds"]
    )

    development_cache = tmp_path / "development.jsonl"
    extension_cache = tmp_path / "extension.jsonl"
    development_records = [
        {"action_id": action["action_id"], "noise_seed": seed, "log_ppl_ratio": 1.0}
        for action in original_plan["actions"]
        for seed in action["noise_seeds"]
    ]
    extension_records = [
        {"action_id": train_action["action_id"], "noise_seed": seed, "log_ppl_ratio": 3.0}
        for seed in extension_plan["actions"][0]["noise_seeds"]
    ]
    development_cache.write_text(
        "\n".join(json.dumps(record) for record in development_records) + "\n",
        encoding="utf-8",
    )
    extension_cache.write_text(
        "\n".join(json.dumps(record) for record in extension_records) + "\n",
        encoding="utf-8",
    )
    manifest = aggregate_tail_seed_extension_dataset(
        extension_plan=extension_plan,
        extension_cache_path=extension_cache,
        development_manifest_path=development_manifest_path,
        development_plan_path=development_plan_path,
        development_cache_path=development_cache,
        output_directory=tmp_path,
        quality_evaluator_metadata={},
    )

    assert manifest["stage"] == "development_seed_extension"
    assert manifest["isolation_audit"]["passed"]
    with np.load(manifest["delta_splits"]["train"]["path"], allow_pickle=False) as data:
        assert data["noise_seed_count"].tolist() == [16]
        assert data["log_ppl_ratio"].tolist() == pytest.approx([2.0])


def test_tail_isolation_audit_rejects_seed_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    test = tmp_path / "test.npz"
    _write_split(train, start=0.01, sources=["tail"], seeds_per_action=4)
    _write_split(validation, start=0.51, sources=["tail_hazard"], seeds_per_action=4)
    _write_split(test, start=0.81, sources=["random"], seeds_per_action=4)
    fields = {
        "train": _field_sets_from_npz([train]),
        "validation": _field_sets_from_npz([validation]),
        "test": _field_sets_from_npz([test]),
    }
    assert _audit_fields(fields)["passed"]
    shared = next(iter(fields["train"]["noise_seed"]))
    fields["validation"]["noise_seed"].add(shared)
    audit = _audit_fields(fields)
    assert not audit["passed"]
    assert audit["pairwise_overlap_counts"]["train_vs_validation"]["noise_seed"] == 1


def test_source_and_variance_weights_are_finite_and_tail_grouped() -> None:
    split = {
        "log_ppl_ratio": np.ones(6, dtype=np.float32),
        "sample_source": np.asarray(
            ["random", "random", "random", "tail", "tail_hazard", "tail_boundary"]
        ),
        "log_ppl_ratio_std": np.asarray(
            [0.1, 0.2, 0.3, 0.4, 0.8, 1.0], dtype=np.float32
        ),
        "noise_seed_count": np.asarray([4, 4, 4, 4, 8, 8], dtype=np.int16),
    }
    config = EnsembleTrainingConfig(
        source_balancing=True,
        variance_weighting=True,
    )

    weights = _sample_weights(split, config)

    assert np.isfinite(weights).all()
    assert weights.mean() == pytest.approx(1.0)
    assert weights.max() / weights.min() <= config.maximum_sample_weight**2


def test_validation_gate_combines_tail_and_non_tail_requirements() -> None:
    baseline = {
        "per_source": {
            "random": {"mae": 0.10},
            "coverage": {"mae": 0.15},
            "tail": {"mae": 0.18},
        }
    }
    validation = {
        "mae": 0.09,
        "per_source": {
            "random": {"mae": 0.11},
            "coverage": {"mae": 0.16},
        },
    }
    gate = assess_tail_validation(
        validation_metrics=validation,
        tail_metrics={"mae": 0.11, "spearman": 0.92},
        baseline_validation_metrics=baseline,
        criteria=TailValidationCriteria(),
    )
    assert gate["passed"]

    validation["per_source"]["random"]["mae"] = 0.14
    failed = assess_tail_validation(
        validation_metrics=validation,
        tail_metrics={"mae": 0.11, "spearman": 0.92},
        baseline_validation_metrics=baseline,
        criteria=TailValidationCriteria(),
    )
    assert not failed["checks"]["non_tail_regression"]


def test_validation_only_training_writes_loadable_checkpoint(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    _write_split(
        train,
        start=0.01,
        sources=["random", "random", "tail", "tail_hazard"] * 6,
        seeds_per_action=4,
    )
    _write_split(
        validation,
        start=0.51,
        sources=["random", "tail", "tail_boundary", "tail_disagreement"] * 2,
        seeds_per_action=8,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 3,
                "dataset_fingerprint": "tail-v3-test",
                "isolation_audit": {"passed": True},
                "splits": {
                    "train": {
                        "sha256": hashlib.sha256(train.read_bytes()).hexdigest()
                    },
                    "validation": {
                        "sha256": hashlib.sha256(validation.read_bytes()).hexdigest()
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "ensemble.pth"
    metrics = tmp_path / "metrics.json"

    result = train_and_validate_ensemble(
        train_path=train,
        validation_path=validation,
        dataset_manifest_path=manifest,
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        training_config=EnsembleTrainingConfig(
            member_count=2,
            hidden_dim=16,
            epochs=30,
            patience=8,
            loss_kind="huber",
            source_balancing=True,
            variance_weighting=True,
        ),
        system=SystemConfig(),
        latency_reference_seconds=1.0,
        device_name="cpu",
    )

    assert result["selection_stage"] == "validation_only"
    loaded = load_surrogate(checkpoint)
    assert isinstance(loaded, PPLSurrogateEnsemble)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "test" not in payload["data_files"]
