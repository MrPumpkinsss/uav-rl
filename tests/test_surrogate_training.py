"""Metrics, regret, and fake-data end-to-end tests for ensemble training."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from uav_rl.config import SystemConfig
from uav_rl.surrogate import PPLSurrogateEnsemble, load_surrogate
from uav_rl.surrogate_training import (
    EnsembleTrainingConfig,
    SurrogateAcceptanceCriteria,
    _verify_dataset_manifest,
    grouped_reward_regret,
    regression_metrics,
    train_and_evaluate_ensemble,
)


def test_dataset_manifest_rejects_failed_isolation_or_hash_mismatch(tmp_path: Path) -> None:
    split = tmp_path / "split.npz"
    split.write_bytes(b"dataset")
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    paths = {"train": split, "validation": split, "test": split}
    manifest = {
        "format_version": 2,
        "isolation_audit": {"passed": False},
        "splits": {name: {"sha256": digest} for name in paths},
    }
    with pytest.raises(ValueError, match="isolation audit"):
        _verify_dataset_manifest(manifest, paths)

    manifest["isolation_audit"] = {"passed": True}
    manifest["splits"]["test"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="test dataset SHA256"):
        _verify_dataset_manifest(manifest, paths)


def test_regression_metrics_include_spearman_and_error_quantiles() -> None:
    target = np.asarray([0.0, 1.0, 2.0, 3.0])
    prediction = np.asarray([0.1, 0.9, 2.2, 2.8])

    metrics = regression_metrics(target, prediction)

    assert metrics["mae"] == pytest.approx(np.mean([0.1, 0.1, 0.2, 0.2]))
    assert metrics["spearman"] == 1.0
    assert metrics["absolute_error_p95"] <= metrics["absolute_error_max"]


def test_grouped_regret_uses_true_reward_of_predicted_choice() -> None:
    result = grouped_reward_regret(
        target=np.asarray([0.1, 0.2, 0.4, 0.3]),
        prediction=np.asarray([0.3, 0.1, 0.3, 0.4]),
        latency_seconds=np.ones(4),
        group_ids=np.asarray(["a", "a", "b", "b"]),
        system=SystemConfig(),
        latency_reference_seconds=1.0,
    )

    assert result["group_count"] == 2
    assert result["mean_reward_regret_fraction"] > 0.0


def _write_split(path: Path, rng: np.random.Generator, count: int) -> None:
    drops = rng.uniform(0.0, 0.4, size=(count, 3)).astype(np.float32)
    target = (drops.sum(axis=1) + 0.2 * drops[:, 0] ** 2).astype(np.float32)
    np.savez_compressed(
        path,
        drop_probabilities=drops,
        log_ppl_ratio=target,
        sample_source=np.asarray(["random" if index % 2 else "coverage" for index in range(count)]),
        group_ids=np.asarray([f"group-{index // 2}" for index in range(count)]),
        latency_seconds=np.linspace(0.8, 1.2, count, dtype=np.float32),
    )


def test_fake_data_end_to_end_training_writes_recoverable_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    test = tmp_path / "test.npz"
    _write_split(train, rng, 24)
    _write_split(validation, rng, 8)
    _write_split(test, rng, 8)
    manifest = tmp_path / "manifest.json"
    split_paths = {"train": train, "validation": validation, "test": test}
    manifest.write_text(
        json.dumps(
            {
                "format_version": 2,
                "dataset_fingerprint": "fake-hash",
                "isolation_audit": {
                    "passed": True,
                    "pairwise_overlap_counts": {
                        "train_vs_validation": {
                            "noise_seed": 0,
                            "channel": 0,
                            "deployment": 1,
                            "channel_deployment_pair": 0,
                            "drop_vector": 0,
                        }
                    },
                    "deployment_vector_note": "Structural reuse is not context leakage.",
                },
                "quality_evaluator": {
                    "model_id": "fake",
                    "clean_perplexity": 1.0,
                    "evaluated_sequences": 1,
                    "evaluated_tokens": 1,
                },
                "ppo_training_context": {
                    "replay_verified_actions": 1000,
                    "sha256": "fake-context-hash",
                },
                "splits": {
                    name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in split_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "ensemble.pth"
    metrics = tmp_path / "metrics.json"
    report = tmp_path / "report.md"

    result = train_and_evaluate_ensemble(
        train_path=train,
        validation_path=validation,
        test_path=test,
        dataset_manifest_path=manifest,
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        report_path=report,
        plot_directory=tmp_path / "plots",
        training_config=EnsembleTrainingConfig(
            member_count=2, hidden_dim=8, epochs=8, patience=3
        ),
        acceptance_criteria=SurrogateAcceptanceCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.0,
    )

    loaded = load_surrogate(checkpoint)
    assert isinstance(loaded, PPLSurrogateEnsemble)
    assert result["samples"] == {"train": 24, "validation": 8, "test": 8}
    assert result["test_metrics"]["worst_error_region"]["count"] == 1
    assert metrics.exists() and report.exists()
    assert len(list((tmp_path / "plots").glob("*.png"))) == 3
