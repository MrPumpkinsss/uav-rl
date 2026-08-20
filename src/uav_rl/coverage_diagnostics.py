"""Feature-space coverage diagnostics for frozen surrogate development data.

The diagnostic distinguishes three quantities that are often conflated:
label precision (multi-seed standard error), local target variation in the
training set, and distance from a frozen validation action to its nearest
training actions. It does not train a model or access final-test data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.surrogate import load_surrogate
from uav_rl.surrogate_training import _load_split, _verify_dataset_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_feature_matrix(drop_probabilities: np.ndarray) -> np.ndarray:
    """Create the same 31 + 5 physical features used by ``PPLSurrogate``."""

    drops = np.asarray(drop_probabilities, dtype=np.float64)
    if drops.ndim != 2 or drops.shape[1] < 1:
        raise ValueError("drop probabilities must be a non-empty [actions, boundaries] matrix")
    total = drops.sum(axis=1, keepdims=True)
    maximum = drops.max(axis=1, keepdims=True)
    square_sum = np.square(drops).sum(axis=1, keepdims=True)
    boundary_fraction = (drops > 0.0).mean(axis=1, keepdims=True)
    cumulative_hazard = -np.log1p(-np.minimum(drops, 1.0 - 1e-6)).sum(
        axis=1, keepdims=True
    )
    return np.concatenate(
        [
            drops,
            total,
            maximum,
            square_sum,
            boundary_fraction,
            cumulative_hazard,
        ],
        axis=1,
    )


def normalize_features(
    train_features: np.ndarray,
    query_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize using train-only statistics and the surrogate's scale floor."""

    mean = train_features.mean(axis=0)
    scale = np.maximum(train_features.std(axis=0), 0.02)
    return (train_features - mean) / scale, (query_features - mean) / scale


def nearest_neighbor_profile(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    neighbors: int = 8,
    exclude_matching_index: bool = False,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return neighbor indices and Euclidean distances without a dense NxN matrix."""

    query = np.asarray(query_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference features must be two-dimensional with equal width")
    maximum_neighbors = reference.shape[0] - int(exclude_matching_index)
    if not 1 <= neighbors <= maximum_neighbors:
        raise ValueError("invalid number of nearest neighbors")
    indices = np.empty((query.shape[0], neighbors), dtype=np.int64)
    distances = np.empty((query.shape[0], neighbors), dtype=np.float64)
    for start in range(0, query.shape[0], batch_size):
        stop = min(start + batch_size, query.shape[0])
        chunk = query[start:stop]
        squared = np.square(chunk[:, None, :] - reference[None, :, :]).sum(axis=2)
        if exclude_matching_index:
            if query.shape[0] != reference.shape[0]:
                raise ValueError("self-exclusion requires query and reference lengths to match")
            local = np.arange(stop - start)
            squared[local, start + local] = np.inf
        nearest = np.argpartition(squared, kth=neighbors - 1, axis=1)[:, :neighbors]
        nearest_distance = np.take_along_axis(squared, nearest, axis=1)
        order = np.argsort(nearest_distance, axis=1)
        indices[start:stop] = np.take_along_axis(nearest, order, axis=1)
        distances[start:stop] = np.sqrt(np.take_along_axis(nearest_distance, order, axis=1))
    return indices, distances


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute deterministic Spearman correlation without a SciPy dependency."""

    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    denominator = left_rank.std() * right_rank.std()
    return float(np.corrcoef(left_rank, right_rank)[0, 1]) if denominator else float("nan")


def _source_summary(
    *,
    sources: np.ndarray,
    label_standard_error: np.ndarray,
    nearest_distances: np.ndarray,
    local_target_error: np.ndarray | None = None,
    observed_absolute_error: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in sorted(set(sources.astype(str))):
        mask = sources.astype(str) == source
        summary: dict[str, Any] = {
            "actions": int(mask.sum()),
            "label_standard_error": _quantiles(label_standard_error[mask]),
            "nearest_train_distance": _quantiles(nearest_distances[mask]),
        }
        if local_target_error is not None:
            summary["local_target_absolute_difference"] = _quantiles(
                local_target_error[mask]
            )
        if observed_absolute_error is not None:
            summary["frozen_residual_absolute_error"] = _quantiles(
                observed_absolute_error[mask]
            )
            summary["error_distance_spearman"] = spearman_correlation(
                nearest_distances[mask], observed_absolute_error[mask]
            )
        result[source] = summary
    return result


def _frozen_prediction(
    checkpoint_path: Path,
    drops: np.ndarray,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(device_name)
    model = load_surrogate(checkpoint_path, device=device)
    model.eval()
    with torch.no_grad():
        mean, uncertainty = model.predict_with_uncertainty(
            torch.from_numpy(drops.astype(np.float32, copy=False)).to(device)
        )
    return mean.cpu().numpy(), uncertainty.cpu().numpy()


def _priority_rows(
    train_by_source: dict[str, dict[str, Any]],
    validation_by_source: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank additions for investigation; validation error is observation only."""

    rows: list[dict[str, Any]] = []
    for source, train in train_by_source.items():
        validation = validation_by_source.get(source)
        if validation is None:
            continue
        rows.append(
            {
                "source": source,
                "train_actions": train["actions"],
                "train_label_se_p90": train["label_standard_error"]["p90"],
                "train_nearest_distance_p90": train["nearest_train_distance"]["p90"],
                "train_local_target_difference_p90": train[
                    "local_target_absolute_difference"
                ]["p90"],
                "validation_nearest_distance_p90": validation[
                    "nearest_train_distance"
                ]["p90"],
                "validation_frozen_residual_mae": validation[
                    "frozen_residual_absolute_error"
                ]["mean"],
                "validation_error_distance_spearman": validation[
                    "error_distance_spearman"
                ],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["validation_frozen_residual_mae"],
            row["validation_nearest_distance_p90"],
        ),
        reverse=True,
    )


def _write_report(path: Path, result: dict[str, Any]) -> None:
    train = result["train"]
    validation = result["validation_monitor"]
    lines = [
        "# Surrogate Data Coverage Diagnostic",
        "",
        "- No true-PPL evaluation was run.",
        "- No model was trained or selected.",
        "- No final-test data was loaded.",
        "- Validation is a frozen monitoring panel only; it must not be reused as new training data.",
        "",
        "## Dataset integrity",
        "",
        f"- Train actions: {train['actions']}",
        f"- Train seed evaluations: {train['seed_evaluations']}",
        f"- Validation actions: {validation['actions']}",
        f"- Train/validation isolation: **{'PASS' if result['isolation_audit_passed'] else 'FAIL'}**",
        "",
        "## Training-source coverage",
        "",
        "| Source | Actions | Label SE p90 | Train NN distance p90 | Local target difference p90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source, metrics in train["by_source"].items():
        lines.append(
            f"| {source} | {metrics['actions']} | "
            f"{metrics['label_standard_error']['p90']:.6f} | "
            f"{metrics['nearest_train_distance']['p90']:.6f} | "
            f"{metrics['local_target_absolute_difference']['p90']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen validation monitor",
            "",
            "| Source | Actions | Residual MAE | Validation NN distance p90 | Error-distance Spearman |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for source, metrics in validation["by_source"].items():
        lines.append(
            f"| {source} | {metrics['actions']} | "
            f"{metrics['frozen_residual_absolute_error']['mean']:.6f} | "
            f"{metrics['nearest_train_distance']['p90']:.6f} | "
            f"{metrics['error_distance_spearman']:.6f} |"
        )
    lines.extend(["", "## Priority observations", ""])
    for index, row in enumerate(result["priority_observations"], start=1):
        lines.append(
            f"{index}. `{row['source']}`: validation MAE "
            f"{row['validation_frozen_residual_mae']:.6f}; train actions "
            f"{row['train_actions']}; train label-SE p90 "
            f"{row['train_label_se_p90']:.6f}; validation NN-distance p90 "
            f"{row['validation_nearest_distance_p90']:.6f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- High label SE means repeat seeds can improve the target mean.",
            "- High nearest-neighbor distance or high local target variation means new action diversity is more valuable than repeated seeds.",
            "- The validation-monitor error is descriptive evidence only. New training actions must be generated independently and must not copy validation channels, deployments, drop vectors, or noise seeds.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose_surrogate_coverage(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    report_path: Path,
    device_name: str = "cuda",
    neighbors: int = 8,
) -> dict[str, Any]:
    """Measure train coverage and frozen validation support without retraining."""

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    _verify_dataset_manifest(
        manifest, {"train": train_path, "validation": validation_path}
    )
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    train_features = build_feature_matrix(train["drop_probabilities"])
    validation_features = build_feature_matrix(validation["drop_probabilities"])
    normalized_train, normalized_validation = normalize_features(
        train_features, validation_features
    )
    train_neighbors, train_distances = nearest_neighbor_profile(
        normalized_train,
        normalized_train,
        neighbors=neighbors,
        exclude_matching_index=True,
    )
    validation_neighbors, validation_distances = nearest_neighbor_profile(
        normalized_validation,
        normalized_train,
        neighbors=neighbors,
    )
    train_target = train["log_ppl_ratio"].astype(np.float64)
    train_local_difference = np.abs(
        train_target - train_target[train_neighbors].mean(axis=1)
    )
    train_se = train["log_ppl_ratio_std"].astype(np.float64) / np.sqrt(
        train["noise_seed_count"].astype(np.float64)
    )
    validation_se = validation["log_ppl_ratio_std"].astype(np.float64) / np.sqrt(
        validation["noise_seed_count"].astype(np.float64)
    )
    prediction, uncertainty = _frozen_prediction(
        checkpoint_path, validation["drop_probabilities"], device_name
    )
    validation_error = np.abs(validation["log_ppl_ratio"] - prediction)
    train_by_source = _source_summary(
        sources=train["sample_source"],
        label_standard_error=train_se,
        nearest_distances=train_distances[:, 0],
        local_target_error=train_local_difference,
    )
    validation_by_source = _source_summary(
        sources=validation["sample_source"],
        label_standard_error=validation_se,
        nearest_distances=validation_distances[:, 0],
        observed_absolute_error=validation_error,
    )
    result = {
        "format_version": 1,
        "stage": "development_data_coverage_diagnostic",
        "not_a_model_selection": True,
        "not_a_final_test": True,
        "train_path": str(train_path),
        "train_sha256": _sha256(train_path),
        "validation_path": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "isolation_audit_passed": bool(manifest["isolation_audit"]["passed"]),
        "feature_space": {
            "description": "31 boundary probabilities plus total, maximum, square sum, nonzero fraction, and cumulative hazard",
            "dimensions": int(normalized_train.shape[1]),
            "neighbors": neighbors,
            "normalization": "train-only mean and std with a 0.02 minimum scale",
        },
        "train": {
            "actions": int(train_target.size),
            "seed_evaluations": int(train["noise_seed_count"].sum()),
            "label_standard_error": _quantiles(train_se),
            "nearest_train_distance": _quantiles(train_distances[:, 0]),
            "local_target_absolute_difference": _quantiles(train_local_difference),
            "by_source": train_by_source,
        },
        "validation_monitor": {
            "actions": int(validation_error.size),
            "label_standard_error": _quantiles(validation_se),
            "nearest_train_distance": _quantiles(validation_distances[:, 0]),
            "frozen_residual_absolute_error": _quantiles(validation_error),
            "ensemble_uncertainty": _quantiles(uncertainty),
            "error_distance_spearman": spearman_correlation(
                validation_distances[:, 0], validation_error
            ),
            "error_uncertainty_spearman": spearman_correlation(
                uncertainty, validation_error
            ),
            "by_source": validation_by_source,
        },
        "priority_observations": _priority_rows(train_by_source, validation_by_source),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result
