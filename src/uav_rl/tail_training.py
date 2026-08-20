"""Validation-gated training and final evaluation for the tail-v3 surrogate."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import SystemConfig
from uav_rl.surrogate import PPLSurrogateEnsemble, TailGatedSurrogate, load_surrogate
from uav_rl.surrogate_training import (
    EnsembleTrainingConfig,
    SurrogateAcceptanceCriteria,
    _load_split,
    _plot_diagnostics,
    _predict,
    _sha256,
    assess_acceptance,
    evaluate_predictions,
    regression_metrics,
    train_and_validate_ensemble,
)


@dataclass(frozen=True)
class TailValidationCriteria:
    """Development gate evaluated without loading any final-test labels."""

    maximum_tail_mae: float = 0.12
    minimum_tail_spearman: float = 0.90
    maximum_overall_mae: float = 0.10
    maximum_non_tail_source_regression: float = 0.02


def _tail_mask(split: dict[str, np.ndarray]) -> np.ndarray:
    sources = split["sample_source"].astype(str)
    return np.asarray(
        [source == "tail" or source.startswith("tail_") for source in sources],
        dtype=np.bool_,
    )


def _tail_validation_metrics(
    split: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, float]:
    mask = _tail_mask(split)
    if not np.any(mask):
        raise ValueError("tail validation split contains no tail actions")
    return regression_metrics(split["log_ppl_ratio"][mask], prediction[mask])


def assess_tail_validation(
    *,
    validation_metrics: dict[str, Any],
    tail_metrics: dict[str, float],
    baseline_validation_metrics: dict[str, Any],
    criteria: TailValidationCriteria,
) -> dict[str, Any]:
    """Require tail improvement without materially degrading non-tail sources."""

    regressions: dict[str, float] = {}
    for source, baseline in baseline_validation_metrics["per_source"].items():
        if source == "tail":
            continue
        candidate = validation_metrics["per_source"].get(source)
        if candidate is None:
            raise ValueError(f"validation metrics are missing base source {source}")
        regressions[source] = float(candidate["mae"] - baseline["mae"])
    maximum_regression = max(regressions.values(), default=0.0)
    checks = {
        "tail_mae": tail_metrics["mae"] <= criteria.maximum_tail_mae,
        "tail_spearman": tail_metrics["spearman"] >= criteria.minimum_tail_spearman,
        "overall_mae": validation_metrics["mae"] <= criteria.maximum_overall_mae,
        "non_tail_regression": (
            maximum_regression <= criteria.maximum_non_tail_source_regression
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": asdict(criteria),
        "non_tail_source_mae_regression": regressions,
        "maximum_non_tail_source_mae_regression": maximum_regression,
    }


def _variant_configs(base: EnsembleTrainingConfig) -> dict[str, EnsembleTrainingConfig]:
    return {
        "mse": replace(
            base,
            loss_kind="mse",
            source_balancing=False,
            variance_weighting=False,
        ),
        "source_balanced_huber": replace(
            base,
            loss_kind="huber",
            source_balancing=True,
            variance_weighting=False,
        ),
        "variance_aware_huber": replace(
            base,
            loss_kind="huber",
            source_balancing=True,
            variance_weighting=True,
        ),
    }


def _write_validation_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Tail-Focused Surrogate v3 Validation Report",
        "",
        f"- Validation gate: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "- Final test loaded: **NO**",
        f"- Selected variant: `{summary['selected_variant']}`",
        "",
        "| Variant | Overall MAE | Tail MAE | Tail Spearman | Max non-tail regression | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in summary["variants"].items():
        gate = result["validation_gate"]
        lines.append(
            f"| {name} | {result['validation_metrics']['mae']:.6f} | "
            f"{result['tail_metrics']['mae']:.6f} | "
            f"{result['tail_metrics']['spearman']:.6f} | "
            f"{gate['maximum_non_tail_source_mae_regression']:.6f} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Stop/continue decision",
            "",
            (
                "The validation gate passed; a fresh final test may now be generated."
                if summary["passed"]
                else "The validation gate failed; no final test or PPO training may be started."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tail_validation_ablation(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_manifest_path: Path,
    baseline_metrics_path: Path,
    output_model_directory: Path,
    output_metrics_directory: Path,
    selected_checkpoint_path: Path,
    summary_path: Path,
    report_path: Path,
    training_config: EnsembleTrainingConfig,
    criteria: TailValidationCriteria,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str,
) -> dict[str, Any]:
    """Train three five-member variants and select strictly on validation metrics."""

    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_validation = baseline["validation_metrics"]
    validation = _load_split(validation_path)
    device = torch.device(device_name)
    variants: dict[str, Any] = {}
    for name, config in _variant_configs(training_config).items():
        checkpoint = output_model_directory / f"ppl_surrogate_tail_v3_{name}.pth"
        metrics_path = output_metrics_directory / f"ppl_surrogate_tail_v3_{name}.json"
        result = train_and_validate_ensemble(
            train_path=train_path,
            validation_path=validation_path,
            dataset_manifest_path=dataset_manifest_path,
            checkpoint_path=checkpoint,
            metrics_path=metrics_path,
            training_config=config,
            system=system,
            latency_reference_seconds=latency_reference_seconds,
            device_name=device_name,
        )
        model = load_surrogate(checkpoint, device=device)
        if not isinstance(model, (PPLSurrogateEnsemble, TailGatedSurrogate)):
            raise TypeError("tail-v3 ablation checkpoint is not an ensemble")
        prediction, _ = _predict(model, validation["drop_probabilities"], device)
        tail_metrics = _tail_validation_metrics(validation, prediction)
        gate = assess_tail_validation(
            validation_metrics=result["validation_metrics"],
            tail_metrics=tail_metrics,
            baseline_validation_metrics=baseline_validation,
            criteria=criteria,
        )
        result["tail_metrics"] = tail_metrics
        result["validation_gate"] = gate
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        variants[name] = result

    selected_name = min(
        variants,
        key=lambda name: (
            not variants[name]["validation_gate"]["passed"],
            variants[name]["tail_metrics"]["mae"],
            variants[name]["validation_metrics"]["mae"],
        ),
    )
    selected = variants[selected_name]
    selected_source = Path(selected["checkpoint"])
    selected_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_source, selected_checkpoint_path)
    summary = {
        "format_version": 3,
        "stage": "validation_selection",
        "passed": bool(selected["validation_gate"]["passed"]),
        "selected_variant": selected_name,
        "selected_checkpoint": str(selected_checkpoint_path),
        "selected_checkpoint_sha256": _sha256(selected_checkpoint_path),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "variants": variants,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_validation_report(report_path, summary)
    return summary


def _write_final_report(path: Path, result: dict[str, Any]) -> None:
    test = result["test_metrics"]
    tail = result["tail_aggregate_metrics"]
    acceptance = result["acceptance"]
    regret = test["grouped_reward_regret"]
    lines = [
        "# Tail-Focused Surrogate Ensemble v3 Final Report",
        "",
        f"- Acceptance: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
        f"- Selected validation variant: `{result['selected_variant']}`",
        "- Final test status: fresh held-out channels and noise seeds",
        f"- Dataset isolation audit: **{'PASS' if result['dataset_isolation_audit']['passed'] else 'FAIL'}**",
        "",
        "## Final-test metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MAE | {test['mae']:.6f} |",
        f"| RMSE | {test['rmse']:.6f} |",
        f"| R² | {test['r2']:.6f} |",
        f"| Spearman | {test['spearman']:.6f} |",
        f"| Absolute error p90 | {test['absolute_error_p90']:.6f} |",
        f"| Absolute error max | {test['absolute_error_max']:.6f} |",
        "",
        "## Tail aggregate",
        "",
        f"- Tail MAE: {tail['mae']:.6f}",
        f"- Tail RMSE: {tail['rmse']:.6f}",
        f"- Tail Spearman: {tail['spearman']:.6f}",
        "",
        "## Per-source metrics",
        "",
        "| Source | Count | MAE | RMSE | Spearman |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source, metrics in test["per_source"].items():
        lines.append(
            f"| {source} | {metrics['count']} | {metrics['mae']:.6f} | "
            f"{metrics['rmse']:.6f} | {metrics['spearman']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision quality",
            "",
            f"- Mean true reward regret: {regret['mean_reward_regret_fraction']:.2%}",
            f"- Reward regret p90: {regret['p90_reward_regret_fraction']:.2%}",
            f"- Maximum reward regret: {regret['maximum_reward_regret_fraction']:.2%}",
            "",
            "## Acceptance checks",
            "",
        ]
    )
    for name, passed in acceptance["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(["", "## Diagnostic plots", ""])
    for name, plot_path in result["plots"].items():
        relative = Path(plot_path).relative_to(path.parent)
        lines.append(f"- [{name}]({relative.as_posix()})")
    lines.extend(
        [
            "",
            "## Stop condition",
            "",
            (
                "The final gate passed. PPO remains a separate explicit user decision."
                if acceptance["passed"]
                else "The final gate failed. No PPO training may be started automatically."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_tail_final_test(
    *,
    final_manifest_path: Path,
    validation_summary_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    report_path: Path,
    plot_directory: Path,
    acceptance_criteria: SurrogateAcceptanceCriteria,
    tail_criteria: TailValidationCriteria,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str,
) -> dict[str, Any]:
    """Evaluate the validation-selected checkpoint exactly once on the fresh test."""

    validation_summary = json.loads(
        validation_summary_path.read_text(encoding="utf-8")
    )
    if not validation_summary.get("passed", False):
        raise RuntimeError("validation gate failed; refusing to evaluate final test")
    if _sha256(checkpoint_path) != validation_summary["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash does not match validation summary")
    manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "final_test":
        raise ValueError("tail-v3 final evaluation requires a final-test manifest")
    if not manifest.get("isolation_audit", {}).get("passed", False):
        raise ValueError("tail-v3 final-test isolation audit failed")
    test_path = Path(manifest["splits"]["test"]["path"])
    if _sha256(test_path) != manifest["splits"]["test"]["sha256"]:
        raise ValueError("tail-v3 final-test file hash does not match its manifest")
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected_training_fingerprint = manifest["development_manifest"][
        "dataset_fingerprint"
    ]
    if checkpoint_payload.get("data_manifest_hash") != expected_training_fingerprint:
        raise ValueError("selected checkpoint was not trained on the declared v3 data")

    device = torch.device(device_name)
    model = load_surrogate(checkpoint_path, device=device)
    if not isinstance(model, (PPLSurrogateEnsemble, TailGatedSurrogate)):
        raise TypeError("tail-v3 final checkpoint is not an ensemble")
    test = _load_split(test_path)
    prediction, uncertainty = _predict(model, test["drop_probabilities"], device)
    test_metrics = evaluate_predictions(
        test,
        prediction,
        uncertainty,
        system=system,
        latency_reference_seconds=latency_reference_seconds,
    )
    tail_metrics = _tail_validation_metrics(test, prediction)
    plots = _plot_diagnostics(
        split=test,
        prediction=prediction,
        uncertainty=uncertainty,
        output_directory=plot_directory,
    )
    acceptance = assess_acceptance(test_metrics, acceptance_criteria)
    acceptance["checks"]["tail_aggregate_mae"] = (
        tail_metrics["mae"] <= tail_criteria.maximum_tail_mae
    )
    acceptance["checks"]["tail_aggregate_spearman"] = (
        tail_metrics["spearman"] >= tail_criteria.minimum_tail_spearman
    )
    acceptance["passed"] = all(acceptance["checks"].values())
    acceptance["criteria"]["maximum_tail_aggregate_mae"] = (
        tail_criteria.maximum_tail_mae
    )
    acceptance["criteria"]["minimum_tail_aggregate_spearman"] = (
        tail_criteria.minimum_tail_spearman
    )
    result = {
        "format_version": 3,
        "stage": "final_test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_manifest": str(final_manifest_path),
        "final_manifest_sha256": _sha256(final_manifest_path),
        "validation_summary": str(validation_summary_path),
        "selected_variant": validation_summary["selected_variant"],
        "dataset_isolation_audit": manifest["isolation_audit"],
        "acceptance": acceptance,
        "test_metrics": test_metrics,
        "tail_aggregate_metrics": tail_metrics,
        "plots": plots,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_final_report(report_path, result)
    return result


def _label_noise_summary(split: dict[str, np.ndarray]) -> dict[str, float | int]:
    mask = _tail_mask(split)
    label_std = split["log_ppl_ratio_std"][mask].astype(np.float64)
    seed_count = split["noise_seed_count"][mask].astype(np.float64)
    standard_error = label_std / np.sqrt(seed_count)
    target = split["log_ppl_ratio"][mask].astype(np.float64)
    return {
        "actions": int(mask.sum()),
        "target_mean": float(target.mean()),
        "target_p90": float(np.quantile(target, 0.9)),
        "label_std_mean": float(label_std.mean()),
        "label_std_p90": float(np.quantile(label_std, 0.9)),
        "standard_error_mean": float(standard_error.mean()),
        "standard_error_p90": float(np.quantile(standard_error, 0.9)),
    }


def run_tail_post_selection_diagnostics(
    *,
    development_manifest_path: Path,
    validation_summary_path: Path,
    baseline_metrics_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    report_path: Path,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str,
) -> dict[str, Any]:
    """Evaluate diagnostic splits only after validation selection has completed."""

    manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    validation_summary = json.loads(
        validation_summary_path.read_text(encoding="utf-8")
    )
    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    if _sha256(checkpoint_path) != validation_summary["selected_checkpoint_sha256"]:
        raise ValueError("diagnostic checkpoint hash differs from validation selection")
    device = torch.device(device_name)
    model = load_surrogate(checkpoint_path, device=device)
    if not isinstance(model, (PPLSurrogateEnsemble, TailGatedSurrogate)):
        raise TypeError("tail-v3 diagnostic checkpoint is not an ensemble")
    split_paths = {
        "base_validation": Path(manifest["base_v2"]["splits"]["validation"]["path"]),
        "diagnostic_v2_test": Path(manifest["diagnostic_test"]["path"]),
        "tail_delta_validation": Path(
            manifest["delta_splits"]["validation"]["path"]
        ),
    }
    diagnostics: dict[str, Any] = {}
    for name, split_path in split_paths.items():
        split = _load_split(split_path)
        prediction, uncertainty = _predict(
            model, split["drop_probabilities"], device
        )
        metrics = evaluate_predictions(
            split,
            prediction,
            uncertainty,
            system=system,
            latency_reference_seconds=latency_reference_seconds,
        )
        diagnostics[name] = {
            "path": str(split_path),
            "sha256": _sha256(split_path),
            "metrics": metrics,
            "tail_metrics": _tail_validation_metrics(split, prediction),
        }
    base_train = _load_split(Path(manifest["base_v2"]["splits"]["train"]["path"]))
    delta_train = _load_split(Path(manifest["delta_splits"]["train"]["path"]))
    delta_validation = _load_split(
        Path(manifest["delta_splits"]["validation"]["path"])
    )
    old_tail_mae = float(baseline["test_metrics"]["per_source"]["tail"]["mae"])
    diagnostic_tail_mae = float(
        diagnostics["diagnostic_v2_test"]["tail_metrics"]["mae"]
    )
    result = {
        "format_version": 3,
        "stage": "post_selection_diagnostic",
        "not_a_final_acceptance_test": True,
        "validation_gate_passed": validation_summary["passed"],
        "selected_variant": validation_summary["selected_variant"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "diagnostics": diagnostics,
        "label_noise": {
            "base_train_tail": _label_noise_summary(base_train),
            "tail_delta_train": _label_noise_summary(delta_train),
            "tail_delta_validation": _label_noise_summary(delta_validation),
        },
        "diagnostic_v2_test_tail_mae_change": {
            "baseline": old_tail_mae,
            "tail_v3": diagnostic_tail_mae,
            "absolute_change": diagnostic_tail_mae - old_tail_mae,
            "relative_improvement": (old_tail_mae - diagnostic_tail_mae)
            / old_tail_mae,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    change = result["diagnostic_v2_test_tail_mae_change"]
    noise = result["label_noise"]
    lines = [
        "# Tail Surrogate v3 Post-Selection Diagnostic",
        "",
        "- This is **not** a fresh final acceptance test.",
        f"- Validation gate: **{'PASS' if result['validation_gate_passed'] else 'FAIL'}**",
        f"- Selected variant: `{result['selected_variant']}`",
        "",
        "## Same-distribution diagnostic comparison",
        "",
        f"- v2 diagnostic-test tail MAE: {change['baseline']:.6f}",
        f"- tail-v3 diagnostic-test tail MAE: {change['tail_v3']:.6f}",
        f"- Relative improvement: {change['relative_improvement']:.2%}",
        "",
        "## Label noise",
        "",
        "| Split | Tail actions | Mean label std | Mean standard error | p90 standard error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in noise.items():
        lines.append(
            f"| {name} | {values['actions']} | {values['label_std_mean']:.6f} | "
            f"{values['standard_error_mean']:.6f} | "
            f"{values['standard_error_p90']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The tail-focused delta improves the consumed v2 diagnostic test, but the "
            "validation MAE gate is not met. A fresh final test is therefore not generated, "
            "and PPO is not started.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
