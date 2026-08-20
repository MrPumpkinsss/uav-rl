"""Tail-only expert training with a deterministic hazard gate."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from uav_rl.config import SystemConfig
from uav_rl.surrogate import PPLSurrogate, PPLSurrogateEnsemble, TailGatedSurrogate, load_surrogate
from uav_rl.surrogate_training import (
    _load_split,
    evaluate_predictions,
    regression_metrics,
)
from uav_rl.tail_training import TailValidationCriteria, assess_tail_validation


@dataclass(frozen=True)
class TailExpertTrainingConfig:
    """Hyperparameters for the tail-only Huber experts."""

    seed: int = 3000
    member_count: int = 5
    hidden_dim: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 800
    patience: int = 120
    minimum_improvement: float = 1e-7
    huber_delta: float = 0.2
    hazard_threshold: float = 0.4
    hazard_temperature: float = 0.03

    def __post_init__(self) -> None:
        if min(self.member_count, self.hidden_dim, self.epochs, self.patience) < 1:
            raise ValueError("tail expert sizes and epoch counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("tail expert optimizer parameters are invalid")
        if self.huber_delta <= 0.0 or self.hazard_temperature <= 0.0:
            raise ValueError("tail expert scale parameters must be positive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tail_mask(split: dict[str, np.ndarray]) -> np.ndarray:
    sources = split["sample_source"].astype(str)
    return np.asarray(
        [source == "tail" or source.startswith("tail_") for source in sources],
        dtype=np.bool_,
    )


def _predict(model: TailGatedSurrogate, drops: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device).eval()
    with torch.no_grad():
        mean, uncertainty = model.predict_with_uncertainty(
            torch.from_numpy(drops.astype(np.float32, copy=False)).to(device)
        )
    return mean.cpu().numpy(), uncertainty.cpu().numpy()


def _train_member(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    config: TailExpertTrainingConfig,
    member_index: int,
    device: torch.device,
) -> tuple[PPLSurrogate, dict[str, float | int]]:
    member_seed = config.seed + member_index
    _set_seed(member_seed)
    model = PPLSurrogate(train_x.shape[1], config.hidden_dim).to(device)
    with torch.no_grad():
        model.feature_mean.copy_(feature_mean)
        model.feature_scale.copy_(feature_scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    completed_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        prediction = model(train_x)
        loss = nn.functional.huber_loss(
            prediction, train_y, delta=config.huber_delta
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.huber_loss(
                    model(validation_x), validation_y, delta=config.huber_delta
                ).item()
            )
        completed_epochs = epoch + 1
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("tail expert training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prediction = model(train_x).cpu().numpy()
        validation_prediction = model(validation_x).cpu().numpy()
    return model.cpu(), {
        "member_index": member_index,
        "seed": member_seed,
        "epochs_completed": completed_epochs,
        "best_validation_loss": best_loss,
        "train_mae": regression_metrics(train_y.cpu().numpy(), train_prediction)[
            "mae"
        ],
        "validation_mae": regression_metrics(
            validation_y.cpu().numpy(), validation_prediction
        )["mae"],
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    selected = result["selected_gate"]
    lines = [
        "# Tail Expert Gated Surrogate Report",
        "",
        f"- Validation gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- Selected hazard threshold: `{selected['hazard_threshold']}`",
        f"- Selected hazard temperature: `{selected['hazard_temperature']}`",
        f"- Tail training actions: {result['samples']['tail_train']}",
        f"- Tail validation actions: {result['samples']['tail_validation']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Overall MAE | {result['validation_metrics']['mae']:.6f} |",
        f"| Overall RMSE | {result['validation_metrics']['rmse']:.6f} |",
        f"| Overall Spearman | {result['validation_metrics']['spearman']:.6f} |",
        f"| Tail MAE | {result['tail_metrics']['mae']:.6f} |",
        f"| Tail Spearman | {result['tail_metrics']['spearman']:.6f} |",
        "",
        "## Gate variants",
        "",
        "| Threshold | Temperature | Overall MAE | Tail MAE | Tail Spearman | Gate |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant in result["gate_variants"]:
        lines.append(
            f"| {variant['hazard_threshold']:.2f} | "
            f"{variant['hazard_temperature']:.2f} | "
            f"{variant['validation_metrics']['mae']:.6f} | "
            f"{variant['tail_metrics']['mae']:.6f} | "
            f"{variant['tail_metrics']['spearman']:.6f} | "
            f"{'PASS' if variant['validation_gate']['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The base ensemble remains frozen; the expert is trained only on tail actions "
            "and blended using cumulative hazard. No final-test labels were loaded.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gate_selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    '''Prefer passed gates, then minimize failed checks and normalized violations.'''

    gate = item['validation_gate']
    checks = gate['checks']
    criteria = gate['criteria']
    tail = item['tail_metrics']
    overall = item['validation_metrics']
    violations = (
        max(0.0, tail['mae'] / criteria['maximum_tail_mae'] - 1.0),
        max(0.0, overall['mae'] / criteria['maximum_overall_mae'] - 1.0),
        max(
            0.0,
            gate['maximum_non_tail_source_mae_regression']
            / criteria['maximum_non_tail_source_regression']
            - 1.0,
        ),
        max(
            0.0,
            (criteria['minimum_tail_spearman'] - tail['spearman'])
            / criteria['minimum_tail_spearman'],
        ),
    )
    return (
        not gate['passed'],
        sum(not value for value in checks.values()),
        max(violations),
        sum(violations),
        overall['mae'],
        tail['mae'],
    )


def train_tail_gated_surrogate(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_manifest_path: Path,
    baseline_metrics_path: Path,
    base_checkpoint_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    report_path: Path,
    training_config: TailExpertTrainingConfig,
    criteria: TailValidationCriteria,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str,
) -> dict[str, Any]:
    """Train a tail-only expert and select its hazard gate on validation only."""

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    for split, path in (("train", train_path), ("validation", validation_path)):
        expected = manifest["splits"][split]["sha256"]
        if _sha256(path) != expected:
            raise ValueError(f"{split} split hash does not match its manifest")
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    train_tail = _tail_mask(train)
    validation_tail = _tail_mask(validation)
    if not np.any(train_tail) or not np.any(validation_tail):
        raise ValueError("tail expert requires tail actions in both splits")
    device = torch.device(device_name)
    base = load_surrogate(base_checkpoint_path, device=device)
    if not isinstance(base, PPLSurrogateEnsemble):
        raise TypeError("tail expert requires a frozen ensemble base checkpoint")
    base.eval()
    tail_x = torch.from_numpy(
        train["drop_probabilities"][train_tail].astype(np.float32)
    ).to(device)
    tail_y = torch.from_numpy(
        train["log_ppl_ratio"][train_tail].astype(np.float32)
    ).to(device)
    validation_x = torch.from_numpy(
        validation["drop_probabilities"][validation_tail].astype(np.float32)
    ).to(device)
    validation_y = torch.from_numpy(
        validation["log_ppl_ratio"][validation_tail].astype(np.float32)
    ).to(device)
    normalization_model = PPLSurrogate(tail_x.shape[1], training_config.hidden_dim).to(
        device
    )
    with torch.no_grad():
        engineered = normalization_model._engineer_features(tail_x)
        feature_mean = engineered.mean(dim=0)
        feature_scale = engineered.std(dim=0).clamp_min(0.02)
    experts: list[PPLSurrogate] = []
    member_metrics: list[dict[str, float | int]] = []
    for member_index in range(training_config.member_count):
        expert, member_result = _train_member(
            train_x=tail_x,
            train_y=tail_y,
            validation_x=validation_x,
            validation_y=validation_y,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            config=training_config,
            member_index=member_index,
            device=device,
        )
        experts.append(expert)
        member_metrics.append(member_result)

    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    gate_variants: list[dict[str, Any]] = []
    thresholds = (0.20, 0.30, 0.40, 0.50, 0.60)
    temperatures = (0.03, 0.06, 0.10, 0.20)
    for threshold in thresholds:
        for temperature in temperatures:
            model = TailGatedSurrogate(
                base,
                experts,
                hazard_threshold=threshold,
                hazard_temperature=temperature,
            )
            prediction, uncertainty = _predict(
                model, validation["drop_probabilities"], device
            )
            validation_metrics = evaluate_predictions(
                validation,
                prediction,
                uncertainty,
                system=system,
                latency_reference_seconds=latency_reference_seconds,
            )
            tail_metrics = regression_metrics(
                validation["log_ppl_ratio"][validation_tail],
                prediction[validation_tail],
            )
            gate = assess_tail_validation(
                validation_metrics=validation_metrics,
                tail_metrics=tail_metrics,
                baseline_validation_metrics=baseline["validation_metrics"],
                criteria=criteria,
            )
            gate_variants.append(
                {
                    "hazard_threshold": threshold,
                    "hazard_temperature": temperature,
                    "validation_metrics": validation_metrics,
                    "tail_metrics": tail_metrics,
                    "validation_gate": gate,
                }
            )
    selected_gate = min(
        gate_variants,
        key=lambda item: (
            not item["validation_gate"]["passed"],
            item["tail_metrics"]["mae"],
            item["validation_metrics"]["mae"],
        ),
    )
    selected_gate = min(gate_variants, key=_gate_selection_key)
    selected_model = TailGatedSurrogate(
        base,
        experts,
        hazard_threshold=selected_gate["hazard_threshold"],
        hazard_temperature=selected_gate["hazard_temperature"],
    ).to(device).eval()
    prediction, uncertainty = _predict(
        selected_model, validation["drop_probabilities"], device
    )
    validation_metrics = evaluate_predictions(
        validation,
        prediction,
        uncertainty,
        system=system,
        latency_reference_seconds=latency_reference_seconds,
    )
    tail_metrics = regression_metrics(
        validation["log_ppl_ratio"][validation_tail], prediction[validation_tail]
    )
    gate = selected_gate["validation_gate"]
    payload = {
        "format_version": 4,
        "workflow_version": 4,
        "model_type": "tail_gated_ensemble",
        "selection_stage": "validation_only",
        "base_model_states": [
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for model in base.models
        ],
        "expert_model_states": [
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for model in experts
        ],
        "num_boundaries": int(train["drop_probabilities"].shape[1]),
        "base_hidden_dim": base.models[0].network[0].out_features,
        "expert_hidden_dim": training_config.hidden_dim,
        "gate": {
            "hazard_threshold": selected_gate["hazard_threshold"],
            "hazard_temperature": selected_gate["hazard_temperature"],
        },
        "training_config": asdict(training_config),
        "data_manifest_hash": manifest.get("dataset_fingerprint")
        or _sha256(dataset_manifest_path),
        "base_checkpoint": {
            "path": str(base_checkpoint_path),
            "sha256": _sha256(base_checkpoint_path),
        },
        "data_files": {
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
            },
        },
        "member_metrics": member_metrics,
        "validation_metrics": validation_metrics,
        "tail_metrics": tail_metrics,
        "validation_gate": gate,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    result = {
        "format_version": 4,
        "model_type": "tail_gated_ensemble",
        "selection_stage": "validation_only",
        "passed": bool(gate["passed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_manifest": str(dataset_manifest_path),
        "data_manifest_hash": payload["data_manifest_hash"],
        "base_checkpoint": str(base_checkpoint_path),
        "training_config": asdict(training_config),
        "samples": {
            "tail_train": int(train_tail.sum()),
            "tail_validation": int(validation_tail.sum()),
            "train": int(train["log_ppl_ratio"].size),
            "validation": int(validation["log_ppl_ratio"].size),
        },
        "member_metrics": member_metrics,
        "selected_gate": {
            "hazard_threshold": selected_gate["hazard_threshold"],
            "hazard_temperature": selected_gate["hazard_temperature"],
        },
        "validation_metrics": validation_metrics,
        "tail_metrics": tail_metrics,
        "validation_gate": gate,
        "gate_variants": gate_variants,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result


def diagnose_tail_gated_surrogate(
    *,
    checkpoint_path: Path,
    seed16_manifest_path: Path,
    output_path: Path,
    report_path: Path,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str,
) -> dict[str, Any]:
    """Evaluate the gated candidate on consumed diagnostics only."""

    manifest = json.loads(seed16_manifest_path.read_text(encoding="utf-8"))
    model = load_surrogate(checkpoint_path, device=torch.device(device_name))
    if not isinstance(model, TailGatedSurrogate):
        raise TypeError("gated diagnostic requires a TailGatedSurrogate checkpoint")
    paths = {
        "base_validation": Path(
            manifest["base_v2"]["splits"]["validation"]["path"]
        ),
        "diagnostic_v2_test": Path(manifest["diagnostic_test"]["path"]),
        "tail_delta_validation": Path(
            manifest["delta_splits"]["validation"]["path"]
        ),
    }
    diagnostics: dict[str, Any] = {}
    device = torch.device(device_name)
    for name, path in paths.items():
        split = _load_split(path)
        prediction, uncertainty = _predict(model, split["drop_probabilities"], device)
        diagnostics[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "metrics": evaluate_predictions(
                split,
                prediction,
                uncertainty,
                system=system,
                latency_reference_seconds=latency_reference_seconds,
            ),
            "tail_metrics": regression_metrics(
                split["log_ppl_ratio"][_tail_mask(split)],
                prediction[_tail_mask(split)],
            ),
        }
    result = {
        "format_version": 4,
        "stage": "post_selection_diagnostic",
        "model_type": "tail_gated_ensemble",
        "not_a_final_acceptance_test": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "diagnostics": diagnostics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    old = diagnostics["diagnostic_v2_test"]["tail_metrics"]["mae"]
    lines = [
        "# Tail Gated Expert v4 Diagnostic Report",
        "",
        "- This is **not** a fresh final acceptance test.",
        "- The base v2 test is consumed diagnostic data only.",
        "",
        "| Split | Overall MAE | Tail MAE | Tail Spearman |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in diagnostics.items():
        lines.append(
            f"| {name} | {values['metrics']['mae']:.6f} | "
            f"{values['tail_metrics']['mae']:.6f} | "
            f"{values['tail_metrics']['spearman']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Seed16 gated expert diagnostic-test tail MAE: `{old:.6f}`",
            "- No final-test labels were loaded and PPO remains stopped.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
