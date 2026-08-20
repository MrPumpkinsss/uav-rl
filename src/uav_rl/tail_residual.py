"""Resumable hazard-gated residual training for the PPL surrogate."""

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
from uav_rl.surrogate import (
    PPLSurrogate,
    PPLSurrogateEnsemble,
    TailResidualSurrogate,
    load_surrogate,
)
from uav_rl.surrogate_training import (
    _load_split,
    _sha256,
    _verify_dataset_manifest,
    evaluate_predictions,
    regression_metrics,
)
from uav_rl.tail_expert import _gate_selection_key
from uav_rl.tail_training import TailValidationCriteria, _tail_mask, assess_tail_validation


@dataclass(frozen=True)
class TailResidualTrainingConfig:
    """Stable hyperparameters for residual ensemble member training."""

    seed: int = 20260818
    member_count: int = 5
    hidden_dim: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 1200
    patience: int = 200
    minimum_improvement: float = 1e-7

    def __post_init__(self) -> None:
        if min(self.member_count, self.hidden_dim, self.epochs, self.patience) < 1:
            raise ValueError("residual ensemble sizes and epoch counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("residual optimizer parameters are invalid")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _experiment_fingerprint(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_manifest_path: Path,
    base_checkpoint_path: Path,
    config: TailResidualTrainingConfig,
    include_epochs: bool = False,
) -> str:
    training_config = asdict(config)
    if not include_epochs:
        # An interrupted run may safely raise its maximum epoch budget. The
        # model architecture, optimizer, data and every other hyperparameter
        # remain immutable and therefore stay in the resume fingerprint.
        training_config.pop("epochs")
    payload = {
        "format_version": 1,
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "base_checkpoint_sha256": _sha256(base_checkpoint_path),
        "training_config": training_config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_resume_state(
    path: Path,
    fingerprint: str,
    legacy_epoch_sensitive_fingerprint: str,
    config: TailResidualTrainingConfig,
) -> dict[str, Any]:
    requested_epochs = config.epochs
    if not path.exists():
        return {
            "completed_model_states": [],
            "member_metrics": [],
            "maximum_epochs": requested_epochs,
        }
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("residual training state format is incompatible")
    state_fingerprint = state.get("fingerprint")
    if state_fingerprint == legacy_epoch_sensitive_fingerprint:
        # Migrate the first state produced before epoch extension was allowed.
        state["fingerprint"] = fingerprint
    elif state_fingerprint != fingerprint:
        completed = state.get("completed_model_states", [])
        complete_legacy_run = (
            state.get("current_member") is None
            and len(completed) == config.member_count
            and all(
                item.get("network.0.weight", torch.empty(0)).shape[0]
                == config.hidden_dim
                for item in completed
            )
        )
        if not complete_legacy_run:
            raise ValueError("residual training state is incompatible; refusing unsafe resume")
        # The initial completed state did not retain its epoch budget. Only a
        # finished five-member state with the current residual architecture can
        # be migrated; an interrupted legacy member remains deliberately
        # non-resumable without an exact old fingerprint.
        state["fingerprint"] = fingerprint
    previous_epochs = int(state.get("maximum_epochs", 0))
    if requested_epochs < previous_epochs:
        raise ValueError("residual resume may increase, but not lower, the epoch budget")
    state["maximum_epochs"] = requested_epochs
    return state


def _save_resume_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _new_residual_model(
    *,
    num_boundaries: int,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    config: TailResidualTrainingConfig,
    member_index: int,
    device: torch.device,
) -> tuple[PPLSurrogate, torch.optim.Optimizer]:
    _set_seed(config.seed + member_index)
    model = PPLSurrogate(num_boundaries, config.hidden_dim).to(device)
    with torch.no_grad():
        model.feature_mean.copy_(feature_mean)
        model.feature_scale.copy_(feature_scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return model, optimizer


def _train_member(
    *,
    member_index: int,
    train_x: torch.Tensor,
    train_residual: torch.Tensor,
    validation_x: torch.Tensor,
    validation_residual: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    config: TailResidualTrainingConfig,
    state_path: Path,
    state: dict[str, Any],
    fingerprint: str,
    device: torch.device,
) -> tuple[PPLSurrogate, dict[str, float | int]]:
    """Train one member, checkpointing every epoch with optimizer and RNG state."""

    current = state.get("current_member")
    model, optimizer = _new_residual_model(
        num_boundaries=train_x.shape[1],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        config=config,
        member_index=member_index,
        device=device,
    )
    if current is None:
        start_epoch = 0
        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
    elif current == member_index:
        model.load_state_dict(state["current_model_state"])
        optimizer.load_state_dict(state["current_optimizer_state"])
        _restore_rng_state(state["rng_state"])
        start_epoch = int(state["next_epoch"])
        best_loss = float(state["best_loss"])
        best_state = state.get("best_model_state")
        stale_epochs = int(state["stale_epochs"])
    else:
        raise ValueError("residual state points to a different active ensemble member")

    completed_epochs = start_epoch
    for epoch in range(start_epoch, config.epochs):
        model.train()
        loss = nn.functional.mse_loss(model(train_x), train_residual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.mse_loss(
                    model(validation_x), validation_residual
                ).item()
            )
        completed_epochs = epoch + 1
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        state.update(
            {
                "format_version": 1,
                "fingerprint": fingerprint,
                "current_member": member_index,
                "next_epoch": completed_epochs,
                "current_model_state": _cpu_state_dict(model),
                "current_optimizer_state": optimizer.state_dict(),
                "best_model_state": best_state,
                "best_loss": best_loss,
                "stale_epochs": stale_epochs,
                "rng_state": _rng_state(),
            }
        )
        _save_resume_state(state_path, state)
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("residual training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prediction = model(train_x).cpu().numpy()
        validation_prediction = model(validation_x).cpu().numpy()
    return model.cpu(), {
        "member_index": member_index,
        "seed": config.seed + member_index,
        "epochs_completed": completed_epochs,
        "best_validation_mse": best_loss,
        "train_residual_mae": regression_metrics(
            train_residual.cpu().numpy(), train_prediction
        )["mae"],
        "validation_residual_mae": regression_metrics(
            validation_residual.cpu().numpy(), validation_prediction
        )["mae"],
    }


def _selected_global_metrics(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_name = summary.get("selected_variant")
    variants = summary.get("variants", {})
    if selected_name not in variants:
        raise ValueError("global validation summary has no selected variant metrics")
    return variants[selected_name]["validation_metrics"]


def _predict(
    model: TailResidualSurrogate,
    drops: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device).eval()
    with torch.no_grad():
        mean, uncertainty = model.predict_with_uncertainty(
            torch.from_numpy(drops.astype(np.float32, copy=False)).to(device)
        )
    return mean.cpu().numpy(), uncertainty.cpu().numpy()


def _write_report(path: Path, result: dict[str, Any]) -> None:
    gate = result["selected_gate"]
    validation = result["validation_metrics"]
    tail = result["tail_metrics"]
    checks = result["validation_gate"]["checks"]
    lines = [
        "# Targeted Residual Surrogate Validation Report",
        "",
        f"- Validation gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        "- Final test loaded: **NO**",
        "- PPO started: **NO**",
        f"- Gate threshold / temperature: `{gate['hazard_threshold']}` / `{gate['hazard_temperature']}`",
        f"- Tail training actions: {result['samples']['tail_train']}",
        f"- Tail validation actions: {result['samples']['tail_validation']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Overall MAE | {validation['mae']:.6f} |",
        f"| Overall RMSE | {validation['rmse']:.6f} |",
        f"| Overall Spearman | {validation['spearman']:.6f} |",
        f"| Tail MAE | {tail['mae']:.6f} |",
        f"| Tail Spearman | {tail['spearman']:.6f} |",
        "",
        "## Per-source validation MAE",
        "",
        "| Source | Count | MAE |",
        "| --- | ---: | ---: |",
    ]
    for source, metrics in validation["per_source"].items():
        lines.append(f"| {source} | {metrics['count']} | {metrics['mae']:.6f} |")
    lines.extend(["", "## Validation checks", ""])
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "The global ensemble was frozen. Each residual member predicts "
            "the paired member's error, and the hazard gate scales only that correction.",
            "",
            (
                "Validation passed; a fresh final test remains an explicit next step."
                if result["passed"]
                else "Validation failed; no final test or PPO training was started."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_tail_residual_surrogate(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_manifest_path: Path,
    global_validation_summary_path: Path,
    base_checkpoint_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    report_path: Path,
    state_path: Path,
    training_config: TailResidualTrainingConfig,
    criteria: TailValidationCriteria,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str,
) -> dict[str, Any]:
    """Train/resume residual members and select a hazard gate on validation only."""

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    _verify_dataset_manifest(
        manifest, {"train": train_path, "validation": validation_path}
    )
    fingerprint = _experiment_fingerprint(
        train_path=train_path,
        validation_path=validation_path,
        dataset_manifest_path=dataset_manifest_path,
        base_checkpoint_path=base_checkpoint_path,
        config=training_config,
    )
    legacy_epoch_sensitive_fingerprint = _experiment_fingerprint(
        train_path=train_path,
        validation_path=validation_path,
        dataset_manifest_path=dataset_manifest_path,
        base_checkpoint_path=base_checkpoint_path,
        config=training_config,
        include_epochs=True,
    )
    state = _load_resume_state(
        state_path,
        fingerprint,
        legacy_epoch_sensitive_fingerprint,
        training_config,
    )
    _save_resume_state(state_path, state)
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    train_tail = _tail_mask(train)
    validation_tail = _tail_mask(validation)
    if not np.any(train_tail) or not np.any(validation_tail):
        raise ValueError("tail residual training requires tail actions in both splits")

    device = torch.device(device_name)
    base = load_surrogate(base_checkpoint_path, device=device)
    if not isinstance(base, PPLSurrogateEnsemble):
        raise TypeError("tail residual training requires a frozen global ensemble checkpoint")
    base.eval()
    if len(base.models) != training_config.member_count:
        raise ValueError("residual member count must match the frozen global ensemble")

    tail_train_x = torch.from_numpy(
        train["drop_probabilities"][train_tail].astype(np.float32)
    ).to(device)
    tail_validation_x = torch.from_numpy(
        validation["drop_probabilities"][validation_tail].astype(np.float32)
    ).to(device)
    tail_train_y = torch.from_numpy(
        train["log_ppl_ratio"][train_tail].astype(np.float32)
    ).to(device)
    tail_validation_y = torch.from_numpy(
        validation["log_ppl_ratio"][validation_tail].astype(np.float32)
    ).to(device)
    normalizer = PPLSurrogate(tail_train_x.shape[1], training_config.hidden_dim).to(device)
    with torch.no_grad():
        engineered = normalizer._engineer_features(tail_train_x)
        feature_mean = engineered.mean(dim=0)
        feature_scale = engineered.std(dim=0).clamp_min(0.02)

    completed_states = list(state.get("completed_model_states", []))
    member_metrics = list(state.get("member_metrics", []))
    if len(completed_states) != len(member_metrics):
        raise ValueError("residual state has mismatched completed member metadata")
    residuals: list[PPLSurrogate] = []
    for member_index, model_state in enumerate(completed_states):
        model = PPLSurrogate(tail_train_x.shape[1], training_config.hidden_dim)
        model.load_state_dict(model_state)
        residuals.append(model)

    for member_index in range(len(residuals), training_config.member_count):
        with torch.no_grad():
            base_train = base.models[member_index](tail_train_x)
            base_validation = base.models[member_index](tail_validation_x)
        residual, member_result = _train_member(
            member_index=member_index,
            train_x=tail_train_x,
            train_residual=tail_train_y - base_train,
            validation_x=tail_validation_x,
            validation_residual=tail_validation_y - base_validation,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            config=training_config,
            state_path=state_path,
            state=state,
            fingerprint=fingerprint,
            device=device,
        )
        residuals.append(residual)
        member_metrics.append(member_result)
        state.update(
            {
                "format_version": 1,
                "fingerprint": fingerprint,
                "completed_model_states": [_cpu_state_dict(item) for item in residuals],
                "member_metrics": member_metrics,
                "current_member": None,
            }
        )
        for key in (
            "next_epoch", "current_model_state", "current_optimizer_state",
            "best_model_state", "best_loss", "stale_epochs", "rng_state",
        ):
            state.pop(key, None)
        _save_resume_state(state_path, state)

    baseline_metrics = _selected_global_metrics(global_validation_summary_path)
    gate_variants: list[dict[str, Any]] = []
    for threshold in (0.20, 0.30, 0.40, 0.50, 0.60):
        for temperature in (0.03, 0.06, 0.10, 0.20):
            candidate = TailResidualSurrogate(
                base, residuals,
                hazard_threshold=threshold,
                hazard_temperature=temperature,
            )
            prediction, uncertainty = _predict(
                candidate, validation["drop_probabilities"], device
            )
            validation_metrics = evaluate_predictions(
                validation, prediction, uncertainty,
                system=system, latency_reference_seconds=latency_reference_seconds,
            )
            tail_metrics = regression_metrics(
                validation["log_ppl_ratio"][validation_tail], prediction[validation_tail]
            )
            gate = assess_tail_validation(
                validation_metrics=validation_metrics,
                tail_metrics=tail_metrics,
                baseline_validation_metrics=baseline_metrics,
                criteria=criteria,
            )
            gate_variants.append({
                "hazard_threshold": threshold,
                "hazard_temperature": temperature,
                "validation_metrics": validation_metrics,
                "tail_metrics": tail_metrics,
                "validation_gate": gate,
            })
    selected_gate = min(gate_variants, key=_gate_selection_key)
    payload = {
        "format_version": 1,
        "model_type": "tail_residual_ensemble",
        "num_boundaries": base.num_boundaries,
        "base_hidden_dim": base.models[0].network[0].out_features,
        "residual_hidden_dim": training_config.hidden_dim,
        "base_model_states": [_cpu_state_dict(model) for model in base.models],
        "residual_model_states": [_cpu_state_dict(model) for model in residuals],
        "gate": {
            "hazard_threshold": selected_gate["hazard_threshold"],
            "hazard_temperature": selected_gate["hazard_temperature"],
        },
        "training_config": asdict(training_config),
        "training_fingerprint": fingerprint,
        "data_manifest_hash": _sha256(dataset_manifest_path),
        "base_checkpoint": {"path": str(base_checkpoint_path), "sha256": _sha256(base_checkpoint_path)},
        "member_metrics": member_metrics,
        "validation_metrics": selected_gate["validation_metrics"],
        "tail_metrics": selected_gate["tail_metrics"],
        "validation_gate": selected_gate["validation_gate"],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    result = {
        "format_version": 1,
        "stage": "validation_only_residual_training",
        "model_type": "tail_residual_ensemble",
        "passed": bool(selected_gate["validation_gate"]["passed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "resume_state": str(state_path),
        "training_fingerprint": fingerprint,
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_sha256": _sha256(base_checkpoint_path),
        "global_validation_summary": str(global_validation_summary_path),
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
        "validation_metrics": selected_gate["validation_metrics"],
        "tail_metrics": selected_gate["tail_metrics"],
        "validation_gate": selected_gate["validation_gate"],
        "gate_variants": gate_variants,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result
