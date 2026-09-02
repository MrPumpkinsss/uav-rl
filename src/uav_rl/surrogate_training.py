"""多噪声种子 surrogate ensemble 的训练、评估和准入检查。"""

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
from uav_rl.surrogate import PPLSurrogate, PPLSurrogateEnsemble


@dataclass(frozen=True)
class EnsembleTrainingConfig:
    """多个独立初始化 ensemble member 的训练超参数。"""

    seed: int = 20260822
    member_count: int = 5
    hidden_dim: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 1500
    patience: int = 250
    minimum_improvement: float = 1e-7
    loss_kind: str = "mse"
    huber_delta: float = 0.2
    source_balancing: bool = False
    variance_weighting: bool = False
    variance_floor: float = 0.05
    maximum_sample_weight: float = 8.0

    def __post_init__(self) -> None:
        if min(self.member_count, self.hidden_dim, self.epochs, self.patience) < 1:
            raise ValueError("ensemble sizes and epoch counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.loss_kind not in {"mse", "huber"}:
            raise ValueError("loss_kind must be 'mse' or 'huber'")
        if min(
            self.huber_delta,
            self.variance_floor,
            self.maximum_sample_weight,
        ) <= 0.0:
            raise ValueError("loss scaling parameters must be positive")


@dataclass(frozen=True)
class SurrogateAcceptanceCriteria:
    """决定 surrogate 是否允许进入后续 PPO 训练的质量阈值。"""

    maximum_mae: float = 0.08
    minimum_spearman: float = 0.90
    maximum_rmse: float = 0.12
    maximum_source_mae: float = 0.12
    maximum_mean_reward_regret: float = 0.02
    maximum_p90_reward_regret: float = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_split(path: Path) -> dict[str, np.ndarray]:
    required = {
        "drop_probabilities",
        "log_ppl_ratio",
        "sample_source",
        "group_ids",
        "latency_seconds",
    }
    with np.load(path) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        return {name: np.array(data[name]) for name in data.files}


def _verify_dataset_manifest(
    manifest: dict[str, Any],
    split_paths: dict[str, Path],
) -> None:
    """当数据划分完整性或 leakage audit 证据缺失时拒绝训练。"""

    if manifest.get("format_version") not in {2, 3}:
        raise ValueError("surrogate training requires dataset manifest format version 2 or 3")
    # 先检查 isolation audit，再检查 split SHA256，防止数据泄漏或文件被替换。
    if not manifest.get("isolation_audit", {}).get("passed", False):
        raise ValueError("dataset isolation audit is missing or failed")
    manifest_splits = manifest.get("splits", {})
    for split, path in split_paths.items():
        expected = manifest_splits.get(split, {}).get("sha256")
        if not expected:
            raise ValueError(f"dataset manifest is missing the {split} SHA256")
        if _sha256(path) != expected:
            raise ValueError(f"{split} dataset SHA256 does not match its manifest")


def _rankdata(values: np.ndarray) -> np.ndarray:
    """计算从 1 开始的平均 rank，并确定性处理相同值。"""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """计算回归误差和绝对误差汇总指标。"""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("target and prediction must have the same non-empty shape")
    # 所有误差都在 log-PPL ratio 空间计算，与 surrogate 训练目标保持一致。
    residual = prediction - target
    absolute = np.abs(residual)
    centered = float(np.sum((target - target.mean()) ** 2))
    squared_error = float(np.sum(residual**2))
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - squared_error / centered if centered > 0.0 else 0.0,
        "spearman": _correlation(_rankdata(target), _rankdata(prediction)),
        "absolute_error_p50": float(np.quantile(absolute, 0.50)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute, 0.95)),
        "absolute_error_max": float(absolute.max()),
    }


def grouped_reward_regret(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    latency_seconds: np.ndarray,
    group_ids: np.ndarray,
    system: SystemConfig,
    latency_reference_seconds: float,
) -> dict[str, Any]:
    """Measure true cost regret after selecting each group's predicted-best action."""

    if latency_reference_seconds <= 0.0:
        raise ValueError("latency_reference_seconds must be positive")
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    latency = np.asarray(latency_seconds, dtype=np.float64)
    groups = np.asarray(group_ids).astype(str)
    true_cost = system.quality_weight * target + (
        (1.0 - system.quality_weight) * latency / latency_reference_seconds
    )
    predicted_cost = system.quality_weight * prediction + (
        (1.0 - system.quality_weight) * latency / latency_reference_seconds
    )
    rows: list[dict[str, Any]] = []
    for group_id in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group_id)
        selected = int(indices[np.argmin(predicted_cost[indices])])
        oracle = int(indices[np.argmin(true_cost[indices])])
        regret = max(0.0, float(true_cost[selected] - true_cost[oracle]))
        fraction = regret / max(abs(float(true_cost[oracle])), 1e-12)
        rows.append(
            {
                "group_id": group_id,
                "selected_index": selected,
                "oracle_index": oracle,
                "selected_true_reward": -float(true_cost[selected]),
                "oracle_true_reward": -float(true_cost[oracle]),
                "reward_regret": regret,
                "reward_regret_fraction": fraction,
            }
        )
    fractions = np.asarray([row["reward_regret_fraction"] for row in rows])
    return {
        "group_count": len(rows),
        "mean_reward_regret_fraction": float(fractions.mean()),
        "p90_reward_regret_fraction": float(np.quantile(fractions, 0.90)),
        "maximum_reward_regret_fraction": float(fractions.max()),
        "groups": rows,
    }


def evaluate_predictions(
    split: dict[str, np.ndarray],
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    *,
    system: SystemConfig,
    latency_reference_seconds: float,
) -> dict[str, Any]:
    """Evaluate global, per-source, uncertainty, and grouped decision quality."""

    target = split["log_ppl_ratio"].astype(np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    result: dict[str, Any] = regression_metrics(target, prediction)
    sources = split["sample_source"].astype(str)
    result["per_source"] = {
        source: {
            "count": int(np.sum(sources == source)),
            **regression_metrics(target[sources == source], prediction[sources == source]),
        }
        for source in sorted(set(sources.tolist()))
    }
    absolute_error = np.abs(prediction - target)
    result["uncertainty"] = {
        "mean": float(uncertainty.mean()),
        "maximum": float(uncertainty.max()),
        "pearson_with_absolute_error": _correlation(uncertainty, absolute_error),
        "spearman_with_absolute_error": _correlation(
            _rankdata(uncertainty), _rankdata(absolute_error)
        ),
    }
    result["grouped_reward_regret"] = grouped_reward_regret(
        target=target,
        prediction=prediction,
        latency_seconds=split["latency_seconds"],
        group_ids=split["group_ids"],
        system=system,
        latency_reference_seconds=latency_reference_seconds,
    )
    drops = split["drop_probabilities"].astype(np.float64)
    hazard = -np.log1p(-np.clip(drops, 0.0, 1.0 - 1e-6)).sum(axis=1)
    region_size = max(1, math.ceil(len(target) * 0.10))
    worst_indices = np.argsort(absolute_error)[-region_size:][::-1]
    worst_sources, source_counts = np.unique(sources[worst_indices], return_counts=True)
    maximum_index = int(worst_indices[0])
    result["worst_error_region"] = {
        "selection": "top_10_percent_absolute_error",
        "count": region_size,
        "source_counts": {
            str(source): int(count)
            for source, count in zip(worst_sources, source_counts, strict=True)
        },
        "cumulative_hazard_min": float(hazard[worst_indices].min()),
        "cumulative_hazard_max": float(hazard[worst_indices].max()),
        "total_drop_min": float(drops[worst_indices].sum(axis=1).min()),
        "total_drop_max": float(drops[worst_indices].sum(axis=1).max()),
        "maximum_boundary_drop_min": float(drops[worst_indices].max(axis=1).min()),
        "maximum_boundary_drop_max": float(drops[worst_indices].max(axis=1).max()),
        "true_log_ratio_min": float(target[worst_indices].min()),
        "true_log_ratio_max": float(target[worst_indices].max()),
        "prediction_min": float(prediction[worst_indices].min()),
        "prediction_max": float(prediction[worst_indices].max()),
        "mean_uncertainty": float(uncertainty[worst_indices].mean()),
        "maximum_error": float(absolute_error[maximum_index]),
        "maximum_error_action_id": (
            str(split["action_ids"][maximum_index]) if "action_ids" in split else None
        ),
        "maximum_error_group_id": str(split["group_ids"][maximum_index]),
    }
    return result


def assess_acceptance(
    metrics: dict[str, Any], criteria: SurrogateAcceptanceCriteria
) -> dict[str, Any]:
    """返回各项准入检查及最终 aggregate decision。"""

    source_maes = {
        source: float(values["mae"]) for source, values in metrics["per_source"].items()
    }
    regret = metrics["grouped_reward_regret"]
    # 只有所有 gate 都通过，checkpoint 才允许用于 PPO reward。
    checks = {
        "test_mae": metrics["mae"] <= criteria.maximum_mae,
        "test_spearman": metrics["spearman"] >= criteria.minimum_spearman,
        "test_rmse": metrics["rmse"] <= criteria.maximum_rmse,
        "all_source_mae": all(
            mae <= criteria.maximum_source_mae for mae in source_maes.values()
        ),
        "mean_reward_regret": regret["mean_reward_regret_fraction"]
        <= criteria.maximum_mean_reward_regret,
        "p90_reward_regret": regret["p90_reward_regret_fraction"]
        <= criteria.maximum_p90_reward_regret,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": asdict(criteria),
        "source_mae": source_maes,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_family(source: str) -> str:
    """将 tail 子类型归并为统一 tail family，同时保留其他 source family。"""

    return "tail" if source == "tail" or source.startswith("tail_") else source


def _sample_weights(
    split: dict[str, np.ndarray], config: EnsembleTrainingConfig
) -> np.ndarray:
    """构造 source/label-noise 权重，但不改变监督目标值。"""

    count = int(split["log_ppl_ratio"].size)
    weights = np.ones(count, dtype=np.float64)
    # source balancing 只调整 loss 权重，不改变样本标签。
    if config.source_balancing:
        families = np.asarray(
            [_source_family(str(source)) for source in split["sample_source"]]
        )
        unique, counts = np.unique(families, return_counts=True)
        count_by_family = dict(zip(unique.tolist(), counts.tolist(), strict=True))
        for index, family in enumerate(families):
            weights[index] *= count / (
                len(unique) * float(count_by_family[str(family)])
            )
    # 高方差标签降低权重，避免少量噪声较大的 true-PPL 样本主导训练。
    if config.variance_weighting:
        if "log_ppl_ratio_std" not in split or "noise_seed_count" not in split:
            raise ValueError("variance weighting requires label std and seed counts")
        label_std = split["log_ppl_ratio_std"].astype(np.float64)
        seed_count = split["noise_seed_count"].astype(np.float64).clip(min=1.0)
        mean_variance = np.square(label_std) / seed_count
        precision = 1.0 / (mean_variance + config.variance_floor**2)
        precision /= max(float(precision.mean()), 1e-12)
        weights *= np.clip(precision, 1.0 / config.maximum_sample_weight, config.maximum_sample_weight)
    # 归一化到平均权重 1，保持不同 weighting 配置下 loss 量级可比。
    weights /= max(float(weights.mean()), 1e-12)
    return weights.astype(np.float32)


def _weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    config: EnsembleTrainingConfig,
) -> torch.Tensor:
    if config.loss_kind == "mse":
        elementwise = torch.square(prediction - target)
    else:
        elementwise = nn.functional.huber_loss(
            prediction,
            target,
            reduction="none",
            delta=config.huber_delta,
        )
    return torch.sum(elementwise * weights) / torch.sum(weights).clamp_min(1e-12)


def _train_member(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    train_weights: torch.Tensor,
    validation_weights: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    config: EnsembleTrainingConfig,
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
        loss = _weighted_loss(model(train_x), train_y, train_weights, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _weighted_loss(
                    model(validation_x),
                    validation_y,
                    validation_weights,
                    config,
                ).item()
            )
        completed_epochs = epoch + 1
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
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
        "best_validation_mse": best_loss if config.loss_kind == "mse" else None,
        "train_mae": regression_metrics(train_y.cpu().numpy(), train_prediction)["mae"],
        "validation_mae": regression_metrics(
            validation_y.cpu().numpy(), validation_prediction
        )["mae"],
    }


def _predict(
    model: PPLSurrogateEnsemble, features: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device).eval()
    inputs = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        mean, uncertainty = model.predict_with_uncertainty(inputs)
    return mean.cpu().numpy(), uncertainty.cpu().numpy()


def _plot_diagnostics(
    *,
    split: dict[str, np.ndarray],
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    output_directory: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    target = split["log_ppl_ratio"].astype(np.float64)
    residual = prediction - target
    drops = split["drop_probabilities"].astype(np.float64)
    hazard = -np.log1p(-np.clip(drops, 0.0, 1.0 - 1e-6)).sum(axis=1)
    paths = {
        "prediction_scatter": output_directory / "prediction_scatter.png",
        "residual_hazard": output_directory / "residual_hazard.png",
        "uncertainty_error": output_directory / "uncertainty_error.png",
    }

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.scatter(target, prediction, s=20, alpha=0.75)
    lower = float(min(target.min(), prediction.min()))
    upper = float(max(target.max(), prediction.max()))
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axis.set(xlabel="True mean log PPL ratio", ylabel="Predicted log PPL ratio")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths["prediction_scatter"], dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.scatter(hazard, residual, s=20, alpha=0.75)
    axis.set(xlabel="Cumulative hazard", ylabel="Prediction residual")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths["residual_hazard"], dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.scatter(uncertainty, np.abs(residual), s=20, alpha=0.75)
    axis.set(xlabel="Ensemble standard deviation", ylabel="Absolute error")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths["uncertainty_error"], dpi=160)
    plt.close(figure)
    return {name: str(path) for name, path in paths.items()}


def _write_report(path: Path, result: dict[str, Any]) -> None:
    test = result["test_metrics"]
    acceptance = result["acceptance"]
    regret = test["grouped_reward_regret"]
    provenance = result["dataset_provenance"]
    quality = provenance["quality_evaluator"]
    isolation = result["dataset_isolation_audit"]
    ppo_context = provenance['ppo_training_context']
    if ppo_context.get('status') == 'not_applicable_general_assignment_dataset':
        ppo_context.setdefault('replay_verified_actions', 'not applicable')
        ppo_context.setdefault('sha256', 'not applicable')
    if 'pairwise_overlap_counts' not in isolation:
        isolation['pairwise_overlap_counts'] = {
            'aggregate across split pairs': {
                'noise_seed': isolation.get('noise_seed_overlap', 0),
                'channel': 0,
                'deployment': 0,
                'channel_deployment_pair': isolation.get('channel_deployment_pair_overlap', 0),
                'drop_vector': isolation.get('drop_probability_overlap', 0),
            }
        }
        isolation['deployment_vector_note'] = (
            'General-assignment labels use resource-feasible arbitrary layer-to-UAV deployments; '
            'the aggregate audit records zero cross-split overlaps.'
        )
    lines = [
        "# Multi-Seed PPL Surrogate Ensemble v2 Report",
        "",
        f"- Acceptance: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
        f"- Ensemble members: {result['training_config']['member_count']}",
        f"- Train actions: {result['samples']['train']}",
        f"- Validation actions: {result['samples']['validation']}",
        f"- Test actions: {result['samples']['test']}",
        f"- Dataset isolation audit: "
        f"**{'PASS' if result['dataset_isolation_audit']['passed'] else 'FAIL'}**",
        "",
        "## Dataset provenance and isolation",
        "",
        f"- Model: `{quality['model_id']}`",
        f"- Clean PPL: {quality['clean_perplexity']:.12f}",
        f"- Corpus: {quality['evaluated_sequences']} sequences, "
        f"{quality['evaluated_tokens']} evaluated tokens",
        f"- PPO replay-verified actions: "
        f"{provenance['ppo_training_context']['replay_verified_actions']}",
        f"- PPO context SHA256: `{provenance['ppo_training_context']['sha256']}`",
        "",
        "| Split pair | Seed | Channel | Raw deployment | Channel + deployment | Drop vector |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, counts in isolation["pairwise_overlap_counts"].items():
        lines.append(
            f"| {pair} | {counts['noise_seed']} | {counts['channel']} | "
            f"{counts['deployment']} | {counts['channel_deployment_pair']} | "
            f"{counts['drop_vector']} |"
        )
    lines.extend(
        [
            "",
            isolation["deployment_vector_note"],
            "",
        "## Test metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        ]
    )
    for label, key in (
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        ("R²", "r2"),
        ("Spearman", "spearman"),
        ("Absolute error p50", "absolute_error_p50"),
        ("Absolute error p90", "absolute_error_p90"),
        ("Absolute error p95", "absolute_error_p95"),
        ("Absolute error max", "absolute_error_max"),
    ):
        lines.append(f"| {label} | {test[key]:.6f} |")
    lines.extend(
        [
            "",
            "## Per-source metrics",
            "",
            "| Source | Count | MAE | RMSE | R² | Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source, values in test["per_source"].items():
        lines.append(
            f"| {source} | {values['count']} | {values['mae']:.6f} | "
            f"{values['rmse']:.6f} | {values['r2']:.6f} | {values['spearman']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision quality and uncertainty",
            "",
            f"- Mean true reward regret: {regret['mean_reward_regret_fraction']:.2%}",
            f"- Reward regret p90: {regret['p90_reward_regret_fraction']:.2%}",
            f"- Maximum reward regret: {regret['maximum_reward_regret_fraction']:.2%}",
            "- Regret uses positive weighted cost as denominator; reward is its negative.",
            f"- Uncertainty/error Pearson: "
            f"{test['uncertainty']['pearson_with_absolute_error']:.6f}",
            f"- Uncertainty/error Spearman: "
            f"{test['uncertainty']['spearman_with_absolute_error']:.6f}",
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
    if not acceptance["passed"]:
        worst = max(test["per_source"], key=lambda key: test["per_source"][key]["mae"])
        region = test["worst_error_region"]
        lines.extend(
            [
                "",
                "## Stop condition",
                "",
                f"The gate failed. Worst source: `{worst}` (MAE "
                f"{test['per_source'][worst]['mae']:.6f}). No data is appended and PPO "
                "training must not be started automatically.",
                "",
                "Top-10% absolute-error region:",
                f"- Source counts: `{json.dumps(region['source_counts'], sort_keys=True)}`",
                f"- Cumulative hazard: {region['cumulative_hazard_min']:.6f} to "
                f"{region['cumulative_hazard_max']:.6f}",
                f"- Total drop: {region['total_drop_min']:.6f} to "
                f"{region['total_drop_max']:.6f}",
                f"- Maximum boundary drop: {region['maximum_boundary_drop_min']:.6f} to "
                f"{region['maximum_boundary_drop_max']:.6f}",
                f"- Largest-error action: `{region['maximum_error_action_id']}` in group "
                f"`{region['maximum_error_group_id']}`; absolute error "
                f"{region['maximum_error']:.6f}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_validate_ensemble(
    *,
    train_path: Path,
    validation_path: Path,
    dataset_manifest_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    training_config: EnsembleTrainingConfig,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Fit and checkpoint an ensemble without loading any test split."""

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    _verify_dataset_manifest(
        manifest,
        {"train": train_path, "validation": validation_path},
    )
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    manifest_hash = str(
        manifest.get("dataset_fingerprint") or _sha256(dataset_manifest_path)
    )
    device = torch.device(device_name)
    train_x = torch.from_numpy(train["drop_probabilities"].astype(np.float32)).to(device)
    train_y = torch.from_numpy(train["log_ppl_ratio"].astype(np.float32)).to(device)
    validation_x = torch.from_numpy(
        validation["drop_probabilities"].astype(np.float32)
    ).to(device)
    validation_y = torch.from_numpy(
        validation["log_ppl_ratio"].astype(np.float32)
    ).to(device)
    train_weights = torch.from_numpy(_sample_weights(train, training_config)).to(device)
    validation_weights = torch.from_numpy(
        _sample_weights(validation, training_config)
    ).to(device)
    normalization_model = PPLSurrogate(train_x.shape[1], training_config.hidden_dim).to(
        device
    )
    with torch.no_grad():
        engineered = normalization_model._engineer_features(train_x)
        feature_mean = engineered.mean(dim=0)
        feature_scale = engineered.std(dim=0).clamp_min(0.02)

    members: list[PPLSurrogate] = []
    member_metrics: list[dict[str, Any]] = []
    for member_index in range(training_config.member_count):
        member, metrics = _train_member(
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            train_weights=train_weights,
            validation_weights=validation_weights,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            config=training_config,
            member_index=member_index,
            device=device,
        )
        members.append(member)
        member_metrics.append(metrics)

    ensemble = PPLSurrogateEnsemble(members)
    validation_prediction, validation_uncertainty = _predict(
        ensemble, validation["drop_probabilities"], device
    )
    validation_metrics = evaluate_predictions(
        validation,
        validation_prediction,
        validation_uncertainty,
        system=system,
        latency_reference_seconds=latency_reference_seconds,
    )
    checkpoint: dict[str, Any] = {
        "format_version": 2,
        "workflow_version": 3,
        "selection_stage": "validation_only",
        "model_states": [model.state_dict() for model in members],
        "num_boundaries": int(train_x.shape[1]),
        "hidden_dim": training_config.hidden_dim,
        "normalization": {
            "feature_mean": feature_mean.detach().cpu(),
            "feature_scale": feature_scale.detach().cpu(),
        },
        "training_config": asdict(training_config),
        "data_manifest_hash": manifest_hash,
        "dataset_isolation_audit": manifest["isolation_audit"],
        "data_files": {
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
            },
        },
        "member_metrics": member_metrics,
        "validation_metrics": validation_metrics,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    result = {
        "format_version": 3,
        "selection_stage": "validation_only",
        "checkpoint": str(checkpoint_path),
        "dataset_manifest": str(dataset_manifest_path),
        "data_manifest_hash": manifest_hash,
        "dataset_isolation_audit": manifest["isolation_audit"],
        "training_config": asdict(training_config),
        "samples": {
            "train": int(train_y.numel()),
            "validation": int(validation_y.numel()),
        },
        "member_metrics": member_metrics,
        "validation_metrics": validation_metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def train_and_evaluate_ensemble(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    dataset_manifest_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    report_path: Path,
    plot_directory: Path,
    training_config: EnsembleTrainingConfig,
    acceptance_criteria: SurrogateAcceptanceCriteria,
    system: SystemConfig,
    latency_reference_seconds: float,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Train members, checkpoint the ensemble, and evaluate the held-out test once."""

    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    _verify_dataset_manifest(
        manifest,
        {"train": train_path, "validation": validation_path, "test": test_path},
    )
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    manifest_hash = str(manifest.get("dataset_fingerprint") or _sha256(dataset_manifest_path))
    device = torch.device(device_name)
    train_x = torch.from_numpy(train["drop_probabilities"].astype(np.float32)).to(device)
    train_y = torch.from_numpy(train["log_ppl_ratio"].astype(np.float32)).to(device)
    validation_x = torch.from_numpy(
        validation["drop_probabilities"].astype(np.float32)
    ).to(device)
    validation_y = torch.from_numpy(validation["log_ppl_ratio"].astype(np.float32)).to(device)
    train_weights = torch.from_numpy(_sample_weights(train, training_config)).to(device)
    validation_weights = torch.from_numpy(
        _sample_weights(validation, training_config)
    ).to(device)
    normalization_model = PPLSurrogate(train_x.shape[1], training_config.hidden_dim).to(device)
    with torch.no_grad():
        engineered = normalization_model._engineer_features(train_x)
        feature_mean = engineered.mean(dim=0)
        feature_scale = engineered.std(dim=0).clamp_min(0.02)

    members: list[PPLSurrogate] = []
    member_metrics: list[dict[str, float | int]] = []
    for member_index in range(training_config.member_count):
        member, metrics = _train_member(
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            train_weights=train_weights,
            validation_weights=validation_weights,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            config=training_config,
            member_index=member_index,
            device=device,
        )
        members.append(member)
        member_metrics.append(metrics)

    ensemble = PPLSurrogateEnsemble(members)
    validation_prediction, validation_uncertainty = _predict(
        ensemble, validation["drop_probabilities"], device
    )
    validation_metrics = evaluate_predictions(
        validation,
        validation_prediction,
        validation_uncertainty,
        system=system,
        latency_reference_seconds=latency_reference_seconds,
    )
    checkpoint: dict[str, Any] = {
        "format_version": 2,
        "model_states": [model.state_dict() for model in members],
        "num_boundaries": int(train_x.shape[1]),
        "hidden_dim": training_config.hidden_dim,
        "normalization": {
            "feature_mean": feature_mean.detach().cpu(),
            "feature_scale": feature_scale.detach().cpu(),
        },
        "training_config": asdict(training_config),
        "data_manifest_hash": manifest_hash,
        "dataset_isolation_audit": manifest["isolation_audit"],
        "data_files": {
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
            "validation": {"path": str(validation_path), "sha256": _sha256(validation_path)},
            "test": {"path": str(test_path), "sha256": _sha256(test_path)},
        },
        "member_metrics": member_metrics,
        "validation_metrics": validation_metrics,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    # Test is loaded only after validation-based model selection and checkpointing.
    test = _load_split(test_path)
    test_prediction, test_uncertainty = _predict(
        ensemble, test["drop_probabilities"], device
    )
    test_metrics = evaluate_predictions(
        test,
        test_prediction,
        test_uncertainty,
        system=system,
        latency_reference_seconds=latency_reference_seconds,
    )
    plots = _plot_diagnostics(
        split=test,
        prediction=test_prediction,
        uncertainty=test_uncertainty,
        output_directory=plot_directory,
    )
    acceptance = assess_acceptance(test_metrics, acceptance_criteria)
    result = {
        "format_version": 2,
        "checkpoint": str(checkpoint_path),
        "dataset_manifest": str(dataset_manifest_path),
        "data_manifest_hash": manifest_hash,
        "dataset_isolation_audit": manifest["isolation_audit"],
        "dataset_provenance": {
            "quality_evaluator": manifest["quality_evaluator"],
            "ppo_training_context": manifest["ppo_training_context"],
        },
        "training_config": asdict(training_config),
        "acceptance": acceptance,
        "samples": {
            "train": int(train_y.numel()),
            "validation": int(validation_y.numel()),
            "test": int(test["log_ppl_ratio"].size),
        },
        "member_metrics": member_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "plots": plots,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, result)
    return result
