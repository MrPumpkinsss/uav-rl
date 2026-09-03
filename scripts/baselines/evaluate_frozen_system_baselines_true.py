"""使用真实因果语言模型评估已经冻结的 deployment。

本脚本只读取 surrogate screening 保存的 channel、deployment 和 PPO Top-K
候选，不会在观察真实 PPL 后重新搜索或修改任何方法。这样得到的结果才是
严格的 frozen-deployment true-LLM evaluation。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from uav_rl.config import DataGenerationConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment
from uav_rl.rl.policy_io import resource_config_from_dict
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    """解析命令行参数，构造本次实验的运行配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default=DataGenerationConfig().model_id,
        help="与 surrogate 数据链匹配的模型名称或本地模型目录。",
    )
    parser.add_argument("--noise-samples", type=int, default=4, help="每个 deployment 使用的独立噪声种子数量。")
    parser.add_argument("--noise-start", type=int, default=1_900_000_000, help="噪声种子的起始整数。")
    parser.add_argument("--device", default="cuda", help="真实 LLM 运行设备，例如 cuda 或 cpu。")
    parser.add_argument("--allow-overwrite", action="store_true", help="允许覆盖已有真实评估结果。")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    """原子保存评估结果，防止真实模型评估中断时破坏已有 JSON。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metrics_for_true(
    deployments: np.ndarray,
    rewards: np.ndarray,
    details: dict[str, np.ndarray],
    evaluator: TruePPLQualityEvaluator,
) -> dict[str, float]:
    # true evaluator 返回相对退化 log(PPL_noisy / PPL_clean)，这里还原绝对 PPL。
    """汇总真实 LLM 评估得到的 PPL、reward 和时延指标。"""
    boundaries = np.count_nonzero(deployments[:, 1:] != deployments[:, :-1], axis=1)
    ppl = evaluator.clean_perplexity * np.exp(details["log_ppl_ratio"].astype(np.float64))
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_standard_error": float(rewards.std(ddof=1) / np.sqrt(len(rewards))),
        "ppl_mean": float(ppl.mean()),
        "ppl_std": float(ppl.std()),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
        "invalid_fraction": float(details["invalid"].mean()),
        "boundary_count_mean": float(boundaries.mean()),
    }


def main() -> None:
    """读取冻结的 deployment，并在同一批 channel 上完成真实 LLM 评估。"""
    args = parse_args()
    if args.noise_samples < 1:
        raise ValueError("noise sample count must be positive")
    screening_path = args.comparison_dir / "comparison_summary.json"
    output_path = args.comparison_dir / "true_model_comparison.json"
    if not screening_path.is_file():
        raise FileNotFoundError(f"screening summary is missing: {screening_path}")
    if output_path.exists() and not args.allow_overwrite:
        raise FileExistsError("true comparison is frozen; pass --allow-overwrite explicitly")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    channels = np.load(args.comparison_dir / "channels.npy")
    deployments = np.load(args.comparison_dir / "frozen_deployments.npz")
    resource_config = resource_config_from_dict(screening["resource_config"])
    generation = DataGenerationConfig(model_id=args.model_id)
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        cache_path=args.comparison_dir / "true_model_ppl_cache.jsonl",
    )
    environment = ResourceDeploymentEnvironment(
        resource_config,
        evaluator,
        float(screening["latency_reference_seconds"]),
    )
    noise_seeds = np.arange(
        args.noise_start,
        args.noise_start + args.noise_samples,
        dtype=np.int64,
    )

    reward_vectors: dict[str, np.ndarray] = {}
    rows: dict[str, dict[str, float]] = {}
    # 保持 screening 阶段的方法顺序，确保 JSON、CSV 和报告可以逐行对应。
    method_order = list(screening["method_order"])
    for name in method_order:
        method_deployments = deployments[name]
        rewards, details = environment.evaluate(
            channels, method_deployments, noise_seeds=noise_seeds
        )
        reward_vectors[name] = rewards
        ppl = evaluator.clean_perplexity * np.exp(
            details["log_ppl_ratio"].astype(np.float64)
        )
        boundaries = np.count_nonzero(
            method_deployments[:, 1:] != method_deployments[:, :-1], axis=1
        )
        rows[name] = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_standard_error": float(rewards.std(ddof=1) / np.sqrt(len(rewards))),
            "ppl_mean": float(ppl.mean()),
            "ppl_std": float(ppl.std()),
            "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
            "latency_mean_seconds": float(details["latency_seconds"].mean()),
            "invalid_fraction": float(details["invalid"].mean()),
            "boundary_count_mean": float(boundaries.mean()),
        }
        print(
            f"true_method={name} reward={rows[name]['reward_mean']:.6f} "
            f"ppl={rows[name]['ppl_mean']:.4f}",
            flush=True,
        )

    # Top-K true oracle 只在冻结候选中取真实 reward 最大者，不是新的在线搜索方法。
    # 没有候选文件时兼容旧 benchmark，只评估普通冻结 deployment。
    topk_path = args.comparison_dir / "ppo_topk_candidates.npz"
    if topk_path.is_file():
        topk_candidates = np.load(topk_path)["deployments"]
        if topk_candidates.shape[0] != len(channels):
            raise ValueError("Top-K candidate channel count does not match channels.npy")
        candidate_count = topk_candidates.shape[1]
        candidate_channels = np.repeat(channels, candidate_count, axis=0)
        candidate_deployments = topk_candidates.reshape(-1, topk_candidates.shape[-1])
        # 重复 channel 后一次性评估 N*K 个候选，保证每个候选使用相同 noise seeds。
        candidate_rewards, candidate_details = environment.evaluate(
            candidate_channels, candidate_deployments, noise_seeds=noise_seeds
        )
        candidate_rewards = candidate_rewards.reshape(len(channels), candidate_count)
        # 每个 channel 独立选择真实 reward 最好的候选，不能跨 channel 共享索引。
        oracle_index = np.argmax(candidate_rewards, axis=1)
        oracle_rewards = candidate_rewards[np.arange(len(channels)), oracle_index]
        oracle_deployments = topk_candidates[np.arange(len(channels)), oracle_index]
        oracle_details = {
            key: value.reshape(len(channels), candidate_count)[
                np.arange(len(channels)), oracle_index
            ]
            for key, value in candidate_details.items()
        }
        rows["ppo_topk_true_oracle"] = _metrics_for_true(
            oracle_deployments, oracle_rewards, oracle_details, evaluator
        )
        reward_vectors["ppo_topk_true_oracle"] = oracle_rewards
        method_order.insert(2, "ppo_topk_true_oracle")

    # 所有方法仍相对同一 PPO deterministic 向量计算 paired difference。
    ppo = reward_vectors["ppo_deterministic"]
    for name in method_order:
        difference = reward_vectors[name] - ppo
        standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        rows[name]["paired_reward_difference_vs_ppo_mean"] = float(difference.mean())
        rows[name]["paired_reward_difference_ci95_low"] = float(
            difference.mean() - 1.96 * standard_error
        )
        rows[name]["paired_reward_difference_ci95_high"] = float(
            difference.mean() + 1.96 * standard_error
        )
        rows[name]["paired_win_rate_vs_ppo"] = float(np.mean(difference > 0.0))

    payload = {
        "format_version": 1,
        "stage": "frozen_true_model_system_baseline_comparison",
        "source_screening_summary": str(screening_path),
        "channels": int(len(channels)),
        "channel_seed": screening["channel_seed"],
        "noise_seeds": noise_seeds.tolist(),
        "model_id": args.model_id,
        "clean_perplexity": evaluator.clean_perplexity,
        "evaluated_sequences": evaluator.evaluated_sequences,
        "evaluated_tokens": evaluator.evaluated_tokens,
        "method_order": method_order,
        "methods": rows,
        "evaluator": evaluator.metadata(),
    }
    _write_json(output_path, payload)
    with (args.comparison_dir / "true_model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["method", *rows[screening["method_order"][0]].keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in method_order:
            writer.writerow({"method": name, **rows[name]})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
