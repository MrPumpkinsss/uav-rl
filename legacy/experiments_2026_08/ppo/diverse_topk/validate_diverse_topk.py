"""Validate diverse Top-K PPO candidates with the true CodeLlama evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import DataGenerationConfig, PPOConfig, SystemConfig
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.diverse_topk import diverse_top_k_deployments
from uav_rl.rl.layerwise_ppo import LayerwisePPOTrainer
from uav_rl.surrogate import load_surrogate
from uav_rl.true_quality import TruePPLQualityEvaluator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--channel-seed", type=int, default=20260824)
    parser.add_argument("--noise-samples", type=int, default=4)
    parser.add_argument("--noise-start", type=int, default=4_100_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--min-hamming-fraction", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def _load_trainer(
    checkpoint_path: Path,
    surrogate_path: Path,
    latency_reference: float,
    device: torch.device,
) -> LayerwisePPOTrainer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ppo_payload = checkpoint["ppo_config"]
    system = SystemConfig(**ppo_payload["system"])
    ppo_config = PPOConfig(
        **{
            field.name: ppo_payload[field.name]
            for field in fields(PPOConfig)
            if field.name != "system"
        },
        system=system,
    )
    resource_payload = checkpoint["resource_config"]
    resource_config = ResourceConstrainedConfig(
        **{key: value for key, value in resource_payload.items() if key != "system"},
        system=SystemConfig(**resource_payload["system"]),
    )
    surrogate = load_surrogate(surrogate_path, device=device)
    environment = ResourceDeploymentEnvironment(resource_config, surrogate, latency_reference)
    trainer = LayerwisePPOTrainer(
        ppo_config,
        resource_config,
        environment,
        max_policy_boundaries=4,
    )
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.model.eval()
    return trainer


def _summary(
    rewards: np.ndarray,
    details: dict[str, np.ndarray],
    clean_perplexity: float,
) -> dict[str, float]:
    perplexity = clean_perplexity * np.exp(details["log_ppl_ratio"])
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "ppl_mean": float(perplexity.mean()),
        "ppl_std": float(perplexity.std()),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
        "invalid_fraction": float(details["invalid"].mean()),
    }


def main() -> None:
    args = parse_args()
    if min(args.channels, args.noise_samples, args.top_k, args.candidate_pool_size) < 1:
        raise ValueError("channels, noise samples, top-k, and pool size must be positive")
    if args.candidate_pool_size < args.top_k:
        raise ValueError("candidate pool size must be at least top-k")
    run_dir = args.run_dir
    checkpoint = args.checkpoint or run_dir / "best_policy.pth"
    output = args.output or run_dir / "diverse_topk_true_validation.json"
    cache = args.cache or Path("artifacts/runs/surrogate_ppo/common_seed_baseline_cache.jsonl")
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"output already exists: {output}")
    launch = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    latency_reference = float(launch["latency_reference_seconds"])
    device = torch.device(args.device)
    trainer = _load_trainer(checkpoint, args.surrogate, latency_reference, device)
    channels = generate_resource_channels(args.channels, args.channel_seed, trainer.resource_config)
    noise_seeds = np.arange(
        args.noise_start, args.noise_start + args.noise_samples, dtype=np.int64
    )
    deployments, surrogate_rewards, diagnostics = diverse_top_k_deployments(
        trainer,
        channels,
        k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        diversity_weight=args.diversity_weight,
        min_hamming_fraction=args.min_hamming_fraction,
    )
    evaluator = TruePPLQualityEvaluator(
        DataGenerationConfig(),
        device_name=args.device,
        cache_path=cache,
        progress_interval=32,
    )
    environment = ResourceDeploymentEnvironment(
        trainer.resource_config, evaluator, latency_reference
    )
    flat_channels = np.repeat(channels, args.top_k, axis=0)
    flat_deployments = deployments.reshape(args.channels * args.top_k, -1)
    true_rewards, details = environment.evaluate(
        flat_channels, flat_deployments, noise_seeds=noise_seeds
    )
    true_rewards = true_rewards.reshape(args.channels, args.top_k)
    true_top1 = true_rewards[:, 0]
    true_oracle = true_rewards.max(axis=1)
    deterministic = trainer.deployments(channels, deterministic=True)
    deterministic_rewards, deterministic_details = environment.evaluate(
        channels, deterministic, noise_seeds=noise_seeds
    )
    comparison_path = Path("artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.json")
    previous = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else {}
    previous_top1 = previous.get("methods", {}).get("ppo_top1_surrogate", {}).get("reward_mean")
    previous_oracle = previous.get("methods", {}).get("ppo_topk_true_oracle", {}).get("reward_mean")
    result: dict[str, Any] = {
        "format_version": 1,
        "stage": "diverse_topk_true_model_validation",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "surrogate": str(args.surrogate),
        "surrogate_sha256": _sha256(args.surrogate),
        "channels": args.channels,
        "channel_seed": args.channel_seed,
        "noise_seeds": noise_seeds.tolist(),
        "top_k": args.top_k,
        "candidate_pool_size": args.candidate_pool_size,
        "diversity_weight": args.diversity_weight,
        "min_hamming_fraction": args.min_hamming_fraction,
        "diagnostics": diagnostics,
        "surrogate_reward_mean_by_rank": surrogate_rewards.mean(axis=0).tolist(),
        "true_reward_mean_by_rank": true_rewards.mean(axis=0).tolist(),
        "diverse_top1": _summary(true_top1, {key: value.reshape(args.channels, args.top_k)[:, 0] for key, value in details.items()}, evaluator.clean_perplexity),
        "diverse_topk_true_oracle": _summary(
            true_oracle,
            {
                "log_ppl_ratio": details["log_ppl_ratio"].reshape(args.channels, args.top_k).mean(axis=1),
                "latency_seconds": details["latency_seconds"].reshape(args.channels, args.top_k).mean(axis=1),
                "invalid": details["invalid"].reshape(args.channels, args.top_k).any(axis=1),
            },
            evaluator.clean_perplexity,
        ),
        "deterministic": _summary(
            deterministic_rewards, deterministic_details, evaluator.clean_perplexity
        ),
        "previous_original_top1_reward_mean": previous_top1,
        "previous_original_topk_oracle_reward_mean": previous_oracle,
        "evaluator": evaluator.metadata(),
    }
    if previous_top1 is not None:
        result["diverse_top1_gain_vs_original_top1"] = result["diverse_top1"]["reward_mean"] - previous_top1
    if previous_oracle is not None:
        result["diverse_oracle_gain_vs_original_oracle"] = (
            result["diverse_topk_true_oracle"]["reward_mean"] - previous_oracle
        )
    _write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
