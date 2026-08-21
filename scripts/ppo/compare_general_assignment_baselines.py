"""Strict common-seed comparison for the current arbitrary-assignment PPO.

The script evaluates deterministic PPO, PPO Top-K candidates, resource-aware
baselines, and surrogate-search upper bounds on exactly the same channels and
activation-noise seeds with the true CodeLlama evaluator.  Results are cached
JSONL by ``TruePPLQualityEvaluator`` and can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import DataGenerationConfig
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.resource_baselines import (
    dynamic_programming_baseline,
    fixed_eight_proxy_baseline,
    proxy_beam_baseline,
    random_feasible_baseline,
    surrogate_random_search,
)
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
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


def _load_policy_trainer(
    checkpoint_path: Path,
    surrogate_path: Path,
    device: torch.device,
    latency_reference: float,
    resource_config: ResourceConstrainedConfig,
) -> LayerwisePPOTrainer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ppo_config = checkpoint["ppo_config"]
    from dataclasses import fields
    from uav_rl.config import PPOConfig, SystemConfig

    system = SystemConfig(**ppo_config["system"])
    config = PPOConfig(**{field.name: ppo_config[field.name] for field in fields(PPOConfig) if field.name != "system"}, system=system)
    surrogate = load_surrogate(surrogate_path, device=device)
    environment = ResourceDeploymentEnvironment(resource_config, surrogate, latency_reference)
    trainer = LayerwisePPOTrainer(config, resource_config, environment, max_policy_boundaries=4)
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.model.eval()
    return trainer


def _summarize(
    rewards: np.ndarray,
    details: dict[str, np.ndarray],
    clean_perplexity: float,
) -> dict[str, float]:
    ppl = clean_perplexity * np.exp(details["log_ppl_ratio"])
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "ppl_mean": float(ppl.mean()),
        "ppl_std": float(ppl.std()),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
        "latency_std_seconds": float(details["latency_seconds"].std()),
        "invalid_fraction": float(details["invalid"].mean()),
    }


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
    parser.add_argument("--candidate-samples", type=int, default=20)
    parser.add_argument("--proxy-beam-width", type=int, default=128)
    parser.add_argument("--surrogate-search-candidates", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.channels, args.noise_samples, args.top_k, args.candidate_samples) < 1:
        raise ValueError("channels, noise samples, Top-K, and candidate samples must be positive")
    output = args.output or args.run_dir / "common_seed_baseline_comparison.json"
    cache = args.cache or args.run_dir / "common_seed_baseline_cache.jsonl"
    checkpoint = args.checkpoint or args.run_dir / "best_policy.pth"
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"output already exists: {output}")
    device = torch.device(args.device)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resource_config = ResourceConstrainedConfig(
        **{
            key: value
            for key, value in checkpoint_payload["resource_config"].items()
            if key != "system"
        },
        system=__import__("uav_rl.config", fromlist=["SystemConfig"]).SystemConfig(
            **checkpoint_payload["resource_config"]["system"]
        ),
    )
    latency_reference = float(
        json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))[
            "latency_reference_seconds"
        ]
    )
    trainer = _load_policy_trainer(checkpoint, args.surrogate, device, latency_reference, resource_config)
    surrogate_environment = trainer.environment
    channels = generate_resource_channels(args.channels, args.channel_seed, resource_config)
    noise_seeds = np.arange(args.noise_start, args.noise_start + args.noise_samples, dtype=np.int64)
    topk_deployments, topk_surrogate_rewards = trainer.top_k_deployments(
        channels, k=args.top_k, samples_per_channel=args.candidate_samples
    )
    deterministic = trainer.deployments(channels, deterministic=True)
    methods: dict[str, np.ndarray] = {
        "ppo_deterministic": deterministic,
        "ppo_top1_surrogate": topk_deployments[:, 0],
        "dynamic_programming": dynamic_programming_baseline(
            channels, resource_config, latency_reference
        ),
        "fixed_eight_strong_link": fixed_eight_proxy_baseline(
            channels, resource_config, latency_reference, score="strong_link"
        ),
        "fixed_eight_compute_greedy": fixed_eight_proxy_baseline(
            channels, resource_config, latency_reference, score="compute"
        ),
        "proxy_beam_128": proxy_beam_baseline(
            channels, resource_config, latency_reference, beam_width=args.proxy_beam_width
        ),
        "random_feasible": random_feasible_baseline(
            channels, resource_config, seed=20260826
        ),
        "surrogate_search_1024": surrogate_random_search(
            channels,
            surrogate_environment,
            seed=20260827,
            candidates_per_channel=args.surrogate_search_candidates,
        ),
    }
    # The fixed-eight surrogate oracle is an interpretable upper bound for the
    # old baseline family; it is evaluated with the same surrogate environment.
    fixed_eight_surrogate = fixed_eight_proxy_baseline(
        channels, resource_config, latency_reference, score="proxy"
    )
    methods["fixed_eight_surrogate_proxy"] = fixed_eight_surrogate

    evaluator = TruePPLQualityEvaluator(
        DataGenerationConfig(), device_name=args.device, cache_path=cache, progress_interval=32
    )
    true_environment = ResourceDeploymentEnvironment(resource_config, evaluator, latency_reference)
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for name, deployments in methods.items():
        print(f"evaluating_method={name}", flush=True)
        rewards, details = true_environment.evaluate(
            channels, deployments, noise_seeds=noise_seeds
        )
        results[name] = _summarize(rewards, details, evaluator.clean_perplexity)
    # True oracle among exactly the PPO Top-K candidates, not a training signal.
    flat_channels = np.repeat(channels, args.top_k, axis=0)
    flat_deployments = topk_deployments.reshape(args.channels * args.top_k, -1)
    candidate_rewards, candidate_details = true_environment.evaluate(
        flat_channels, flat_deployments, noise_seeds=noise_seeds
    )
    candidate_rewards = candidate_rewards.reshape(args.channels, args.top_k)
    oracle_rewards = candidate_rewards.max(axis=1)
    oracle_indices = candidate_rewards.argmax(axis=1)
    oracle_deployments = topk_deployments[np.arange(args.channels), oracle_indices]
    oracle_details_rewards, oracle_details = true_environment.evaluate(
        channels, oracle_deployments, noise_seeds=noise_seeds
    )
    results["ppo_topk_true_oracle"] = _summarize(
        oracle_details_rewards, oracle_details, evaluator.clean_perplexity
    )
    results["ppo_topk_true_oracle"]["candidate_reward_mean"] = float(oracle_rewards.mean())
    results["ppo_topk_true_oracle"]["candidate_selection_gap_vs_top1"] = float(
        oracle_rewards.mean() - results["ppo_top1_surrogate"]["reward_mean"]
    )
    baseline_reward = results["ppo_top1_surrogate"]["reward_mean"]
    for name, metrics in results.items():
        if name == "ppo_top1_surrogate":
            continue
        metrics["reward_improvement_vs_ppo_top1_percent"] = float(
            100.0 * (metrics["reward_mean"] - baseline_reward) / abs(baseline_reward)
        )
    payload: dict[str, Any] = {
        "format_version": 1,
        "stage": "common_seed_true_model_baseline_comparison",
        "not_used_for_surrogate_training": True,
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "surrogate": str(args.surrogate),
        "surrogate_sha256": _sha256(args.surrogate),
        "channels": args.channels,
        "channel_seed": args.channel_seed,
        "noise_seeds": noise_seeds.tolist(),
        "top_k": args.top_k,
        "candidate_samples": args.candidate_samples,
        "proxy_beam_width": args.proxy_beam_width,
        "surrogate_search_candidates": args.surrogate_search_candidates,
        "clean_perplexity": evaluator.clean_perplexity,
        "evaluated_sequences": evaluator.evaluated_sequences,
        "evaluated_tokens": evaluator.evaluated_tokens,
        "methods": results,
        "topk_surrogate_reward_mean_by_rank": topk_surrogate_rewards.mean(axis=0).tolist(),
        "evaluator": evaluator.metadata(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
