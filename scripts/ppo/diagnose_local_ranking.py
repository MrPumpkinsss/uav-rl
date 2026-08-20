"""Measure local surrogate ranking around strong-link with isolated true PPL labels.

This diagnostic is not PPO validation and its true labels must never be used to
train the current surrogate. It evaluates only strong-link and seven nearby
four-segment alternatives on fresh channels and a diagnostic-only noise stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import DataGenerationConfig, PPOConfig, SystemConfig
from uav_rl.experiment import estimate_latency_reference
from uav_rl.noise_seeds import diagnostic_noise_seeds
from uav_rl.ranking_diagnostic import strong_link_neighborhood, summarize_local_ranking
from uav_rl.rl.environment import DeploymentEnvironment, generate_channels
from uav_rl.surrogate import load_surrogate
from uav_rl.true_quality import TruePPLQualityEvaluator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs/diagnostics/strong_link_local_ranking_2026-08-18"),
    )
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual.pth"),
    )
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--channel-seed", type=int, default=24681357)
    parser.add_argument("--noise-samples", type=int, default=4)
    parser.add_argument("--noise-seed", type=int, default=97531864)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_paths(directory: Path) -> dict[str, Path]:
    return {
        "config": directory / "diagnostic_config.json",
        "cache": directory / "true_ppl_cache.jsonl",
        "detail": directory / "local_ranking_detail.npz",
        "result": directory / "local_ranking_report.json",
    }


def prepare_run_directory(
    args: argparse.Namespace, paths: dict[str, Path], config_payload: dict[str, Any]
) -> None:
    if args.resume:
        if not paths["config"].is_file():
            raise FileNotFoundError("--resume requires the diagnostic config")
        saved = json.loads(paths["config"].read_text(encoding="utf-8"))
        if saved != config_payload:
            raise ValueError("resume diagnostic configuration differs from the saved run")
        return
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("diagnostic directory already contains artifacts; choose another or pass --resume")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(paths["config"], config_payload)


def main() -> None:
    args = parse_args()
    if args.channels < 1 or args.noise_samples < 1:
        raise ValueError("channels and noise samples must be positive")
    if not args.surrogate.is_file():
        raise FileNotFoundError(f"surrogate checkpoint does not exist: {args.surrogate}")
    system = SystemConfig()
    generation = DataGenerationConfig()
    noise_seeds = diagnostic_noise_seeds(args.noise_samples, args.noise_seed)
    immutable = {
        "format_version": 1,
        "purpose": "strong_link_local_surrogate_ranking_diagnostic",
        "not_used_for_surrogate_training": True,
        "not_a_ppo_validation": True,
        "not_a_final_test": True,
        "system": asdict(system),
        "generation": asdict(generation),
        "surrogate_checkpoint": str(args.surrogate),
        "surrogate_checkpoint_sha256": _sha256(args.surrogate),
        "channels": args.channels,
        "channel_seed": args.channel_seed,
        "noise_samples": args.noise_samples,
        "noise_seed": args.noise_seed,
        "noise_seeds": noise_seeds.tolist(),
        "noise_seed_stream": "diagnostic range; disjoint from train/validation/test",
    }
    config_payload = {**immutable, "configuration_fingerprint": _fingerprint(immutable)}
    paths = run_paths(args.run_dir)
    prepare_run_directory(args, paths, config_payload)

    channels = generate_channels(args.channels, args.channel_seed, system)
    candidates = strong_link_neighborhood(channels, system)
    candidate_count = len(candidates.labels)
    flat_channels = np.repeat(channels, candidate_count, axis=0)
    flat_deployments = candidates.deployments.reshape(-1, system.num_layers)
    latency_reference = estimate_latency_reference(system, PPOConfig().seed)
    surrogate = load_surrogate(args.surrogate, device=torch.device(args.device))
    surrogate_environment = DeploymentEnvironment(system, surrogate, latency_reference)
    surrogate_rewards, surrogate_details = surrogate_environment.evaluate(
        flat_channels, flat_deployments
    )
    true_evaluator = TruePPLQualityEvaluator(
        generation, device_name=args.device, cache_path=paths["cache"]
    )
    true_environment = DeploymentEnvironment(system, true_evaluator, latency_reference)
    true_rewards, true_details = true_environment.evaluate(
        flat_channels, flat_deployments, noise_seeds=noise_seeds
    )
    surrogate_rewards = surrogate_rewards.reshape(args.channels, candidate_count)
    true_rewards = true_rewards.reshape(args.channels, candidate_count)
    metrics = summarize_local_ranking(candidates.labels, surrogate_rewards, true_rewards)
    np.savez_compressed(
        paths["detail"],
        candidate_labels=np.asarray(candidates.labels),
        channels=channels,
        deployments=candidates.deployments,
        surrogate_rewards=surrogate_rewards,
        true_rewards=true_rewards,
        surrogate_log_ppl_ratio=surrogate_details["log_ppl_ratio"].reshape(args.channels, candidate_count),
        true_log_ppl_ratio=true_details["log_ppl_ratio"].reshape(args.channels, candidate_count),
    )
    result = {
        **config_payload,
        "latency_reference_seconds": latency_reference,
        "candidate_count_per_channel": candidate_count,
        "true_ppl_forwards": true_evaluator.model_forwards,
        "true_ppl_cache_hits": true_evaluator.cache_hits,
        "true_quality_evaluator": true_evaluator.metadata(),
        "detail_path": str(paths["detail"]),
        "metrics": metrics,
    }
    _write_json(paths["result"], result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
