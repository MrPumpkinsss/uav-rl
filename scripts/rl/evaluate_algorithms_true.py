"""Evaluate trained PPO/A2C/DQN checkpoints on one common true-LLM test set.

Run this only after the surrogate comparison has frozen all checkpoints.  The
script never retrains a policy and uses one shared true-PPL cache so identical
drop vectors are evaluated once.  The output is the evidence suitable for the
paper's RL-algorithm table; surrogate-only screening numbers are not.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import DataGenerationConfig
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.policy_io import load_layerwise_policy, resource_config_from_dict
from uav_rl.true_quality import TruePPLQualityEvaluator


class ZeroQualityEvaluator:
    """Placeholder used only while a checkpoint generates deployments."""

    def evaluate(
        self, drop_probabilities: np.ndarray, *, noise_seeds: np.ndarray | None = None
    ) -> np.ndarray:
        del noise_seeds
        return np.zeros(len(drop_probabilities), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--channel-seed", type=int, default=20260904)
    parser.add_argument("--noise-samples", type=int, default=4)
    parser.add_argument("--noise-start", type=int, default=1_800_000_000)
    parser.add_argument(
        "--model-id",
        default=DataGenerationConfig().model_id,
        help="Hugging Face model id or an existing local model directory.",
    )
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(args.channels, args.noise_samples) < 1:
        raise ValueError("channels and noise samples must be positive")
    summary_path = args.comparison_dir / "comparison_summary.json"
    output_path = args.comparison_dir / "true_model_comparison.json"
    csv_path = args.comparison_dir / "true_model_comparison.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"comparison summary is missing: {summary_path}")
    if output_path.exists() and not args.allow_overwrite:
        raise FileExistsError("true-model comparison is frozen; pass --allow-overwrite explicitly")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary["rows"]
    if not rows:
        raise ValueError("comparison summary contains no trained policies")

    first_payload = torch.load(Path(rows[0]["checkpoint"]), map_location="cpu", weights_only=False)
    resource_config = resource_config_from_dict(first_payload["resource_config"])
    first_evaluation = json.loads(
        (Path(rows[0]["checkpoint"]).parent / "evaluation.json").read_text(encoding="utf-8")
    )
    latency_reference = float(first_evaluation["latency_reference_seconds"])
    channels = generate_resource_channels(args.channels, args.channel_seed, resource_config)
    noise_seeds = np.arange(
        args.noise_start, args.noise_start + args.noise_samples, dtype=np.int64
    )

    generation = DataGenerationConfig(model_id=args.model_id)
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.model_device,
        cache_path=args.comparison_dir / "true_model_ppl_cache.jsonl",
    )
    true_environment = ResourceDeploymentEnvironment(
        resource_config, evaluator, latency_reference
    )
    policy_environment = ResourceDeploymentEnvironment(
        resource_config, ZeroQualityEvaluator(), latency_reference
    )

    true_rows: list[dict[str, Any]] = []
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        policy, _ = load_layerwise_policy(
            checkpoint,
            policy_environment,
            policy_device=args.policy_device,
        )
        deployments = policy.deployments(channels, deterministic=True)
        started = time.perf_counter()
        rewards, details = true_environment.evaluate(
            channels, deployments, noise_seeds=noise_seeds
        )
        elapsed = time.perf_counter() - started
        boundary_counts = np.count_nonzero(
            deployments[:, 1:] != deployments[:, :-1], axis=1
        )
        true_rows.append(
            {
                "algorithm": row["algorithm"],
                "seed": int(row["seed"]),
                "checkpoint": str(checkpoint),
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "reward_standard_error": float(
                    rewards.std(ddof=1) / np.sqrt(len(rewards))
                ),
                "ppl_mean": float(
                    (
                        evaluator.clean_perplexity
                        * np.exp(details["log_ppl_ratio"].astype(np.float64))
                    ).mean()
                ),
                "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
                "latency_mean_seconds": float(details["latency_seconds"].mean()),
                "invalid_fraction": float(details["invalid"].mean()),
                "boundary_count_mean": float(boundary_counts.mean()),
                "true_evaluation_wall_clock_seconds": elapsed,
            }
        )

    aggregate: dict[str, Any] = {}
    for algorithm in summary["algorithms"]:
        selected = [row for row in true_rows if row["algorithm"] == algorithm]
        aggregate[algorithm] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std_across_seeds": float(
                    np.std([row[metric] for row in selected], ddof=1)
                )
                if len(selected) > 1
                else 0.0,
            }
            for metric in (
                "reward_mean",
                "ppl_mean",
                "log_ppl_ratio_mean",
                "latency_mean_seconds",
                "invalid_fraction",
                "boundary_count_mean",
            )
        }
    payload = {
        "format_version": 1,
        "stage": "frozen_rl_algorithm_true_model_comparison",
        "source_surrogate_summary": str(summary_path),
        "channels": args.channels,
        "channel_seed": args.channel_seed,
        "noise_seeds": noise_seeds.tolist(),
        "quality_evaluator": evaluator.metadata(),
        "rows": true_rows,
        "aggregate": aggregate,
    }
    _write_json(output_path, payload)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(true_rows[0]))
        writer.writeheader()
        writer.writerows(true_rows)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
