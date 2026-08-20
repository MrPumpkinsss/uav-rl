"""Select frozen surrogate-PPO candidates with independent true CodeLlama PPL.

This is a validation-only command. It never updates PPO and never appends its
labels to the surrogate training dataset. A final test remains separate.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.noise_seeds import validation_noise_seeds
from uav_rl.policy_validation import evaluate_policy_candidates, sha256
from uav_rl.rl import DeploymentEnvironment
from uav_rl.rl.environment import generate_channels
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage-name",
        default="true_validation",
        help="Artifact prefix for this independent validation stage.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        help="Optional frozen candidate checkpoint; repeat to evaluate a subset.",
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--channel-seed", type=int, default=20260901)
    parser.add_argument("--noise-samples", type=int, default=16)
    parser.add_argument("--noise-seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate_paths(directory: Path) -> list[Path]:
    candidates = sorted(directory.glob("episode_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no frozen candidate policies found in {directory}")
    return candidates


def _stage_paths(run_directory: Path, stage_name: str) -> tuple[Path, Path, Path]:
    """Return isolated evidence, selected-policy, and PPL-cache paths for one stage."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", stage_name):
        raise ValueError("stage name must contain lowercase letters, digits, and underscores")
    return (
        run_directory / f"{stage_name}.json",
        run_directory / f"{stage_name}_best_policy.pth",
        run_directory / f"{stage_name}_cache.jsonl",
    )


def _requested_candidates(
    candidate_directory: Path,
    requested: list[Path] | None,
) -> list[Path]:
    """Resolve an explicit candidate subset or all frozen candidates in order."""

    available = {path.resolve() for path in _candidate_paths(candidate_directory)}
    if requested is None:
        return sorted(available)
    candidates = [path.resolve() for path in requested]
    if not candidates:
        raise ValueError("at least one candidate checkpoint is required")
    outside = [path for path in candidates if path not in available]
    if outside:
        rendered = ", ".join(str(path) for path in outside)
        raise ValueError(f"requested candidates are not frozen run candidates: {rendered}")
    return candidates


def main() -> None:
    args = parse_args()
    if args.channels < 1 or args.noise_samples < 1:
        raise ValueError("channels and noise samples must be positive")
    launch_path = args.run_dir / "run_config.json"
    if not launch_path.is_file():
        raise FileNotFoundError(f"surrogate-PPO launch config is missing: {launch_path}")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    metadata = launch.get("run_metadata", {})
    if metadata.get("quality_backend") != "frozen_surrogate_training_only":
        raise ValueError("run directory was not produced by surrogate-only PPO training")
    candidates = _requested_candidates(args.run_dir / "candidate_policies", args.candidate)
    output_path, selected_path, cache_path = _stage_paths(args.run_dir, args.stage_name)
    if not args.allow_overwrite and (output_path.exists() or selected_path.exists()):
        raise FileExistsError(
            "true validation already exists; it is frozen evidence. Use --allow-overwrite only for an intentional rerun."
        )

    system = SystemConfig(**launch["ppo_config"]["system"])
    generation = DataGenerationConfig(**launch["true_validation_generation"])
    latency_reference = float(launch["latency_reference_seconds"])
    channels = generate_channels(args.channels, args.channel_seed, system)
    seeds = validation_noise_seeds(args.noise_samples, args.noise_seed)
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        cache_path=cache_path,
    )
    environment = DeploymentEnvironment(system, evaluator, latency_reference)
    result = evaluate_policy_candidates(
        environment=environment,
        candidate_paths=candidates,
        channels=channels,
        noise_seeds=seeds,
    )
    selected_candidate = Path(result["selected"]["path"])
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_candidate, selected_path)
    payload: dict[str, Any] = {
        "format_version": 1,
        "stage": "true_model_validation_only",
        "not_a_final_test": True,
        "not_used_for_surrogate_training": True,
        "validation_stage": args.stage_name,
        "run_directory": str(args.run_dir),
        "surrogate_checkpoint": metadata["surrogate_checkpoint"],
        "surrogate_checkpoint_sha256": metadata["surrogate_checkpoint_sha256"],
        "candidate_count": len(candidates),
        "candidate_paths": [str(path) for path in candidates],
        "candidate_sha256": {str(path): sha256(path) for path in candidates},
        "validation_channels": args.channels,
        "validation_channel_seed": args.channel_seed,
        "validation_noise_seeds": seeds.tolist(),
        "noise_seed_stream": "validation range; disjoint from surrogate training labels",
        "generation": asdict(generation),
        "latency_reference_seconds": latency_reference,
        "quality_evaluator": evaluator.metadata(),
        "cache_path": str(cache_path),
        "selected_policy": str(selected_path),
        "selected_policy_sha256": sha256(selected_path),
        "results": result,
    }
    _write_json(output_path, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
