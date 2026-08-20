"""Compare the true-validated surrogate-PPO policy against common baselines.

The command reuses the already consumed true-validation channels and noise
seeds. It is a development comparison only, not a fresh final test, and it
never writes labels into the surrogate training dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.evaluation import STANDARD_METHODS, evaluate_methods
from uav_rl.policy_validation import load_policy_candidate, sha256
from uav_rl.rl import DeploymentEnvironment
from uav_rl.rl.environment import generate_channels
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-stage",
        default="true_validation",
        help="Completed true-validation stage whose inputs and cache are reused.",
    )
    parser.add_argument(
        "--baseline-methods",
        choices=STANDARD_METHODS[1:],
        nargs="+",
        default=list(STANDARD_METHODS[1:]),
        help="Baselines to compare with PPO; PPO itself is always included.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def validation_stage_paths(run_directory: Path, stage_name: str) -> tuple[Path, Path]:
    """Return the immutable validation evidence and comparison output for one stage."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", stage_name):
        raise ValueError("validation stage must contain lowercase letters, digits, and underscores")
    return (
        run_directory / f"{stage_name}.json",
        run_directory / f"{stage_name}_baseline_comparison.json",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validation_path, output_path = validation_stage_paths(args.run_dir, args.validation_stage)
    launch_path = args.run_dir / "run_config.json"
    if not validation_path.is_file() or not launch_path.is_file():
        raise FileNotFoundError("true validation and surrogate-PPO launch records are required")
    if output_path.exists() and not args.allow_overwrite:
        raise FileExistsError(
            "baseline comparison already exists; it is frozen validation evidence"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if validation.get("stage") != "true_model_validation_only":
        raise ValueError("expected a true-model validation record")
    recorded_stage = validation.get("validation_stage", "true_validation")
    if recorded_stage != args.validation_stage:
        raise ValueError("validation record stage differs from --validation-stage")
    policy_path = Path(validation["selected_policy"])
    if not policy_path.is_file():
        raise FileNotFoundError(f"selected true-validated policy is missing: {policy_path}")
    if sha256(policy_path) != validation["selected_policy_sha256"]:
        raise ValueError("selected policy hash differs from the validation record")

    system = SystemConfig(**launch["ppo_config"]["system"])
    generation = DataGenerationConfig(**launch["true_validation_generation"])
    channels = generate_channels(
        int(validation["validation_channels"]),
        int(validation["validation_channel_seed"]),
        system,
    )
    seeds = validation["validation_noise_seeds"]
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        cache_path=Path(
            validation.get(
                "cache_path",
                args.run_dir / f"{args.validation_stage}_cache.jsonl",
            )
        ),
    )
    environment = DeploymentEnvironment(
        system, evaluator, float(validation["latency_reference_seconds"])
    )
    policy = load_policy_candidate(policy_path, system)
    methods = evaluate_methods(
        environment=environment,
        policy=policy,
        channels=channels,
        clean_perplexity=evaluator.clean_perplexity,
        random_seed=int(launch["ppo_config"]["seed"]),
        method_names=("ppo", *args.baseline_methods),
        noise_seeds=seeds,
    )
    payload: dict[str, Any] = {
        "format_version": 1,
        "stage": "consumed_true_validation_baseline_comparison",
        "not_a_final_test": True,
        "not_used_for_surrogate_training": True,
        "run_directory": str(args.run_dir),
        "validation_stage": args.validation_stage,
        "baseline_methods": args.baseline_methods,
        "policy": str(policy_path),
        "policy_sha256": sha256(policy_path),
        "validation_record": str(validation_path),
        "validation_record_sha256": sha256(validation_path),
        "validation_channels": int(validation["validation_channels"]),
        "validation_channel_seed": int(validation["validation_channel_seed"]),
        "validation_noise_seeds": seeds,
        "generation": asdict(generation),
        "quality_evaluator": evaluator.metadata(),
        "methods": methods,
    }
    _write_json(output_path, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
