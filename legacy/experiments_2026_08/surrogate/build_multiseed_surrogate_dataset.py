"""Collect and aggregate a resumable multi-seed CodeLlama surrogate dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.data.surrogate_dataset import (
    SurrogateDatasetConfig,
    collect_and_aggregate_surrogate_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--dataset-arrow", type=Path)
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--existing-ppo-cache",
        type=Path,
        default=Path("artifacts/cache/ppo_true_ppl_multiseed.jsonl"),
    )
    parser.add_argument(
        "--existing-ppo-context",
        type=Path,
        default=Path("artifacts/data/ppo_true_ppl_multiseed_training_context.npz"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_plan.json"),
    )
    parser.add_argument(
        "--sample-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_multiseed_v2.jsonl"),
    )
    parser.add_argument(
        "--ppl-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_multiseed_v2_ppl.jsonl"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/data"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.existing_ppo_cache.exists():
        raise FileNotFoundError(f"PPO cache does not exist: {args.existing_ppo_cache}")
    if not args.existing_ppo_context.exists():
        raise FileNotFoundError(f"PPO context does not exist: {args.existing_ppo_context}")
    generation = replace(
        DataGenerationConfig(),
        model_id=args.model,
        dataset_arrow_file=str(args.dataset_arrow) if args.dataset_arrow else None,
        text_sample_limit=args.sample_limit,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    result = collect_and_aggregate_surrogate_dataset(
        generation=generation,
        system=SystemConfig(),
        config=SurrogateDatasetConfig(),
        existing_ppo_cache=args.existing_ppo_cache,
        existing_ppo_context=args.existing_ppo_context,
        plan_path=args.plan,
        sample_cache_path=args.sample_cache,
        ppl_cache_path=args.ppl_cache,
        output_directory=args.output_directory,
        device_name=args.device,
        progress_interval=args.progress_interval,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
