"""Measure the time required for one complete CodeLlama PPL evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.benchmarks import PerplexityBenchmarkConfig, run_perplexity_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="codellama/CodeLlama-7b-hf")
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/codellama_ppl.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PerplexityBenchmarkConfig(
        model_id=args.model,
        sample_limit=args.sample_limit,
        max_length=args.max_length,
        batch_size=args.batch_size,
        measured_runs=args.runs,
        device=args.device,
        dtype=args.dtype,
    )
    result = run_perplexity_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2))
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
