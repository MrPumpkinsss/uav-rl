"""Audit final multi-seed surrogate v2 data, checkpoint, metrics, and plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from uav_rl.surrogate_training import SurrogateAcceptanceCriteria, assess_acceptance


EXPECTED_SPLITS = {
    "train": {
        "actions": 2024,
        "noise_seeds_per_action": 4,
        "source_counts": {
            "ppo_cache": 1000,
            "coverage": 320,
            "random": 256,
            "tail": 128,
            "strong_link": 128,
            "dynamic_programming": 128,
            "compute_greedy": 64,
        },
    },
    "validation": {
        "actions": 128,
        "noise_seeds_per_action": 16,
        "source_counts": {
            "coverage": 32,
            "random": 32,
            "tail": 16,
            "strong_link": 16,
            "dynamic_programming": 16,
            "compute_greedy": 16,
        },
    },
    "test": {
        "actions": 128,
        "noise_seeds_per_action": 16,
        "source_counts": {
            "coverage": 32,
            "random": 32,
            "tail": 16,
            "strong_link": 16,
            "dynamic_programming": 16,
            "compute_greedy": 16,
        },
    },
}


def audit_outputs(
    manifest_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Return exhaustive structural checks plus the independently recomputed gate."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalization = checkpoint.get("normalization", {})
    context = manifest.get("ppo_training_context", {})
    checks: dict[str, bool] = {
        "dataset_format_v2": manifest.get("format_version") == 2,
        "dataset_isolation_passed": manifest.get("isolation_audit", {}).get("passed") is True,
        "ppo_context_replay_count": manifest.get("ppo_training_context", {}).get(
            "replay_verified_actions"
        )
        == 1000,
        "exact_corpus_metadata": (
            manifest.get("quality_evaluator", {}).get("model_id")
            == "codellama/CodeLlama-7b-hf"
            and abs(
                manifest.get("quality_evaluator", {}).get("clean_perplexity", 0.0)
                - 12.87591098442322
            )
            <= 1e-9
            and manifest.get("quality_evaluator", {}).get("evaluated_sequences") == 27
            and manifest.get("quality_evaluator", {}).get("evaluated_tokens") == 1689
            and manifest.get("quality_evaluator", {}).get(
                "sample_derived_clean_ppl_maximum_deviation", 1.0
            )
            <= 1e-5
        ),
        "checkpoint_format_v2": checkpoint.get("format_version") == 2,
        "five_checkpoint_members": len(checkpoint.get("model_states", [])) == 5,
        "five_member_metrics": len(checkpoint.get("member_metrics", [])) == 5,
        "checkpoint_training_config": (
            checkpoint.get("training_config", {}).get("member_count") == 5
            and checkpoint.get("training_config", {}).get("hidden_dim") == 512
        ),
        "checkpoint_has_normalization": set(checkpoint.get("normalization", {}))
        == {"feature_mean", "feature_scale"},
        "normalization_has_36_features": (
            tuple(normalization.get("feature_mean", torch.empty(0)).shape) == (36,)
            and tuple(normalization.get("feature_scale", torch.empty(0)).shape) == (36,)
        ),
        "manifest_hash_matches_checkpoint": checkpoint.get("data_manifest_hash")
        == manifest.get("dataset_fingerprint"),
        "manifest_hash_matches_metrics": metrics.get("data_manifest_hash")
        == manifest.get("dataset_fingerprint"),
        "report_exists": report_path.is_file() and report_path.stat().st_size > 0,
    }
    context_path = Path(str(context.get("path", "")))
    context_metadata_path = Path(str(context.get("replay_metadata_path", "")))
    checks["ppo_context_sha256"] = (
        context_path.is_file()
        and hashlib.sha256(context_path.read_bytes()).hexdigest() == context.get("sha256")
    )
    checks["ppo_context_metadata_sha256"] = (
        context_metadata_path.is_file()
        and hashlib.sha256(context_metadata_path.read_bytes()).hexdigest()
        == context.get("replay_metadata_sha256")
    )
    for split, expected in EXPECTED_SPLITS.items():
        observed = manifest.get("splits", {}).get(split, {})
        checks[f"{split}_actions"] = observed.get("actions") == expected["actions"]
        checks[f"{split}_seed_count"] = (
            observed.get("noise_seeds_per_action") == expected["noise_seeds_per_action"]
        )
        checks[f"{split}_sources"] = observed.get("source_counts") == expected["source_counts"]
        split_path = Path(str(observed.get("path", "")))
        checks[f"{split}_file_exists"] = split_path.is_file() and split_path.stat().st_size > 0
        checks[f"{split}_sha256"] = (
            split_path.is_file()
            and hashlib.sha256(split_path.read_bytes()).hexdigest() == observed.get("sha256")
        )
    required_test_metrics = {
        "mae",
        "rmse",
        "r2",
        "spearman",
        "absolute_error_p50",
        "absolute_error_p90",
        "absolute_error_p95",
        "absolute_error_max",
        "per_source",
        "uncertainty",
        "grouped_reward_regret",
        "worst_error_region",
    }
    test_metrics = metrics.get("test_metrics", {})
    checks["all_test_metrics_present"] = required_test_metrics.issubset(test_metrics)
    checks["all_test_sources_present"] = set(test_metrics.get("per_source", {})) == set(
        EXPECTED_SPLITS["test"]["source_counts"]
    )
    checks["sixteen_test_channel_groups"] = (
        test_metrics.get("grouped_reward_regret", {}).get("group_count") == 16
    )
    plot_paths = [Path(path) for path in metrics.get("plots", {}).values()]
    checks["three_nonempty_plots"] = len(plot_paths) == 3 and all(
        path.is_file() and path.stat().st_size > 0 for path in plot_paths
    )
    independent_acceptance = assess_acceptance(
        test_metrics,
        SurrogateAcceptanceCriteria(),
    )
    checks["acceptance_recomputed_consistently"] = (
        independent_acceptance == metrics.get("acceptance")
    )
    return {
        "structural_audit_passed": all(checks.values()),
        "surrogate_accepted": independent_acceptance["passed"],
        "checks": checks,
        "acceptance": independent_acceptance,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_multiseed_v2_manifest.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_multiseed_ensemble_v2.pth"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_multiseed_ensemble_v2_metrics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_multiseed_ensemble_v2_report.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/results/ppl_surrogate_multiseed_ensemble_v2_acceptance_audit.json"
        ),
    )
    args = parser.parse_args()
    result = audit_outputs(args.manifest, args.checkpoint, args.metrics, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["structural_audit_passed"]:
        raise SystemExit(1)
    if not result["surrogate_accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
