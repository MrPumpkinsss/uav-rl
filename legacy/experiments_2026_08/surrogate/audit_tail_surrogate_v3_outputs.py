"""Audit tail-v3 development data, ensembles, gate, and stop conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_DELTA_ACTIONS = {"train": 512, "validation": 64}
EXPECTED_DELTA_RECORDS = {"train": 4096, "validation": 1024}
EXPECTED_COMBINED_ACTIONS = {"train": 2536, "validation": 192}
EXPECTED_MEMBERS = 5
EXPECTED_VARIANTS = {
    "mse": "ppl_surrogate_tail_v3_mse.pth",
    "source_balanced_huber": "ppl_surrogate_tail_v3_source_balanced_huber.pth",
    "variance_aware_huber": "ppl_surrogate_tail_v3_variance_aware_huber.pth",
}
FINAL_ARTIFACTS = (
    Path("artifacts/data/codellama_surrogate_tail_v3_final_test_plan.json"),
    Path("artifacts/data/codellama_surrogate_tail_v3_test.npz"),
    Path("artifacts/data/codellama_surrogate_tail_v3_final_manifest.json"),
    Path("artifacts/cache/surrogate_tail_v3_final_test.jsonl"),
    Path("artifacts/cache/surrogate_tail_v3_final_test_ppl.jsonl"),
    Path("artifacts/results/ppl_surrogate_tail_ensemble_v3_metrics.json"),
    Path("artifacts/results/ppl_surrogate_tail_ensemble_v3_report.md"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}"
                    ) from error
    return records


def _npz_action_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return int(data["drop_probabilities"].shape[0])


def audit_outputs(
    *,
    manifest_path: Path,
    cache_path: Path,
    validation_summary_path: Path,
    selected_checkpoint_path: Path,
    ablation_model_directory: Path,
) -> dict[str, Any]:
    """Return reproducible structural checks for a failed development gate."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    records = _load_jsonl(cache_path)
    manifest_sha256 = _sha256(manifest_path)
    validation_sha256 = _sha256(validation_summary_path)
    cache_sha256 = _sha256(cache_path)
    checks: dict[str, bool] = {
        "development_manifest_format_v3": manifest.get("format_version") == 3,
        "development_stage": manifest.get("stage") == "development",
        "dataset_isolation_passed": manifest.get("isolation_audit", {}).get(
            "passed"
        )
        is True,
        "cache_has_5120_records": len(records) == 5120,
        "validation_gate_failed": validation.get("passed") is False,
        "selected_variant_is_declared": validation.get("selected_variant")
        in EXPECTED_VARIANTS,
        "validation_manifest_sha256_matches": validation.get(
            "dataset_manifest_sha256"
        )
        == manifest_sha256,
    }

    record_keys = [(record["action_id"], int(record["noise_seed"])) for record in records]
    checks["cache_has_5120_unique_action_seed_keys"] = (
        len(record_keys) == len(set(record_keys)) == 5120
    )
    records_by_split = Counter(str(record["split"]) for record in records)
    actions_by_split: dict[str, set[str]] = defaultdict(set)
    seeds_by_action: dict[tuple[str, str], set[int]] = defaultdict(set)
    for record in records:
        split = str(record["split"])
        action_id = str(record["action_id"])
        actions_by_split[split].add(action_id)
        seeds_by_action[(split, action_id)].add(int(record["noise_seed"]))
    for split in ("train", "validation"):
        checks[f"{split}_cache_records"] = (
            records_by_split[split] == EXPECTED_DELTA_RECORDS[split]
        )
        checks[f"{split}_delta_actions"] = (
            len(actions_by_split[split]) == EXPECTED_DELTA_ACTIONS[split]
        )
        expected_seed_count = 8 if split == "train" else 16
        checks[f"{split}_seeds_per_action"] = all(
            len(seeds_by_action[(split, action_id)]) == expected_seed_count
            for action_id in actions_by_split[split]
        )

    for group_name, expected_counts in (
        ("delta_splits", EXPECTED_DELTA_ACTIONS),
        ("splits", EXPECTED_COMBINED_ACTIONS),
    ):
        for split, expected_count in expected_counts.items():
            metadata = manifest.get(group_name, {}).get(split, {})
            path = Path(str(metadata.get("path", "")))
            prefix = f"{group_name}_{split}"
            checks[f"{prefix}_exists"] = path.is_file() and path.stat().st_size > 0
            checks[f"{prefix}_sha256"] = (
                path.is_file() and _sha256(path) == metadata.get("sha256")
            )
            checks[f"{prefix}_manifest_count"] = (
                metadata.get("actions") == expected_count
            )
            checks[f"{prefix}_npz_count"] = (
                path.is_file() and _npz_action_count(path) == expected_count
            )

    dataset_fingerprint = manifest.get("dataset_fingerprint")
    variant_hashes: dict[str, str] = {}
    for variant, filename in EXPECTED_VARIANTS.items():
        path = ablation_model_directory / filename
        exists = path.is_file() and path.stat().st_size > 0
        checks[f"{variant}_checkpoint_exists"] = exists
        if not exists:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checks[f"{variant}_has_five_members"] = (
            len(payload.get("model_states", [])) == EXPECTED_MEMBERS
            and len(payload.get("member_metrics", [])) == EXPECTED_MEMBERS
        )
        checks[f"{variant}_manifest_fingerprint"] = (
            payload.get("data_manifest_hash") == dataset_fingerprint
        )
        variant_hashes[variant] = _sha256(path)

    selected_exists = (
        selected_checkpoint_path.is_file()
        and selected_checkpoint_path.stat().st_size > 0
    )
    checks["selected_checkpoint_exists"] = selected_exists
    selected_hash = _sha256(selected_checkpoint_path) if selected_exists else ""
    checks["selected_checkpoint_hash_matches_summary"] = (
        selected_hash == validation.get("selected_checkpoint_sha256")
    )
    selected_variant = str(validation.get("selected_variant", ""))
    selected_gate = (
        validation.get("variants", {})
        .get(selected_variant, {})
        .get("validation_gate", {})
    )
    checks["selected_gate_matches_summary"] = selected_gate.get(
        "passed"
    ) == validation.get("passed")
    checks["selected_checkpoint_matches_variant"] = (
        selected_variant in variant_hashes
        and selected_hash == variant_hashes[selected_variant]
    )
    if selected_exists:
        selected_payload = torch.load(
            selected_checkpoint_path, map_location="cpu", weights_only=False
        )
        checks["selected_checkpoint_has_five_members"] = (
            len(selected_payload.get("model_states", [])) == EXPECTED_MEMBERS
            and len(selected_payload.get("member_metrics", [])) == EXPECTED_MEMBERS
        )
        checks["selected_checkpoint_manifest_fingerprint"] = (
            selected_payload.get("data_manifest_hash") == dataset_fingerprint
        )

    checks["no_final_test_artifacts"] = not any(
        path.exists() for path in FINAL_ARTIFACTS
    )
    return {
        "format_version": 3,
        "stage": "development_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "cache_records": len(records),
        "cache_unique_keys": len(set(record_keys)),
        "actions_by_split": {
            split: len(action_ids) for split, action_ids in actions_by_split.items()
        },
        "input_sha256": {
            "development_manifest": manifest_sha256,
            "development_cache": cache_sha256,
            "validation_summary": validation_sha256,
        },
        "selected_checkpoint_sha256": selected_hash,
        "variant_checkpoint_sha256": variant_hashes,
        "final_artifacts_checked": [str(path) for path in FINAL_ARTIFACTS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_manifest.json"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_tail_v3_development.jsonl"),
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_validation.json"),
    )
    parser.add_argument(
        "--selected-checkpoint",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_tail_ensemble_v3_selected.pth"),
    )
    parser.add_argument(
        "--ablation-model-directory",
        type=Path,
        default=Path("artifacts/models/tail_v3_ablation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_audit.json"),
    )
    args = parser.parse_args()
    result = audit_outputs(
        manifest_path=args.manifest,
        cache_path=args.cache,
        validation_summary_path=args.validation_summary,
        selected_checkpoint_path=args.selected_checkpoint,
        ablation_model_directory=args.ablation_model_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
