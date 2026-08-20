"""Audit the tail-v3 8-to-16-seed extension, models, and stop conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.data.surrogate_dataset import canonical_json_hash


VARIANT_FILES = {
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
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def audit_outputs(
    *,
    extension_plan_path: Path,
    extension_cache_path: Path,
    development_plan_path: Path,
    development_cache_path: Path,
    manifest_path: Path,
    validation_path: Path,
    diagnostic_path: Path,
    model_directory: Path,
    selected_checkpoint_path: Path,
) -> dict[str, Any]:
    """Independently verify the completed seed16 development pipeline."""

    extension_plan = json.loads(extension_plan_path.read_text(encoding="utf-8"))
    development_plan = json.loads(development_plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    extension_records = _load_jsonl(extension_cache_path)
    development_records = _load_jsonl(development_cache_path)
    extension_keys = [
        (str(record["action_id"]), int(record["noise_seed"]))
        for record in extension_records
    ]
    development_keys = {
        (str(record["action_id"]), int(record["noise_seed"]))
        for record in development_records
    }
    planned_keys = {
        (str(action["action_id"]), int(seed))
        for action in extension_plan["actions"]
        for seed in action["noise_seeds"]
    }
    records_by_action: dict[str, set[int]] = defaultdict(set)
    for action_id, noise_seed in extension_keys:
        records_by_action[action_id].add(noise_seed)

    checks: dict[str, bool] = {
        "extension_plan_stage": extension_plan.get("stage") == "seed_extension",
        "extension_plan_audit_passed": extension_plan.get("extension_audit", {}).get(
            "passed"
        )
        is True,
        "extension_has_512_actions": len(extension_plan.get("actions", [])) == 512,
        "extension_cache_has_4096_records": len(extension_records) == 4096,
        "extension_cache_keys_unique": len(extension_keys)
        == len(set(extension_keys))
        == 4096,
        "extension_cache_matches_plan": set(extension_keys) == planned_keys,
        "eight_new_seeds_per_action": set(
            len(seeds) for seeds in records_by_action.values()
        )
        == {8},
        "extension_keys_do_not_overlap_old_cache": not (
            set(extension_keys) & development_keys
        ),
        "seed16_manifest_stage": manifest.get("stage")
        == "development_seed_extension",
        "seed16_manifest_isolation_passed": manifest.get(
            "isolation_audit", {}
        ).get("passed")
        is True,
        "seed16_manifest_parent_matches": manifest.get(
            "parent_development", {}
        ).get("dataset_fingerprint")
        == extension_plan.get("parent_development", {}).get("dataset_fingerprint"),
        "seed16_manifest_plan_hash": manifest.get("extension_plan", {}).get(
            "sha256"
        )
        == canonical_json_hash(extension_plan),
        "validation_gate_failed": validation.get("passed") is False,
        "diagnostic_not_final_test": diagnostic.get("not_a_final_acceptance_test")
        is True,
        "diagnostic_gate_failed": diagnostic.get("validation_gate_passed") is False,
    }

    original_train_actions = {
        str(action["action_id"]): {int(seed) for seed in action["noise_seeds"]}
        for action in development_plan["actions"]
        if action["split"] == "train"
    }
    extension_train_actions = {
        str(action["action_id"]): {int(seed) for seed in action["noise_seeds"]}
        for action in extension_plan["actions"]
    }
    train_delta_metadata = manifest["delta_splits"]["train"]
    train_delta_path = Path(train_delta_metadata["path"])
    checks["train_delta_sha256"] = (
        train_delta_path.is_file()
        and _sha256(train_delta_path) == train_delta_metadata["sha256"]
    )
    with np.load(train_delta_path, allow_pickle=False) as data:
        action_ids = data["action_ids"].astype(str).tolist()
        noise_seeds = data["noise_seeds"].astype(np.int64)
        checks["train_delta_has_512_actions"] = len(action_ids) == 512
        checks["train_delta_has_16_seeds_per_action"] = (
            noise_seeds.shape == (512, 16)
            and set(data["noise_seed_count"].astype(int).tolist()) == {16}
        )
        checks["train_delta_contains_exact_old_and_new_seeds"] = all(
            set(noise_seeds[index].tolist())
            == original_train_actions[action_id] | extension_train_actions[action_id]
            for index, action_id in enumerate(action_ids)
        )

    for split, expected_actions in (("train", 2536), ("validation", 192)):
        metadata = manifest["splits"][split]
        path = Path(metadata["path"])
        checks[f"{split}_combined_sha256"] = (
            path.is_file() and _sha256(path) == metadata["sha256"]
        )
        with np.load(path, allow_pickle=False) as data:
            checks[f"{split}_combined_actions"] = (
                int(data["action_ids"].size) == expected_actions
            )
    checks["combined_train_seed_range_is_4_to_16"] = (
        manifest["splits"]["train"]["noise_seeds_per_action_min"] == 4
        and manifest["splits"]["train"]["noise_seeds_per_action_max"] == 16
    )

    dataset_fingerprint = manifest["dataset_fingerprint"]
    variant_hashes: dict[str, str] = {}
    for variant, filename in VARIANT_FILES.items():
        path = model_directory / filename
        exists = path.is_file() and path.stat().st_size > 0
        checks[f"{variant}_checkpoint_exists"] = exists
        if not exists:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checks[f"{variant}_five_members"] = (
            len(payload.get("model_states", [])) == 5
            and len(payload.get("member_metrics", [])) == 5
        )
        checks[f"{variant}_manifest_fingerprint"] = (
            payload.get("data_manifest_hash") == dataset_fingerprint
        )
        variant_hashes[variant] = _sha256(path)

    selected_variant = str(validation.get("selected_variant", ""))
    selected_hash = _sha256(selected_checkpoint_path)
    selected_gate = (
        validation.get("variants", {})
        .get(selected_variant, {})
        .get("validation_gate", {})
    )
    checks["selected_checkpoint_hash_matches"] = (
        selected_hash == validation.get("selected_checkpoint_sha256")
    )
    checks["selected_checkpoint_matches_variant"] = (
        variant_hashes.get(selected_variant) == selected_hash
    )
    checks["selected_gate_matches_summary"] = selected_gate.get(
        "passed"
    ) == validation.get("passed")
    checks["no_final_test_artifacts"] = not any(
        path.exists() for path in FINAL_ARTIFACTS
    )

    source_counts = Counter(str(record["source"]) for record in extension_records)
    return {
        "format_version": 3,
        "stage": "seed16_development_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "extension_records": len(extension_records),
        "extension_unique_keys": len(set(extension_keys)),
        "extension_source_counts": dict(source_counts),
        "selected_variant": selected_variant,
        "selected_checkpoint_sha256": selected_hash,
        "variant_checkpoint_sha256": variant_hashes,
        "input_sha256": {
            "extension_plan": _sha256(extension_plan_path),
            "extension_cache": _sha256(extension_cache_path),
            "seed16_manifest": _sha256(manifest_path),
            "validation_summary": _sha256(validation_path),
            "diagnostic": _sha256(diagnostic_path),
        },
        "final_artifacts_checked": [str(path) for path in FINAL_ARTIFACTS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-plan",
        type=Path,
        default=Path(
            "artifacts/data/codellama_surrogate_tail_v3_seed_extension_plan.json"
        ),
    )
    parser.add_argument(
        "--extension-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_tail_v3_seed_extension.jsonl"),
    )
    parser.add_argument(
        "--development-plan",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_plan.json"),
    )
    parser.add_argument(
        "--development-cache",
        type=Path,
        default=Path("artifacts/cache/surrogate_tail_v3_development.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_v3_seed16_manifest.json"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_seed16_validation.json"),
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_seed16_diagnostics.json"),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=Path("artifacts/models/tail_v3_seed16_ablation"),
    )
    parser.add_argument(
        "--selected-checkpoint",
        type=Path,
        default=Path(
            "artifacts/models/ppl_surrogate_tail_seed16_ensemble_v3_selected.pth"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_v3_seed16_audit.json"),
    )
    args = parser.parse_args()
    result = audit_outputs(
        extension_plan_path=args.extension_plan,
        extension_cache_path=args.extension_cache,
        development_plan_path=args.development_plan,
        development_cache_path=args.development_cache,
        manifest_path=args.manifest,
        validation_path=args.validation,
        diagnostic_path=args.diagnostic,
        model_directory=args.model_directory,
        selected_checkpoint_path=args.selected_checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
