"""Resumable seed extensions for high-variance surrogate training labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_rl.data.surrogate_dataset import (
    _aggregate_fresh_actions,
    _atomic_write_json,
    _load_jsonl,
    canonical_json_hash,
)
from uav_rl.data.tail_dataset import _sample_unique_excluding
from uav_rl.data.tail_seed24 import _full_tail_actions, _identity
from uav_rl.noise_seeds import TRAIN_NOISE_SEED_RANGE


@dataclass(frozen=True)
class TargetedLabelConfig:
    """Target seed counts for sources whose labels dominate validation error."""

    training_noise_seed: int = 2026081801
    source_targets: tuple[tuple[str, int], ...] = (
        ("coverage", 16),
        ("tail_disagreement", 48),
    )
    expected_action_counts: tuple[tuple[str, int], ...] = (
        ("coverage", 320),
        ("tail_disagreement", 160),
    )

    def target_by_source(self) -> dict[str, int]:
        return dict(self.source_targets)

    def expected_by_source(self) -> dict[str, int]:
        return dict(self.expected_action_counts)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_plan_seeds(*plans: dict[str, Any]) -> set[int]:
    return {
        int(seed)
        for plan in plans
        for action in plan["actions"]
        for seed in action["noise_seeds"]
    }


def _full_seed24_tail_actions(
    v2_plan: dict[str, Any],
    v3_plan: dict[str, Any],
    seed16_plan: dict[str, Any],
    seed24_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Combine the historical tail plans into their 24-seed parent actions."""

    extension_by_id = {str(action["action_id"]): action for action in seed24_plan["actions"]}
    combined: list[dict[str, Any]] = []
    for action in _full_tail_actions(v2_plan, v3_plan, seed16_plan):
        extension = extension_by_id.get(str(action["action_id"]))
        if extension is None:
            raise ValueError(f"missing seed24 extension for {action['action_id']}")
        seeds = [*(int(seed) for seed in action["noise_seeds"]), *(int(seed) for seed in extension["noise_seeds"])]
        if len(seeds) != 24 or len(seeds) != len(set(seeds)):
            raise ValueError(f"invalid seed24 parent action: {action['action_id']}")
        combined.append({**_identity(action), "noise_seeds": seeds})
    if len(extension_by_id) != len(combined):
        raise ValueError("seed24 plan contains unexpected action identifiers")
    return combined


def build_targeted_seed_plan(
    *,
    v2_plan_path: Path,
    v3_plan_path: Path,
    seed16_plan_path: Path,
    seed24_plan_path: Path,
    ppo_cache_path: Path,
    plan_path: Path,
    config: TargetedLabelConfig,
) -> dict[str, Any]:
    """Build an immutable, train-only seed extension for selected action sources."""

    v2_plan = _load(v2_plan_path)
    v3_plan = _load(v3_plan_path)
    seed16_plan = _load(seed16_plan_path)
    seed24_plan = _load(seed24_plan_path)
    target_by_source = config.target_by_source()
    expected_by_source = config.expected_by_source()
    if set(target_by_source) != set(expected_by_source):
        raise ValueError("source targets and expected action counts must have identical keys")

    parent_actions = [
        {**_identity(action), "noise_seeds": list(action["noise_seeds"])}
        for action in v2_plan["actions"]
        if action["split"] == "train" and action["source"] == "coverage"
    ]
    parent_actions.extend(
        action
        for action in _full_seed24_tail_actions(v2_plan, v3_plan, seed16_plan, seed24_plan)
        if action["split"] == "train" and action["source"] == "tail_disagreement"
    )
    counts = {
        source: sum(action["source"] == source for action in parent_actions)
        for source in target_by_source
    }
    if counts != expected_by_source:
        raise ValueError(f"unexpected targeted action counts: {counts}")

    metadata = {
        "format_version": 1,
        "stage": "targeted_label_seed_extension",
        "config": asdict(config),
        "parents": {
            "v2_plan": {"path": str(v2_plan_path), "sha256": _sha256(v2_plan_path)},
            "v3_plan": {"path": str(v3_plan_path), "sha256": _sha256(v3_plan_path)},
            "seed16_plan": {"path": str(seed16_plan_path), "sha256": _sha256(seed16_plan_path)},
            "seed24_plan": {"path": str(seed24_plan_path), "sha256": _sha256(seed24_plan_path)},
            "ppo_cache": {"path": str(ppo_cache_path), "sha256": _sha256(ppo_cache_path)},
        },
    }
    fingerprint = canonical_json_hash(metadata)
    if plan_path.exists():
        existing = _load(plan_path)
        if existing.get("config_fingerprint") != fingerprint:
            raise ValueError("existing targeted seed plan is incompatible")
        return existing

    excluded = _all_plan_seeds(v2_plan, v3_plan, seed16_plan, seed24_plan)
    excluded.update(int(record["noise_seed"]) for record in _load_jsonl(ppo_cache_path))
    rng = np.random.default_rng(config.training_noise_seed)
    extensions: list[dict[str, Any]] = []
    for source in sorted(target_by_source):
        group = [action for action in parent_actions if action["source"] == source]
        current_counts = {len(action["noise_seeds"]) for action in group}
        if len(current_counts) != 1:
            raise ValueError(f"target source has mixed seed counts: {source}")
        additional = target_by_source[source] - current_counts.pop()
        if additional < 1:
            raise ValueError(f"target source already meets requested count: {source}")
        sampled = _sample_unique_excluding(rng, TRAIN_NOISE_SEED_RANGE, (len(group), additional), excluded)
        excluded.update(int(seed) for seed in sampled.reshape(-1))
        extensions.extend(
            {**_identity(action), "noise_seeds": seeds.astype(np.int64).tolist()}
            for action, seeds in zip(group, sampled, strict=True)
        )

    new_seeds = [int(seed) for action in extensions for seed in action["noise_seeds"]]
    audit = {
        "train_only": all(action["split"] == "train" for action in extensions),
        "new_seed_count": len(new_seeds),
        "new_seeds_unique": len(new_seeds) == len(set(new_seeds)),
        "all_new_seeds_in_training_range": all(seed in TRAIN_NOISE_SEED_RANGE for seed in new_seeds),
        "source_action_counts": {
            source: sum(action["source"] == source for action in extensions)
            for source in target_by_source
        },
        "target_seed_counts": {
            source: all(
                len(action["noise_seeds"]) + target_by_source[source] - len(action["noise_seeds"])
                == target_by_source[source]
                for action in parent_actions
                if action["source"] == source
            )
            for source in target_by_source
        },
    }
    audit["expected_action_counts"] = audit["source_action_counts"] == expected_by_source
    audit["passed"] = all(
        audit[key]
        for key in (
            "train_only",
            "new_seeds_unique",
            "all_new_seeds_in_training_range",
            "expected_action_counts",
        )
    ) and all(audit["target_seed_counts"].values())
    if not audit["passed"]:
        raise ValueError(f"targeted seed extension audit failed: {audit}")
    payload = {**metadata, "config_fingerprint": fingerprint, "audit": audit, "actions": extensions}
    _atomic_write_json(plan_path, payload)
    return payload


def _merge_target_rows(
    *, parent_train_path: Path, rows: list[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    """Replace only selected aggregate labels while preserving all parent rows verbatim."""

    with np.load(parent_train_path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    indices = {str(action_id): index for index, action_id in enumerate(payload["action_ids"])}
    for row in rows:
        index = indices.get(str(row["action_id"]))
        if index is None:
            raise ValueError(f"target action absent from parent train data: {row['action_id']}")
        if str(payload["sample_source"][index]) != str(row["source"]):
            raise ValueError(f"target action source changed: {row['action_id']}")
        if not np.array_equal(payload["drop_probabilities"][index], row["drop_probabilities"]):
            raise ValueError(f"target action drop vector changed: {row['action_id']}")
        qualities = np.asarray(row["qualities"], dtype=np.float64)
        payload["log_ppl_ratio"][index] = qualities.mean()
        payload["log_ppl_ratio_std"][index] = qualities.std()
        payload["noise_seed_count"][index] = len(qualities)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(output_path)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "actions": int(len(payload["action_ids"])),
        "source_seed_counts": {
            source: sorted({int(value) for value in payload["noise_seed_count"][payload["sample_source"].astype(str) == source]})
            for source in ("coverage", "tail_disagreement")
        },
    }


def aggregate_targeted_training_data(
    *,
    parent_manifest_path: Path,
    v2_plan_path: Path,
    v3_plan_path: Path,
    seed16_plan_path: Path,
    seed24_plan_path: Path,
    targeted_plan_path: Path,
    cache_paths: tuple[Path, ...],
    output_train_path: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    """Aggregate completed extensions into a new train NPZ without touching validation."""

    parent_manifest = _load(parent_manifest_path)
    v2_plan, v3_plan, seed16_plan, seed24_plan, targeted_plan = (
        _load(path)
        for path in (v2_plan_path, v3_plan_path, seed16_plan_path, seed24_plan_path, targeted_plan_path)
    )
    target_by_source = dict(targeted_plan["config"]["source_targets"])
    base_actions = [
        {**_identity(action), "noise_seeds": list(action["noise_seeds"])}
        for action in v2_plan["actions"]
        if action["split"] == "train" and action["source"] == "coverage"
    ]
    base_actions.extend(
        action
        for action in _full_seed24_tail_actions(v2_plan, v3_plan, seed16_plan, seed24_plan)
        if action["split"] == "train" and action["source"] == "tail_disagreement"
    )
    extension_by_id = {str(action["action_id"]): action for action in targeted_plan["actions"]}
    full_actions = []
    for action in base_actions:
        extension = extension_by_id.get(str(action["action_id"]))
        if extension is None:
            raise ValueError(f"missing targeted extension for {action['action_id']}")
        expected = target_by_source[action["source"]]
        seeds = [*(int(seed) for seed in action["noise_seeds"]), *(int(seed) for seed in extension["noise_seeds"])]
        if len(seeds) != expected or len(seeds) != len(set(seeds)):
            raise ValueError(f"target seed count is invalid: {action['action_id']}")
        full_actions.append({**_identity(action), "noise_seeds": seeds})
    if len(extension_by_id) != len(full_actions):
        raise ValueError("targeted plan contains unexpected action identifiers")

    records = [record for path in cache_paths for record in _load_jsonl(path)]
    keys = [(str(record["action_id"]), int(record["noise_seed"])) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("parent and extension caches contain duplicate action/seed records")
    rows = _aggregate_fresh_actions({"actions": full_actions}, records)["train"]
    parent_train_path = Path(parent_manifest["splits"]["train"]["path"])
    train = _merge_target_rows(parent_train_path=parent_train_path, rows=rows, output_path=output_train_path)
    validation = parent_manifest["splits"]["validation"]
    manifest = {
        "format_version": 3,
        "stage": "targeted_label_extension",
        "not_a_final_acceptance_test": True,
        "parent_manifest": {"path": str(parent_manifest_path), "sha256": _sha256(parent_manifest_path)},
        "targeted_plan": {"path": str(targeted_plan_path), "sha256": _sha256(targeted_plan_path)},
        "cache_paths": [{"path": str(path), "sha256": _sha256(path)} for path in cache_paths],
        "splits": {"train": train, "validation": validation},
        "isolation_audit": parent_manifest["isolation_audit"],
    }
    if not manifest["isolation_audit"].get("passed"):
        raise ValueError("parent dataset isolation audit did not pass")
    manifest["dataset_fingerprint"] = canonical_json_hash(manifest)
    _atomic_write_json(output_manifest_path, manifest)
    return manifest
