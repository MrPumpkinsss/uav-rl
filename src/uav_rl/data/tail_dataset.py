"""Tail-focused, resumable surrogate-v3 data generation and aggregation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from uav_rl.baselines import (
    compute_greedy_baseline,
    dynamic_programming_baseline,
    strong_link_baseline,
)
from uav_rl.config import SystemConfig
from uav_rl.data.surrogate_dataset import (
    SurrogateAction,
    _aggregate_fresh_actions,
    _atomic_write_json,
    _load_jsonl,
    _write_split_dataset,
    canonical_json_hash,
)
from uav_rl.deployment import coverage_continuous_deployment, random_continuous_deployment
from uav_rl.noise_seeds import (
    TEST_NOISE_SEED_RANGE,
    TRAIN_NOISE_SEED_RANGE,
    VALIDATION_NOISE_SEED_RANGE,
)
from uav_rl.surrogate import PPLSurrogateEnsemble, load_surrogate
from uav_rl.wireless import (
    boundary_drop_probabilities,
    collaborative_latency,
    sample_channel,
)

TailSubtype = Literal["tail_hazard", "tail_boundary", "tail_disagreement"]
TAIL_DATASET_FORMAT_VERSION = 3


@dataclass(frozen=True)
class TailDatasetConfig:
    """Deterministic counts and selectors for the tail-v3 delta."""

    action_seed: int = 20260827
    training_noise_seed: int = 20260828
    validation_noise_seed: int = 20260829
    final_test_action_seed: int = 20260830
    final_test_noise_seed: int = 20260831
    training_hazard_actions: int = 192
    training_boundary_actions: int = 160
    training_disagreement_actions: int = 160
    validation_hazard_actions: int = 24
    validation_boundary_actions: int = 20
    validation_disagreement_actions: int = 20
    training_noise_samples: int = 8
    validation_noise_samples: int = 16
    final_test_noise_samples: int = 16
    final_test_channels: int = 16
    candidate_pool: int = 256
    latency_reference_seconds: float = 1.3077757414751234

    def __post_init__(self) -> None:
        counts = (
            self.training_hazard_actions,
            self.training_boundary_actions,
            self.training_disagreement_actions,
            self.validation_hazard_actions,
            self.validation_boundary_actions,
            self.validation_disagreement_actions,
            self.training_noise_samples,
            self.validation_noise_samples,
            self.final_test_noise_samples,
            self.final_test_channels,
            self.candidate_pool,
        )
        if min(counts) < 1:
            raise ValueError("all tail-v3 counts must be positive")

    @property
    def training_actions(self) -> int:
        return (
            self.training_hazard_actions
            + self.training_boundary_actions
            + self.training_disagreement_actions
        )

    @property
    def validation_actions(self) -> int:
        return (
            self.validation_hazard_actions
            + self.validation_boundary_actions
            + self.validation_disagreement_actions
        )


@dataclass(frozen=True)
class TailSeedExtensionConfig:
    """Add independent labels to existing tail actions without changing them."""

    noise_seed: int = 20260901
    additional_training_noise_samples: int = 8
    target_training_noise_samples: int = 16

    def __post_init__(self) -> None:
        if self.additional_training_noise_samples < 1:
            raise ValueError("additional tail training noise samples must be positive")
        if self.target_training_noise_samples <= self.additional_training_noise_samples:
            raise ValueError("target noise samples must exceed the extension sample count")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name]) for name in data.files}


def _existing_seed_sets(paths: list[Path]) -> dict[str, set[int]]:
    result = {"train": set(), "validation": set(), "test": set()}
    for path in paths:
        split = "validation" if "validation" in path.name else "test" if "test" in path.name else "train"
        data = _load_npz(path)
        if "noise_seeds" in data:
            result[split].update(int(seed) for seed in data["noise_seeds"].reshape(-1))
    return result


def _sample_unique_excluding(
    rng: np.random.Generator,
    seed_range: range,
    shape: tuple[int, ...],
    excluded: set[int],
) -> np.ndarray:
    required = int(np.prod(shape))
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < required:
        candidate = int(rng.integers(seed_range.start, seed_range.stop))
        if candidate in excluded or candidate in selected_set:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
    return np.asarray(selected, dtype=np.int64).reshape(shape)


def _candidate_deployments(
    rng: np.random.Generator, system: SystemConfig, count: int
) -> np.ndarray:
    deployments = [
        coverage_continuous_deployment(rng, system)
        if index % 2 == 0
        else random_continuous_deployment(rng, system)
        for index in range(count)
    ]
    return np.asarray(deployments, dtype=np.int64)


def _candidate_statistics(
    deployments: np.ndarray, channel: np.ndarray, system: SystemConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.stack(
        [boundary_drop_probabilities(deployment, channel, system) for deployment in deployments]
    ).astype(np.float32)
    hazard = -np.log1p(-np.clip(probabilities, 0.0, 1.0 - 1e-6)).sum(axis=1)
    maximum_boundary = probabilities.max(axis=1)
    return probabilities, hazard, maximum_boundary


def _nearest_available(
    values: np.ndarray, target: float, excluded: set[int]
) -> int:
    order = np.argsort(np.abs(values - target), kind="stable")
    for index in order.tolist():
        if int(index) not in excluded:
            return int(index)
    raise RuntimeError("no unused tail candidate remains")


def _disagreement_index(
    model: PPLSurrogateEnsemble,
    probabilities: np.ndarray,
    hazard: np.ndarray,
    excluded: set[int],
    device: torch.device,
) -> int:
    eligible = np.flatnonzero((hazard >= 0.34) & (hazard <= 0.95))
    if eligible.size == 0:
        eligible = np.arange(probabilities.shape[0])
    eligible = np.asarray([index for index in eligible if int(index) not in excluded])
    if eligible.size == 0:
        eligible = np.asarray(
            [index for index in range(probabilities.shape[0]) if index not in excluded]
        )
    inputs = torch.from_numpy(probabilities[eligible]).to(device)
    with torch.no_grad():
        _, uncertainty = model.predict_with_uncertainty(inputs)
    return int(eligible[int(torch.argmax(uncertainty).item())])


def _make_action(
    *,
    action_id: str,
    split: str,
    source: str,
    group_id: str,
    channel: np.ndarray,
    deployment: np.ndarray,
    noise_seeds: np.ndarray,
    system: SystemConfig,
) -> SurrogateAction:
    probabilities = boundary_drop_probabilities(deployment, channel, system)
    latency = collaborative_latency(deployment, channel, system).total_seconds
    return SurrogateAction(
        action_id=action_id,
        split=split,  # type: ignore[arg-type]
        source=source,
        group_id=group_id,
        channel=channel.astype(np.float32).tolist(),
        deployment=deployment.astype(np.int64).tolist(),
        drop_probabilities=probabilities.astype(np.float32).tolist(),
        latency_seconds=float(latency),
        noise_seeds=noise_seeds.astype(np.int64).tolist(),
    )


def _build_tail_actions(
    *,
    split: Literal["train", "validation"],
    counts: dict[TailSubtype, int],
    noise_seeds: np.ndarray,
    rng: np.random.Generator,
    system: SystemConfig,
    config: TailDatasetConfig,
    selector: PPLSurrogateEnsemble,
    device: torch.device,
) -> list[SurrogateAction]:
    hazard_targets = (0.40, 0.50, 0.62, 0.75)
    boundary_targets = (0.15, 0.20, 0.27, 0.35)
    actions: list[SurrogateAction] = []
    noise_index = 0
    for source, count in counts.items():
        for local_index in range(count):
            channel = sample_channel(rng, system)
            deployments = _candidate_deployments(rng, system, config.candidate_pool)
            probabilities, hazard, maximum_boundary = _candidate_statistics(
                deployments, channel, system
            )
            if source == "tail_hazard":
                selected = _nearest_available(
                    hazard, hazard_targets[local_index % len(hazard_targets)], set()
                )
            elif source == "tail_boundary":
                selected = _nearest_available(
                    maximum_boundary,
                    boundary_targets[local_index % len(boundary_targets)],
                    set(),
                )
            else:
                selected = _disagreement_index(
                    selector, probabilities, hazard, set(), device
                )
            action_noise = noise_seeds[noise_index]
            noise_index += 1
            action_id = f"{split}-v3-{source}-{local_index:04d}"
            actions.append(
                _make_action(
                    action_id=action_id,
                    split=split,
                    source=source,
                    group_id=action_id,
                    channel=channel,
                    deployment=deployments[selected],
                    noise_seeds=action_noise,
                    system=system,
                )
            )
    if noise_index != noise_seeds.shape[0]:
        raise RuntimeError("tail action/noise allocation mismatch")
    return actions


def _manifest_split_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        split: Path(manifest["splits"][split]["path"])
        for split in ("train", "validation", "test")
    }


def _field_sets_from_npz(paths: list[Path]) -> dict[str, set[Any]]:
    fields: dict[str, set[Any]] = {
        "noise_seed": set(),
        "channel": set(),
        "deployment": set(),
        "channel_deployment_pair": set(),
        "drop_vector": set(),
    }
    for path in paths:
        data = _load_npz(path)
        if "noise_seeds" in data:
            fields["noise_seed"].update(int(seed) for seed in data["noise_seeds"].reshape(-1))
        channels = data["channels"].astype("<f4", copy=False)
        deployments = data["deployments"].astype("<i8", copy=False)
        drops = data["drop_probabilities"].astype("<f4", copy=False)
        for channel, deployment, drop in zip(channels, deployments, drops, strict=True):
            channel_key = channel.tobytes()
            deployment_key = deployment.tobytes()
            fields["channel"].add(channel_key)
            fields["deployment"].add(deployment_key)
            fields["channel_deployment_pair"].add((channel_key, deployment_key))
            fields["drop_vector"].add(drop.tobytes())
    return fields


def _field_sets_from_actions(actions: list[dict[str, Any]]) -> dict[str, set[Any]]:
    fields: dict[str, set[Any]] = {
        "noise_seed": set(),
        "channel": set(),
        "deployment": set(),
        "channel_deployment_pair": set(),
        "drop_vector": set(),
    }
    for action in actions:
        channel_key = np.asarray(action["channel"], dtype="<f4").tobytes()
        deployment_key = np.asarray(action["deployment"], dtype="<i8").tobytes()
        fields["noise_seed"].update(int(seed) for seed in action["noise_seeds"])
        fields["channel"].add(channel_key)
        fields["deployment"].add(deployment_key)
        fields["channel_deployment_pair"].add((channel_key, deployment_key))
        fields["drop_vector"].add(
            np.asarray(action["drop_probabilities"], dtype="<f4").tobytes()
        )
    return fields


def _merge_field_sets(*items: dict[str, set[Any]]) -> dict[str, set[Any]]:
    return {
        field: set().union(*(item[field] for item in items))
        for field in items[0]
    }


def _audit_fields(
    fields: dict[str, dict[str, set[Any]]]
) -> dict[str, Any]:
    pairs: dict[str, dict[str, int]] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        pairs[f"{left}_vs_{right}"] = {
            field: len(fields[left][field] & fields[right][field])
            for field in fields[left]
        }
    leakage_fields = (
        "noise_seed",
        "channel",
        "channel_deployment_pair",
        "drop_vector",
    )
    passed = all(
        counts[field] == 0
        for counts in pairs.values()
        for field in leakage_fields
    )
    return {
        "pairwise_overlap_counts": pairs,
        "leakage_fields": list(leakage_fields),
        "passed": passed,
        "deployment_vector_note": (
            "Raw deployments may repeat across independent channels; leakage is gated "
            "on the complete channel/deployment pair."
        ),
    }


def _audit_development_plan(
    base_paths: dict[str, Path], actions: list[dict[str, Any]]
) -> dict[str, Any]:
    by_split = {
        split: [action for action in actions if action["split"] == split]
        for split in ("train", "validation")
    }
    fields = {
        "train": _merge_field_sets(
            _field_sets_from_npz([base_paths["train"]]),
            _field_sets_from_actions(by_split["train"]),
        ),
        "validation": _merge_field_sets(
            _field_sets_from_npz([base_paths["validation"]]),
            _field_sets_from_actions(by_split["validation"]),
        ),
        "test": _field_sets_from_npz([base_paths["test"]]),
    }
    return _audit_fields(fields)


def build_tail_development_plan(
    *,
    base_manifest_path: Path,
    selector_checkpoint: Path,
    plan_path: Path,
    system: SystemConfig,
    config: TailDatasetConfig,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Build or verify the immutable train/validation tail-delta plan."""

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_paths = _manifest_split_paths(base_manifest)
    for split, path in base_paths.items():
        if _sha256(path) != base_manifest["splits"][split]["sha256"]:
            raise ValueError(f"base v2 {split} file hash does not match its manifest")
    metadata = {
        "format_version": TAIL_DATASET_FORMAT_VERSION,
        "stage": "development",
        "system": system.to_dict(),
        "generation": base_manifest["generation"],
        "tail_config": asdict(config),
        "base_manifest": {
            "path": str(base_manifest_path),
            "sha256": _sha256(base_manifest_path),
            "dataset_fingerprint": base_manifest["dataset_fingerprint"],
        },
        "selector_checkpoint": {
            "path": str(selector_checkpoint),
            "sha256": _sha256(selector_checkpoint),
        },
    }
    fingerprint = canonical_json_hash(metadata)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise ValueError("existing tail-v3 plan has an incompatible fingerprint")
        audit = _audit_development_plan(base_paths, existing["actions"])
        if not audit["passed"]:
            raise ValueError("existing tail-v3 plan fails split isolation")
        return existing

    device = torch.device(device_name)
    selector = load_surrogate(selector_checkpoint, device=device)
    if not isinstance(selector, PPLSurrogateEnsemble):
        raise TypeError("tail disagreement sampling requires an ensemble checkpoint")
    selector = selector.to(device).eval()
    existing_seeds = _existing_seed_sets(list(base_paths.values()))
    training_noise = _sample_unique_excluding(
        np.random.default_rng(config.training_noise_seed),
        TRAIN_NOISE_SEED_RANGE,
        (config.training_actions, config.training_noise_samples),
        existing_seeds["train"],
    )
    validation_seed_row = _sample_unique_excluding(
        np.random.default_rng(config.validation_noise_seed),
        VALIDATION_NOISE_SEED_RANGE,
        (config.validation_noise_samples,),
        existing_seeds["validation"],
    )
    validation_noise = np.repeat(
        validation_seed_row[None, :], config.validation_actions, axis=0
    )
    rng = np.random.default_rng(config.action_seed)
    actions = _build_tail_actions(
        split="train",
        counts={
            "tail_hazard": config.training_hazard_actions,
            "tail_boundary": config.training_boundary_actions,
            "tail_disagreement": config.training_disagreement_actions,
        },
        noise_seeds=training_noise,
        rng=rng,
        system=system,
        config=config,
        selector=selector,
        device=device,
    )
    actions.extend(
        _build_tail_actions(
            split="validation",
            counts={
                "tail_hazard": config.validation_hazard_actions,
                "tail_boundary": config.validation_boundary_actions,
                "tail_disagreement": config.validation_disagreement_actions,
            },
            noise_seeds=validation_noise,
            rng=rng,
            system=system,
            config=config,
            selector=selector,
            device=device,
        )
    )
    action_payload = [action.to_dict() for action in actions]
    if len({action["action_id"] for action in action_payload}) != len(action_payload):
        raise ValueError("tail-v3 plan contains duplicate action ids")
    audit = _audit_development_plan(base_paths, action_payload)
    if not audit["passed"]:
        raise ValueError("tail-v3 action plan fails split isolation")
    payload = {
        **metadata,
        "config_fingerprint": fingerprint,
        "isolation_audit": audit,
        "actions": action_payload,
    }
    _atomic_write_json(plan_path, payload)
    return payload


def _extension_excluded_seeds(
    development: dict[str, Any], original_plan: dict[str, Any]
) -> set[int]:
    excluded = {
        int(seed)
        for action in original_plan["actions"]
        for seed in action["noise_seeds"]
    }
    for metadata in development["base_v2"]["splits"].values():
        data = _load_npz(Path(metadata["path"]))
        if "noise_seeds" in data:
            excluded.update(int(seed) for seed in data["noise_seeds"].reshape(-1))
    return excluded


def _seed_extension_audit(
    *,
    original_plan: dict[str, Any],
    extension_actions: list[dict[str, Any]],
    excluded_seeds: set[int],
    config: TailSeedExtensionConfig,
) -> dict[str, Any]:
    original = {
        str(action["action_id"]): action
        for action in original_plan["actions"]
        if action["split"] == "train"
    }
    extension = {
        str(action["action_id"]): action for action in extension_actions
    }
    identity_fields = (
        "split",
        "source",
        "group_id",
        "channel",
        "deployment",
        "drop_probabilities",
        "latency_seconds",
    )
    identities_match = original.keys() == extension.keys() and all(
        all(original[action_id][field] == extension[action_id][field] for field in identity_fields)
        for action_id in original
    )
    original_seed_counts = {len(action["noise_seeds"]) for action in original.values()}
    extension_seed_counts = {
        len(action["noise_seeds"]) for action in extension.values()
    }
    new_seeds = [
        int(seed) for action in extension.values() for seed in action["noise_seeds"]
    ]
    overlap = set(new_seeds) & excluded_seeds
    target_matches = (
        len(original_seed_counts) == 1
        and next(iter(original_seed_counts))
        + config.additional_training_noise_samples
        == config.target_training_noise_samples
    )
    checks = {
        "same_training_action_ids": original.keys() == extension.keys(),
        "action_identity_unchanged": identities_match,
        "extension_seed_count": extension_seed_counts
        == {config.additional_training_noise_samples},
        "target_seed_count": target_matches,
        "new_seeds_globally_unique": len(new_seeds) == len(set(new_seeds)),
        "new_seeds_do_not_overlap_existing": not overlap,
        "all_extension_actions_are_train": all(
            action["split"] == "train" for action in extension.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "training_actions": len(extension),
        "additional_samples": len(new_seeds),
        "existing_seed_overlap": len(overlap),
    }


def build_tail_seed_extension_plan(
    *,
    development_manifest_path: Path,
    development_plan_path: Path,
    plan_path: Path,
    config: TailSeedExtensionConfig,
) -> dict[str, Any]:
    """Build an immutable plan containing only the second eight train seeds."""

    development = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    original_plan = json.loads(development_plan_path.read_text(encoding="utf-8"))
    if development.get("stage") != "development":
        raise ValueError("tail seed extension requires the original development manifest")
    if canonical_json_hash(original_plan) != development["plan"]["sha256"]:
        raise ValueError("tail development plan hash does not match its manifest")
    excluded = _extension_excluded_seeds(development, original_plan)
    training_actions = [
        action for action in original_plan["actions"] if action["split"] == "train"
    ]
    if not training_actions:
        raise ValueError("tail development plan contains no training actions")
    original_seed_counts = {len(action["noise_seeds"]) for action in training_actions}
    expected_original = (
        config.target_training_noise_samples
        - config.additional_training_noise_samples
    )
    if original_seed_counts != {expected_original}:
        raise ValueError(
            "tail seed extension target is incompatible with the original seed count"
        )
    metadata = {
        "format_version": TAIL_DATASET_FORMAT_VERSION,
        "stage": "seed_extension",
        "generation": development["generation"],
        "system": development["system"],
        "extension_config": asdict(config),
        "parent_development": {
            "manifest_path": str(development_manifest_path),
            "manifest_sha256": _sha256(development_manifest_path),
            "dataset_fingerprint": development["dataset_fingerprint"],
            "plan_path": str(development_plan_path),
            "plan_sha256": canonical_json_hash(original_plan),
        },
    }
    fingerprint = canonical_json_hash(metadata)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise ValueError("existing tail seed-extension plan is incompatible")
        audit = _seed_extension_audit(
            original_plan=original_plan,
            extension_actions=existing["actions"],
            excluded_seeds=excluded,
            config=config,
        )
        if not audit["passed"]:
            raise ValueError("existing tail seed-extension plan failed its audit")
        return existing

    noise_seeds = _sample_unique_excluding(
        np.random.default_rng(config.noise_seed),
        TRAIN_NOISE_SEED_RANGE,
        (len(training_actions), config.additional_training_noise_samples),
        excluded,
    )
    extension_actions = []
    for action, action_seeds in zip(training_actions, noise_seeds, strict=True):
        extension_actions.append(
            {
                **action,
                "noise_seeds": action_seeds.astype(np.int64).tolist(),
            }
        )
    audit = _seed_extension_audit(
        original_plan=original_plan,
        extension_actions=extension_actions,
        excluded_seeds=excluded,
        config=config,
    )
    if not audit["passed"]:
        raise ValueError("new tail seed-extension plan failed its audit")
    payload = {
        **metadata,
        "config_fingerprint": fingerprint,
        "extension_audit": audit,
        "actions": extension_actions,
    }
    _atomic_write_json(plan_path, payload)
    return payload


def _combine_summary_splits(paths: list[Path], output_path: Path) -> dict[str, Any]:
    shards = [_load_npz(path) for path in paths]
    fields = (
        "action_ids",
        "sample_source",
        "group_ids",
        "channels",
        "deployments",
        "drop_probabilities",
        "latency_seconds",
        "log_ppl_ratio",
        "log_ppl_ratio_std",
        "noise_seed_count",
        "has_context",
    )
    payload = {field: np.concatenate([shard[field] for shard in shards]) for field in fields}
    if len(set(payload["action_ids"].astype(str).tolist())) != len(payload["action_ids"]):
        raise ValueError("combined surrogate split contains duplicate action ids")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    sources, counts = np.unique(payload["sample_source"].astype(str), return_counts=True)
    seed_counts = payload["noise_seed_count"].astype(np.int64)
    return {
        "path": str(output_path),
        "actions": int(payload["action_ids"].size),
        "noise_seeds_per_action_min": int(seed_counts.min()),
        "noise_seeds_per_action_max": int(seed_counts.max()),
        "source_counts": dict(zip(sources.tolist(), counts.astype(int).tolist(), strict=True)),
        "sha256": _sha256(output_path),
    }


def aggregate_tail_development_dataset(
    *,
    plan: dict[str, Any],
    sample_cache_path: Path,
    base_manifest_path: Path,
    output_directory: Path,
    quality_evaluator_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write immutable delta shards plus combined train/validation summary files."""

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if _sha256(base_manifest_path) != plan["base_manifest"]["sha256"]:
        raise ValueError("base manifest changed after the tail-v3 plan was created")
    base_paths = _manifest_split_paths(base_manifest)
    fresh = _aggregate_fresh_actions(plan, _load_jsonl(sample_cache_path))
    counts = {split: len(rows) for split, rows in fresh.items()}
    expected = {
        "train": int(plan["tail_config"]["training_hazard_actions"])
        + int(plan["tail_config"]["training_boundary_actions"])
        + int(plan["tail_config"]["training_disagreement_actions"]),
        "validation": int(plan["tail_config"]["validation_hazard_actions"])
        + int(plan["tail_config"]["validation_boundary_actions"])
        + int(plan["tail_config"]["validation_disagreement_actions"]),
        "test": 0,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected tail-v3 aggregate counts: {counts}")

    train_delta_path = output_directory / "codellama_surrogate_tail_v3_train_delta.npz"
    validation_delta_path = (
        output_directory / "codellama_surrogate_tail_v3_validation_delta.npz"
    )
    train_delta = _write_split_dataset(train_delta_path, fresh["train"])
    validation_delta = _write_split_dataset(
        validation_delta_path, fresh["validation"]
    )
    train_combined = _combine_summary_splits(
        [base_paths["train"], train_delta_path],
        output_directory / "codellama_surrogate_tail_v3_train.npz",
    )
    validation_combined = _combine_summary_splits(
        [base_paths["validation"], validation_delta_path],
        output_directory / "codellama_surrogate_tail_v3_validation.npz",
    )
    fields = {
        "train": _field_sets_from_npz([base_paths["train"], train_delta_path]),
        "validation": _field_sets_from_npz(
            [base_paths["validation"], validation_delta_path]
        ),
        "test": _field_sets_from_npz([base_paths["test"]]),
    }
    isolation = _audit_fields(fields)
    if not isolation["passed"]:
        raise ValueError("aggregated tail-v3 development splits are not isolated")
    manifest = {
        "format_version": TAIL_DATASET_FORMAT_VERSION,
        "stage": "development",
        "config_fingerprint": plan["config_fingerprint"],
        "generation": plan["generation"],
        "system": plan["system"],
        "tail_config": plan["tail_config"],
        "quality_evaluator": quality_evaluator_metadata,
        "plan": {
            "path": str(
                output_directory / "codellama_surrogate_tail_v3_plan.json"
            ),
            "sha256": canonical_json_hash(plan),
        },
        "base_v2": {
            **plan["base_manifest"],
            "splits": {
                split: {
                    "path": str(path),
                    "sha256": base_manifest["splits"][split]["sha256"],
                }
                for split, path in base_paths.items()
            },
        },
        "selector_checkpoint": plan["selector_checkpoint"],
        "delta_splits": {
            "train": train_delta,
            "validation": validation_delta,
        },
        "splits": {
            "train": train_combined,
            "validation": validation_combined,
        },
        "diagnostic_test": {
            "path": str(base_paths["test"]),
            "sha256": base_manifest["splits"]["test"]["sha256"],
            "status": "development-diagnostic-only; not a final acceptance test",
        },
        "isolation_audit": isolation,
    }
    manifest["dataset_fingerprint"] = canonical_json_hash(manifest)
    _atomic_write_json(
        output_directory / "codellama_surrogate_tail_v3_manifest.json", manifest
    )
    return manifest


def aggregate_tail_seed_extension_dataset(
    *,
    extension_plan: dict[str, Any],
    extension_cache_path: Path,
    development_manifest_path: Path,
    development_plan_path: Path,
    development_cache_path: Path,
    output_directory: Path,
    quality_evaluator_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge immutable 8-seed labels with the extension into a 16-seed train shard."""

    development = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    original_plan = json.loads(development_plan_path.read_text(encoding="utf-8"))
    parent = extension_plan["parent_development"]
    if _sha256(development_manifest_path) != parent["manifest_sha256"]:
        raise ValueError("tail development manifest changed after extension planning")
    if development["dataset_fingerprint"] != parent["dataset_fingerprint"]:
        raise ValueError("tail development dataset fingerprint changed")
    if canonical_json_hash(original_plan) != parent["plan_sha256"]:
        raise ValueError("tail development plan changed after extension planning")

    extension_by_id = {
        str(action["action_id"]): action for action in extension_plan["actions"]
    }
    combined_actions: list[dict[str, Any]] = []
    for action in original_plan["actions"]:
        if action["split"] != "train":
            combined_actions.append(action)
            continue
        extension = extension_by_id.get(str(action["action_id"]))
        if extension is None:
            raise ValueError(f"missing seed extension for {action['action_id']}")
        combined_seeds = [
            *(int(seed) for seed in action["noise_seeds"]),
            *(int(seed) for seed in extension["noise_seeds"]),
        ]
        if len(combined_seeds) != len(set(combined_seeds)):
            raise ValueError(f"duplicate combined seed for {action['action_id']}")
        combined_actions.append({**action, "noise_seeds": combined_seeds})
    if len(extension_by_id) != sum(
        action["split"] == "train" for action in original_plan["actions"]
    ):
        raise ValueError("tail seed extension contains unexpected action ids")

    records = [
        *_load_jsonl(development_cache_path),
        *_load_jsonl(extension_cache_path),
    ]
    record_keys = [
        (str(record["action_id"]), int(record["noise_seed"])) for record in records
    ]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("combined tail caches contain duplicate action/seed records")
    fresh = _aggregate_fresh_actions({"actions": combined_actions}, records)
    expected_counts = {
        "train": int(development["delta_splits"]["train"]["actions"]),
        "validation": int(development["delta_splits"]["validation"]["actions"]),
        "test": 0,
    }
    observed_counts = {split: len(rows) for split, rows in fresh.items()}
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"unexpected tail seed-extension aggregate counts: {observed_counts}"
        )

    base_paths = {
        split: Path(metadata["path"])
        for split, metadata in development["base_v2"]["splits"].items()
    }
    validation_delta_path = Path(development["delta_splits"]["validation"]["path"])
    train_delta_path = (
        output_directory / "codellama_surrogate_tail_v3_seed16_train_delta.npz"
    )
    train_delta = _write_split_dataset(train_delta_path, fresh["train"])
    if train_delta["noise_seeds_per_action"] != 16:
        raise ValueError("tail seed-extension train shard did not reach 16 seeds/action")
    train_combined = _combine_summary_splits(
        [base_paths["train"], train_delta_path],
        output_directory / "codellama_surrogate_tail_v3_seed16_train.npz",
    )
    fields = {
        "train": _field_sets_from_npz([base_paths["train"], train_delta_path]),
        "validation": _field_sets_from_npz(
            [base_paths["validation"], validation_delta_path]
        ),
        "test": _field_sets_from_npz([base_paths["test"]]),
    }
    isolation = _audit_fields(fields)
    if not isolation["passed"]:
        raise ValueError("tail seed16 development splits are not isolated")

    manifest = {
        "format_version": TAIL_DATASET_FORMAT_VERSION,
        "stage": "development_seed_extension",
        "config_fingerprint": extension_plan["config_fingerprint"],
        "generation": development["generation"],
        "system": development["system"],
        "tail_config": development["tail_config"],
        "extension_config": extension_plan["extension_config"],
        "quality_evaluator": quality_evaluator_metadata,
        "parent_development": {
            "path": str(development_manifest_path),
            "sha256": _sha256(development_manifest_path),
            "dataset_fingerprint": development["dataset_fingerprint"],
        },
        "extension_plan": {
            "path": str(
                output_directory
                / "codellama_surrogate_tail_v3_seed_extension_plan.json"
            ),
            "sha256": canonical_json_hash(extension_plan),
        },
        "base_v2": development["base_v2"],
        "selector_checkpoint": development["selector_checkpoint"],
        "delta_splits": {
            "train": train_delta,
            "validation": development["delta_splits"]["validation"],
        },
        "splits": {
            "train": train_combined,
            "validation": development["splits"]["validation"],
        },
        "diagnostic_test": development["diagnostic_test"],
        "extension_audit": extension_plan["extension_audit"],
        "isolation_audit": isolation,
    }
    manifest["dataset_fingerprint"] = canonical_json_hash(manifest)
    _atomic_write_json(
        output_directory / "codellama_surrogate_tail_v3_seed16_manifest.json",
        manifest,
    )
    return manifest


def _test_tail_indices(
    *,
    selector: PPLSurrogateEnsemble,
    probabilities: np.ndarray,
    hazard: np.ndarray,
    maximum_boundary: np.ndarray,
    device: torch.device,
) -> tuple[int, int, int]:
    excluded: set[int] = set()
    hazard_index = _nearest_available(hazard, 0.68, excluded)
    excluded.add(hazard_index)
    boundary_index = _nearest_available(maximum_boundary, 0.30, excluded)
    excluded.add(boundary_index)
    disagreement_index = _disagreement_index(
        selector, probabilities, hazard, excluded, device
    )
    return hazard_index, boundary_index, disagreement_index


def build_tail_final_test_plan(
    *,
    development_manifest_path: Path,
    selector_checkpoint: Path,
    plan_path: Path,
    system: SystemConfig,
    config: TailDatasetConfig,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Build a fresh, tail-heavy final test only after validation selection."""

    development = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    if development.get("stage") != "development":
        raise ValueError("tail final test requires a development-stage manifest")
    base_paths = {
        split: Path(metadata["path"])
        for split, metadata in development["base_v2"]["splits"].items()
    }
    delta_paths = {
        split: Path(development["delta_splits"][split]["path"])
        for split in ("train", "validation")
    }
    metadata = {
        "format_version": TAIL_DATASET_FORMAT_VERSION,
        "stage": "final_test",
        "system": system.to_dict(),
        "generation": development["generation"],
        "tail_config": asdict(config),
        "development_manifest": {
            "path": str(development_manifest_path),
            "sha256": _sha256(development_manifest_path),
            "dataset_fingerprint": development["dataset_fingerprint"],
        },
        "selector_checkpoint": {
            "path": str(selector_checkpoint),
            "sha256": _sha256(selector_checkpoint),
        },
    }
    fingerprint = canonical_json_hash(metadata)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise ValueError("existing tail-v3 final-test plan is incompatible")
        return existing

    device = torch.device(device_name)
    selector = load_surrogate(selector_checkpoint, device=device)
    if not isinstance(selector, PPLSurrogateEnsemble):
        raise TypeError("tail final-test disagreement selection requires an ensemble")
    selector = selector.to(device).eval()
    excluded_test_seeds = _existing_seed_sets(
        list(base_paths.values()) + list(delta_paths.values())
    )["test"]
    test_seeds = _sample_unique_excluding(
        np.random.default_rng(config.final_test_noise_seed),
        TEST_NOISE_SEED_RANGE,
        (config.final_test_noise_samples,),
        excluded_test_seeds,
    )
    rng = np.random.default_rng(config.final_test_action_seed)
    actions: list[SurrogateAction] = []
    for channel_index in range(config.final_test_channels):
        channel = sample_channel(rng, system)
        channel_batch = channel[None, :, :]
        dynamic = dynamic_programming_baseline(
            channel_batch, system, config.latency_reference_seconds
        )[0]
        strong = strong_link_baseline(channel_batch, system)[0]
        compute = compute_greedy_baseline(channel_batch, system)
        candidates = _candidate_deployments(rng, system, config.candidate_pool)
        probabilities, hazard, maximum_boundary = _candidate_statistics(
            candidates, channel, system
        )
        hazard_index, boundary_index, disagreement_index = _test_tail_indices(
            selector=selector,
            probabilities=probabilities,
            hazard=hazard,
            maximum_boundary=maximum_boundary,
            device=device,
        )
        deployments: list[tuple[str, np.ndarray]] = [
            ("dynamic_programming", dynamic),
            ("strong_link", strong),
            ("compute_greedy", compute),
            ("random", random_continuous_deployment(rng, system)),
            ("coverage", coverage_continuous_deployment(rng, system)),
            ("tail_hazard", candidates[hazard_index]),
            ("tail_boundary", candidates[boundary_index]),
            ("tail_disagreement", candidates[disagreement_index]),
        ]
        group_id = f"test-v3-channel-{channel_index:03d}"
        for candidate_index, (source, deployment) in enumerate(deployments):
            actions.append(
                _make_action(
                    action_id=f"{group_id}-{candidate_index:02d}-{source}",
                    split="test",
                    source=source,
                    group_id=group_id,
                    channel=channel,
                    deployment=deployment,
                    noise_seeds=test_seeds,
                    system=system,
                )
            )
    action_payload = [action.to_dict() for action in actions]
    fields = {
        "train": _field_sets_from_npz([base_paths["train"], delta_paths["train"]]),
        "validation": _field_sets_from_npz(
            [base_paths["validation"], delta_paths["validation"]]
        ),
        "test": _field_sets_from_actions(action_payload),
    }
    isolation = _audit_fields(fields)
    if not isolation["passed"]:
        raise ValueError("tail-v3 final-test plan fails train/validation isolation")
    diagnostic_overlap = {
        field: len(
            fields["test"][field]
            & _field_sets_from_npz([base_paths["test"]])[field]
        )
        for field in fields["test"]
    }
    payload = {
        **metadata,
        "config_fingerprint": fingerprint,
        "isolation_audit": isolation,
        "diagnostic_test_overlap": diagnostic_overlap,
        "actions": action_payload,
    }
    _atomic_write_json(plan_path, payload)
    return payload


def aggregate_tail_final_test(
    *,
    plan: dict[str, Any],
    sample_cache_path: Path,
    development_manifest_path: Path,
    output_directory: Path,
    quality_evaluator_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Publish the fresh final test and an immutable final manifest."""

    development = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    if _sha256(development_manifest_path) != plan["development_manifest"]["sha256"]:
        raise ValueError("development manifest changed after final-test planning")
    fresh = _aggregate_fresh_actions(plan, _load_jsonl(sample_cache_path))
    if len(fresh["test"]) != 128 or fresh["train"] or fresh["validation"]:
        raise RuntimeError("tail-v3 final-test aggregate must contain exactly 128 test actions")
    test_path = output_directory / "codellama_surrogate_tail_v3_test.npz"
    test_metadata = _write_split_dataset(test_path, fresh["test"])
    train_delta = Path(development["delta_splits"]["train"]["path"])
    validation_delta = Path(development["delta_splits"]["validation"]["path"])
    base_train = Path(development["base_v2"]["splits"]["train"]["path"])
    base_validation = Path(
        development["base_v2"]["splits"]["validation"]["path"]
    )
    isolation = _audit_fields(
        {
            "train": _field_sets_from_npz([base_train, train_delta]),
            "validation": _field_sets_from_npz(
                [base_validation, validation_delta]
            ),
            "test": _field_sets_from_npz([test_path]),
        }
    )
    if not isolation["passed"]:
        raise ValueError("aggregated tail-v3 final test is not isolated")
    manifest = {
        "format_version": TAIL_DATASET_FORMAT_VERSION,
        "stage": "final_test",
        "config_fingerprint": plan["config_fingerprint"],
        "generation": plan["generation"],
        "system": plan["system"],
        "tail_config": plan["tail_config"],
        "quality_evaluator": quality_evaluator_metadata,
        "development_manifest": plan["development_manifest"],
        "selector_checkpoint": plan["selector_checkpoint"],
        "isolation_audit": isolation,
        "splits": {
            "train": development["splits"]["train"],
            "validation": development["splits"]["validation"],
            "test": test_metadata,
        },
        "delta_splits": development["delta_splits"],
        "base_v2": development["base_v2"],
        "diagnostic_test": development["diagnostic_test"],
        "diagnostic_test_overlap": plan["diagnostic_test_overlap"],
    }
    manifest["dataset_fingerprint"] = canonical_json_hash(manifest)
    _atomic_write_json(
        output_directory / "codellama_surrogate_tail_v3_final_manifest.json",
        manifest,
    )
    return manifest
