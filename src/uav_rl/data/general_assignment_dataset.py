"""Resumable true-PPL labels for paper-style arbitrary layer assignments."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from uav_rl.config import DataGenerationConfig
from uav_rl.noise_seeds import (
    TEST_NOISE_SEED_RANGE,
    TRAIN_NOISE_SEED_RANGE,
    VALIDATION_NOISE_SEED_RANGE,
)
from uav_rl.resource_assignment import (
    ResourceConstrainedConfig,
    layerwise_drop_probabilities,
    layerwise_latency,
    validate_layerwise_deployment,
)
from uav_rl.resource_environment import generate_resource_channels
from uav_rl.true_quality import TruePPLQualityEvaluator

SplitName = Literal["train", "validation", "test"]
FORMAT_VERSION = 3


@dataclass(frozen=True)
class GeneralAssignmentDatasetConfig:
    """Action and label counts for one general-assignment surrogate dataset."""

    action_seed: int = 20260819
    training_noise_seed: int = 2026081901
    validation_noise_seed: int = 2026081902
    test_noise_seed: int = 2026081903
    train_actions: int = 384
    validation_actions: int = 64
    test_actions: int = 64
    training_noise_samples: int = 4
    validation_noise_samples: int = 16
    test_noise_samples: int = 16
    max_generation_attempts: int = 20_000
    latency_reference_seconds: float = 1.3077757414751234

    def __post_init__(self) -> None:
        if min(
            self.train_actions,
            self.validation_actions,
            self.test_actions,
            self.training_noise_samples,
            self.validation_noise_samples,
            self.test_noise_samples,
            self.max_generation_attempts,
        ) < 1:
            raise ValueError("dataset counts and generation attempts must be positive")


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing_seeds(cache_paths: tuple[Path, ...]) -> set[int]:
    values: set[int] = set()
    for path in cache_paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                values.add(int(json.loads(line)["noise_seed"]))
    return values


def _fresh_seeds(
    rng: np.random.Generator, seed_range: range, count: int, excluded: set[int]
) -> np.ndarray:
    selected: list[int] = []
    while len(selected) < count:
        candidate = int(rng.integers(seed_range.start, seed_range.stop))
        if candidate not in excluded and candidate not in selected:
            selected.append(candidate)
    return np.asarray(selected, dtype=np.int64)


def _composition(rng: np.random.Generator, total: int, parts: int) -> np.ndarray:
    # Split layers into non-empty contiguous runs.
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(np.int64)


def sample_general_assignment(
    rng: np.random.Generator,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
    *,
    target_boundaries: int,
    max_attempts: int,
) -> np.ndarray:
    """Sample a feasible layer assignment with an approximately chosen boundary count."""

    if not 0 <= target_boundaries < config.system.num_layers:
        raise ValueError("target_boundaries is out of range")
    capacities = np.asarray(config.uav_memory_capacity_units, dtype=np.float64)
    budgets = np.asarray(config.uav_energy_budget_joule, dtype=np.float64)
    hover = np.asarray(config.uav_hover_energy_joule, dtype=np.float64)
    layer_memory = np.asarray(config.layer_memory_units, dtype=np.float64)
    compute_energy_per_layer = (
        config.compute_energy_coefficient
        * np.asarray(config.system.compute_speed, dtype=np.float64)[:, None] ** 2
        * np.asarray(config.layer_compute_seconds_at_unit_speed, dtype=np.float64)[None, :]
    )
    for _ in range(max_attempts):
        run_lengths = _composition(rng, config.system.num_layers, target_boundaries + 1)
        ends = np.cumsum(run_lengths)
        starts = np.concatenate(([0], ends[:-1]))
        run_memory = np.asarray(
            [layer_memory[start:end].sum() for start, end in zip(starts, ends, strict=True)]
        )
        run_energy = np.stack(
            [
                compute_energy_per_layer[:, start:end].sum(axis=1)
                for start, end in zip(starts, ends, strict=True)
            ]
        )
        if any(
            not np.any(run_memory[index] <= capacities + 1e-9)
            for index in range(len(run_lengths))
        ):
            continue
        assignment_by_run = np.full(len(run_lengths), -1, dtype=np.int64)
        memory_used = np.zeros(config.system.num_uavs, dtype=np.float64)
        energy_used = hover.copy()

        def assign_run(run_index: int, previous_uav: int) -> bool:
            if run_index == len(run_lengths):
                return True
            feasible = np.flatnonzero(
                (memory_used + run_memory[run_index] <= capacities + 1e-9)
                & (energy_used + run_energy[run_index] <= budgets + 1e-9)
            )
            feasible = feasible[feasible != previous_uav]
            if not len(feasible):
                return False
            rng.shuffle(feasible)
            projected = np.maximum(
                (memory_used[feasible] + run_memory[run_index]) / capacities[feasible],
                (energy_used[feasible] + run_energy[run_index, feasible]) / budgets[feasible],
            )
            for uav in feasible[np.argsort(projected, kind='stable')]:
                uav = int(uav)
                memory_used[uav] += run_memory[run_index]
                energy_used[uav] += run_energy[run_index, uav]
                assignment_by_run[run_index] = uav
                if assign_run(run_index + 1, uav):
                    return True
                memory_used[uav] -= run_memory[run_index]
                energy_used[uav] -= run_energy[run_index, uav]
                assignment_by_run[run_index] = -1
            return False

        if not assign_run(0, -1):
            continue
        assignment = np.repeat(assignment_by_run, run_lengths)
        try:
            validate_layerwise_deployment(assignment, config, channel=channel)
        except ValueError:
            continue
        return assignment
    raise RuntimeError(
        f"could not sample a feasible assignment with {target_boundaries} boundaries "
        f"after {max_attempts} attempts"
    )


def _source_target(source: str, rng: np.random.Generator) -> int:
    if source == "low_boundary":
        return int(rng.integers(3, 6))
    if source == "medium_boundary":
        return int(rng.integers(6, 11))
    if source == "high_boundary":
        return int(rng.integers(11, 14))
    return int(rng.integers(3, 14))


def _sources(count: int) -> list[str]:
    families = ["low_boundary", "medium_boundary", "high_boundary", "resource_balanced"]
    return [families[index % len(families)] for index in range(count)]


def _build_actions(
    split: SplitName,
    count: int,
    channel_seed: int,
    action_seed: int,
    config: ResourceConstrainedConfig,
    dataset: GeneralAssignmentDatasetConfig,
) -> list[dict[str, Any]]:
    channels = generate_resource_channels(count, channel_seed, config)
    rng = np.random.default_rng(action_seed)
    actions: list[dict[str, Any]] = []
    for index, (channel, source) in enumerate(zip(channels, _sources(count), strict=True)):
        deployment = sample_general_assignment(
            rng,
            channel,
            config,
            target_boundaries=_source_target(source, rng),
            max_attempts=dataset.max_generation_attempts,
        )
        drops = layerwise_drop_probabilities(deployment, channel, config)
        latency = layerwise_latency(deployment, channel, config).total_seconds
        actions.append(
            {
                "action_id": f"general-{split}-{index:04d}",
                "split": split,
                "source": source,
                "group_id": f"general-{split}-{index:04d}",
                "channel": channel.tolist(),
                "deployment": deployment.tolist(),
                "drop_probabilities": drops.tolist(),
                "latency_seconds": float(latency),
                "boundary_count": int(np.count_nonzero(drops)),
            }
        )
    return actions


def build_general_assignment_plan(
    *,
    config: ResourceConstrainedConfig,
    generation: DataGenerationConfig,
    dataset: GeneralAssignmentDatasetConfig,
    plan_path: Path,
    existing_cache_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Create a deterministic, split-isolated action/noise plan."""

    excluded = _existing_seeds(existing_cache_paths)
    train_seeds = _fresh_seeds(
        np.random.default_rng(dataset.training_noise_seed),
        TRAIN_NOISE_SEED_RANGE,
        dataset.train_actions * dataset.training_noise_samples,
        excluded,
    ).reshape(dataset.train_actions, dataset.training_noise_samples)
    excluded |= set(train_seeds.ravel().tolist())
    validation_seeds = _fresh_seeds(
        np.random.default_rng(dataset.validation_noise_seed),
        VALIDATION_NOISE_SEED_RANGE,
        dataset.validation_actions * dataset.validation_noise_samples,
        excluded,
    ).reshape(dataset.validation_actions, dataset.validation_noise_samples)
    excluded |= set(validation_seeds.ravel().tolist())
    test_seeds = _fresh_seeds(
        np.random.default_rng(dataset.test_noise_seed),
        TEST_NOISE_SEED_RANGE,
        dataset.test_actions * dataset.test_noise_samples,
        excluded,
    ).reshape(dataset.test_actions, dataset.test_noise_samples)
    immutable = {
        "format_version": FORMAT_VERSION,
        "purpose": "general_layer_assignment_surrogate_dataset",
        "resource_config": config.to_dict(),
        "generation": asdict(generation),
        "dataset_config": asdict(dataset),
        "existing_cache_paths": [str(path) for path in existing_cache_paths],
    }
    fingerprint = _hash(immutable)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise ValueError("general assignment plan configuration differs")
        return existing
    actions = _build_actions(
        "train", dataset.train_actions, dataset.action_seed, dataset.action_seed, config, dataset
    )
    actions += _build_actions(
        "validation",
        dataset.validation_actions,
        dataset.action_seed + 1,
        dataset.action_seed + 101,
        config,
        dataset,
    )
    actions += _build_actions(
        "test",
        dataset.test_actions,
        dataset.action_seed + 2,
        dataset.action_seed + 202,
        config,
        dataset,
    )
    for action, seeds in zip(
        actions[: dataset.train_actions], train_seeds, strict=True
    ):
        action["noise_seeds"] = seeds.tolist()
    offset = dataset.train_actions
    for action, seeds in zip(
        actions[offset : offset + dataset.validation_actions], validation_seeds, strict=True
    ):
        action["noise_seeds"] = seeds.tolist()
    offset += dataset.validation_actions
    for action, seeds in zip(actions[offset:], test_seeds, strict=True):
        action["noise_seeds"] = seeds.tolist()
    payload = {**immutable, "config_fingerprint": fingerprint, "actions": actions}
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Match the JSON-normalized representation returned on a later resume.
    return json.loads(plan_path.read_text(encoding='utf-8'))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()


def collect_general_assignment_labels(
    plan: dict[str, Any],
    evaluator: TruePPLQualityEvaluator,
    cache_path: Path,
    *,
    progress_interval: int = 50,
) -> dict[str, int]:
    """Evaluate each planned action/seed once and resume from JSONL cache."""

    completed: set[tuple[str, int]] = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((str(row["action_id"]), int(row["noise_seed"])))
    expected = {
        (str(action["action_id"]), int(seed))
        for action in plan["actions"]
        for seed in action["noise_seeds"]
    }
    unknown = completed - expected
    if unknown:
        raise ValueError(f"cache contains unknown general-assignment keys: {sorted(unknown)[:3]}")
    newly_completed = 0
    started = time.perf_counter()
    for action in plan["actions"]:
        probabilities = np.asarray(action["drop_probabilities"], dtype=np.float32)
        for seed in action["noise_seeds"]:
            key = (str(action["action_id"]), int(seed))
            if key in completed:
                continue
            before = evaluator.model_forwards
            sample_started = time.perf_counter()
            quality = float(
                evaluator.evaluate(
                    probabilities[None, :], noise_seeds=np.asarray([seed], dtype=np.int64)
                )[0]
            )
            _append_jsonl(
                cache_path,
                {
                    "format_version": FORMAT_VERSION,
                    "config_fingerprint": plan["config_fingerprint"],
                    **action,
                    "noise_seed": int(seed),
                    "perplexity": evaluator.clean_perplexity * math.exp(quality),
                    "log_ppl_ratio": quality,
                    "evaluation_seconds": time.perf_counter() - sample_started,
                    "ppl_cache_hit": evaluator.model_forwards == before,
                },
            )
            completed.add(key)
            newly_completed += 1
            if progress_interval and newly_completed % progress_interval == 0:
                remaining = len(expected) - len(completed)
                elapsed = time.perf_counter() - started
                rate = newly_completed / max(elapsed, 1e-9)
                print(
                    f"general_assignment_samples={len(completed)}/{len(expected)} "
                    f"eta_seconds={remaining / max(rate, 1e-9):.1f}",
                    flush=True,
                )
    return {
        "expected_samples": len(expected),
        "completed_samples": len(completed),
        "new_samples": newly_completed,
    }


def aggregate_general_assignment_cache(
    plan: dict[str, Any],
    cache_path: Path,
    output_directory: Path,
    *,
    legacy_train_path: Path | None = None,
    quality_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Aggregate JSONL labels into split NPZ files and write a leakage audit manifest."""

    records: dict[str, list[dict[str, Any]]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records.setdefault(str(row["action_id"]), []).append(row)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for action in plan["actions"]:
        action_rows = records.get(str(action["action_id"]), [])
        if {int(row["noise_seed"]) for row in action_rows} != set(map(int, action["noise_seeds"])):
            raise ValueError(f"incomplete general assignment labels for {action['action_id']}")
        qualities = np.asarray([row["log_ppl_ratio"] for row in action_rows], dtype=np.float32)
        row = {
            "action_id": action["action_id"],
            "sample_source": action["source"],
            "group_id": action["group_id"],
            "channels": np.asarray(action["channel"], dtype=np.float32),
            "deployments": np.asarray(action["deployment"], dtype=np.int64),
            "drop_probabilities": np.asarray(action["drop_probabilities"], dtype=np.float32),
            "latency_seconds": np.float32(action["latency_seconds"]),
            "log_ppl_ratio": np.float32(qualities.mean()),
            "log_ppl_ratio_std": np.float32(qualities.std(ddof=0)),
            "noise_seed_count": np.int32(len(qualities)),
            "has_context": np.bool_(True),
        }
        rows[str(action["split"])].append(row)
    if legacy_train_path is not None:
        with np.load(legacy_train_path) as legacy:
            required = {
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
            }
            if missing := required.difference(legacy.files):
                raise ValueError(f"legacy train split is missing fields: {sorted(missing)}")
            for index in range(len(legacy["action_ids"])):
                rows["train"].append(
                    {
                        "action_id": str(legacy["action_ids"][index]),
                        "sample_source": f"legacy_{legacy['sample_source'][index]}",
                        "group_id": str(legacy["group_ids"][index]),
                        "channels": legacy["channels"][index],
                        "deployments": legacy["deployments"][index],
                        "drop_probabilities": legacy["drop_probabilities"][index],
                        "latency_seconds": legacy["latency_seconds"][index],
                        "log_ppl_ratio": legacy["log_ppl_ratio"][index],
                        "log_ppl_ratio_std": legacy["log_ppl_ratio_std"][index],
                        "noise_seed_count": legacy["noise_seed_count"][index],
                        "has_context": legacy["has_context"][index],
                    }
                )
    def split_audit_key(row: dict[str, Any]) -> tuple[bytes, bytes]:
        channel_key = np.asarray(row['channels'], dtype='<f4').tobytes()
        deployment_key = np.asarray(row['deployments'], dtype='<i8').tobytes()
        return channel_key, deployment_key

    def drop_key(row: dict[str, Any]) -> bytes:
        return np.asarray(row['drop_probabilities'], dtype='<f4').tobytes()

    split_pair_overlap = 0
    split_drop_overlap = 0
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        split_pair_overlap += len(
            {split_audit_key(row) for row in rows[left]}
            & {split_audit_key(row) for row in rows[right]}
        )
        split_drop_overlap += len(
            {drop_key(row) for row in rows[left]}
            & {drop_key(row) for row in rows[right]}
        )
    planned_seeds = {
        split: {
            int(seed)
            for action in plan['actions']
            if action['split'] == split
            for seed in action['noise_seeds']
        }
        for split in ('train', 'validation', 'test')
    }
    split_seed_overlap = sum(
        len(planned_seeds[left] & planned_seeds[right])
        for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))
    )
    if split_pair_overlap or split_drop_overlap or split_seed_overlap:
        raise ValueError('general-assignment splits are not isolated')

    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, split_rows in rows.items():
        path = output_directory / f"general_assignment_{split}.npz"
        np.savez_compressed(
            path,
            **{
                key: np.asarray([row[key] for row in split_rows])
                for key in (
                    "action_id",
                    "sample_source",
                    "group_id",
                    "channels",
                    "deployments",
                    "drop_probabilities",
                    "latency_seconds",
                    "log_ppl_ratio",
                    "log_ppl_ratio_std",
                    "noise_seed_count",
                    "has_context",
                )
            },
        )
        with np.load(path) as saved:
            normalized = {name: np.array(saved[name]) for name in saved.files}
        normalized['group_ids'] = normalized.pop('group_id')
        np.savez_compressed(
            path,
            action_ids=normalized.pop("action_id"),
            **normalized,
        )
        paths[split] = path
    manifest = {
        "format_version": FORMAT_VERSION,
        "purpose": "general_assignment_surrogate_labels",
        "config_fingerprint": plan["config_fingerprint"],
        "resource_config": plan["resource_config"],
        "generation": plan["generation"],
        "splits": {
            split: {
                "path": str(path),
                "actions": len(rows[split]),
                "sha256": _file_sha256(path),
            }
            for split, path in paths.items()
        },
        "isolation_audit": {
            "passed": True,
            "noise_seed_overlap": split_seed_overlap,
            "channel_deployment_pair_overlap": split_pair_overlap,
            "drop_probability_overlap": split_drop_overlap,
            "diagnostic_labels_excluded": True,
        },
        "quality_evaluator": quality_metadata
        or {
            "model_id": plan["generation"]["model_id"],
            "clean_perplexity": "recorded_by_collection_cli",
            "evaluated_sequences": "recorded_by_collection_cli",
            "evaluated_tokens": "recorded_by_collection_cli",
        },
        "ppo_training_context": {"status": "not_applicable_general_assignment_dataset"},
        "legacy_train": (
            {"path": str(legacy_train_path), "sha256": _file_sha256(legacy_train_path)}
            if legacy_train_path is not None
            else None
        ),
    }
    manifest_path = output_directory / "general_assignment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths
