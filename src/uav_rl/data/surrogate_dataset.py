"""Resumable multi-seed dataset collection for robust PPL surrogates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from uav_rl.baselines import (
    compute_greedy_baseline,
    dynamic_programming_baseline,
    strong_link_baseline,
)
from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.deployment import coverage_continuous_deployment, random_continuous_deployment
from uav_rl.noise_seeds import (
    TEST_NOISE_SEED_RANGE,
    TRAIN_NOISE_SEED_RANGE,
    VALIDATION_NOISE_SEED_RANGE,
)
from uav_rl.true_quality import TruePPLQualityEvaluator
from uav_rl.wireless import (
    boundary_drop_probabilities,
    collaborative_latency,
    sample_channel,
)

SplitName = Literal["train", "validation", "test"]
DATASET_FORMAT_VERSION = 2
ACTION_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SurrogateDatasetConfig:
    """Reproducible action and noise sampling configuration."""

    action_seed: int = 20260818
    training_noise_seed: int = 20260819
    validation_noise_seed: int = 20260820
    test_noise_seed: int = 20260821
    training_noise_samples: int = 4
    validation_noise_samples: int = 16
    test_noise_samples: int = 16
    validation_channels: int = 16
    test_channels: int = 16
    tail_candidate_pool: int = 64
    latency_reference_seconds: float = 1.3077757414751234

    def __post_init__(self) -> None:
        if min(
            self.training_noise_samples,
            self.validation_noise_samples,
            self.test_noise_samples,
            self.validation_channels,
            self.test_channels,
            self.tail_candidate_pool,
        ) < 1:
            raise ValueError("all dataset counts must be positive")


@dataclass(frozen=True)
class SurrogateAction:
    """One channel-conditioned deployment with a deterministic seed set."""

    action_id: str
    split: SplitName
    source: str
    group_id: str
    channel: list[list[float]]
    deployment: list[int]
    drop_probabilities: list[float]
    latency_seconds: float
    noise_seeds: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_hash(payload: Any) -> str:
    """Return a stable SHA256 for serializable experiment metadata."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sample_seeds_excluding(
    rng: np.random.Generator,
    seed_range: range,
    count: int,
    excluded: set[int],
) -> np.ndarray:
    """Sample globally unique seeds while excluding prior experiment values."""

    if count < 1:
        raise ValueError("seed count must be positive")
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < count:
        candidate = int(rng.integers(seed_range.start, seed_range.stop))
        if candidate in excluded or candidate in selected_set:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
    return np.asarray(selected, dtype=np.int64)


def _load_existing_noise_seeds(cache_path: Path) -> dict[SplitName, set[int]]:
    seeds: dict[SplitName, set[int]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    if not cache_path.exists():
        return seeds
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        noise_seed = int(record["noise_seed"])
        if noise_seed in TRAIN_NOISE_SEED_RANGE:
            seeds["train"].add(noise_seed)
        elif noise_seed in VALIDATION_NOISE_SEED_RANGE:
            seeds["validation"].add(noise_seed)
        elif noise_seed in TEST_NOISE_SEED_RANGE:
            seeds["test"].add(noise_seed)
    return seeds


def _tail_deployment(
    rng: np.random.Generator,
    channel: np.ndarray,
    system: SystemConfig,
    candidate_pool: int,
) -> np.ndarray:
    """Select the candidate with the largest cumulative packet-loss hazard."""

    best_deployment: np.ndarray | None = None
    best_score = -float("inf")
    for index in range(candidate_pool):
        deployment = (
            coverage_continuous_deployment(rng, system)
            if index % 2 == 0
            else random_continuous_deployment(rng, system)
        )
        probabilities = boundary_drop_probabilities(deployment, channel, system)
        score = float(-np.log1p(-probabilities.clip(max=1.0 - 1e-6)).sum())
        if score > best_score:
            best_score = score
            best_deployment = deployment
    assert best_deployment is not None
    return best_deployment


def _make_action(
    *,
    action_id: str,
    split: SplitName,
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
        split=split,
        source=source,
        group_id=group_id,
        channel=channel.astype(np.float32).tolist(),
        deployment=deployment.astype(np.int64).tolist(),
        drop_probabilities=probabilities.astype(np.float32).tolist(),
        latency_seconds=float(latency),
        noise_seeds=noise_seeds.astype(np.int64).tolist(),
    )


def _build_training_actions(
    rng: np.random.Generator,
    noise_seeds: np.ndarray,
    system: SystemConfig,
    config: SurrogateDatasetConfig,
) -> list[SurrogateAction]:
    source_counts = (
        ("coverage", 320),
        ("random", 256),
        ("tail", 128),
        ("strong_link", 128),
        ("dynamic_programming", 128),
        ("compute_greedy", 64),
    )
    actions: list[SurrogateAction] = []
    noise_index = 0
    for source, count in source_counts:
        channels = np.stack([sample_channel(rng, system) for _ in range(count)])
        if source == "strong_link":
            deployments = strong_link_baseline(channels, system)
        elif source == "dynamic_programming":
            deployments = dynamic_programming_baseline(
                channels,
                system,
                config.latency_reference_seconds,
            )
        elif source == "compute_greedy":
            deployment = compute_greedy_baseline(channels, system)
            deployments = np.repeat(deployment[None, :], count, axis=0)
        else:
            deployments = []
            for channel in channels:
                if source == "coverage":
                    deployments.append(coverage_continuous_deployment(rng, system))
                elif source == "random":
                    deployments.append(random_continuous_deployment(rng, system))
                else:
                    deployments.append(
                        _tail_deployment(rng, channel, system, config.tail_candidate_pool)
                    )
            deployments = np.asarray(deployments, dtype=np.int64)
        for local_index, (channel, deployment) in enumerate(
            zip(channels, deployments, strict=True)
        ):
            action_noise = noise_seeds[noise_index]
            noise_index += 1
            actions.append(
                _make_action(
                    action_id=f"train-{source}-{local_index:04d}",
                    split="train",
                    source=source,
                    group_id=f"train-{source}-{local_index:04d}",
                    channel=channel,
                    deployment=deployment,
                    noise_seeds=action_noise,
                    system=system,
                )
            )
    if noise_index != len(noise_seeds):
        raise RuntimeError("training noise seed count does not match the action plan")
    return actions


def _build_grouped_evaluation_actions(
    *,
    split: Literal["validation", "test"],
    channel_seed: int,
    noise_seeds: np.ndarray,
    channel_count: int,
    system: SystemConfig,
    config: SurrogateDatasetConfig,
) -> list[SurrogateAction]:
    rng = np.random.default_rng(channel_seed)
    channels = np.stack([sample_channel(rng, system) for _ in range(channel_count)])
    strong = strong_link_baseline(channels, system)
    dynamic = dynamic_programming_baseline(
        channels,
        system,
        config.latency_reference_seconds,
    )
    compute = compute_greedy_baseline(channels, system)
    actions: list[SurrogateAction] = []
    for group_index, channel in enumerate(channels):
        candidates: list[tuple[str, np.ndarray]] = [
            ("dynamic_programming", dynamic[group_index]),
            ("strong_link", strong[group_index]),
            ("compute_greedy", compute),
            ("random", random_continuous_deployment(rng, system)),
            ("random", random_continuous_deployment(rng, system)),
            ("coverage", coverage_continuous_deployment(rng, system)),
            ("coverage", coverage_continuous_deployment(rng, system)),
            ("tail", _tail_deployment(rng, channel, system, config.tail_candidate_pool)),
        ]
        group_id = f"{split}-channel-{group_index:03d}"
        for candidate_index, (source, deployment) in enumerate(candidates):
            actions.append(
                _make_action(
                    action_id=f"{group_id}-{candidate_index:02d}-{source}",
                    split=split,
                    source=source,
                    group_id=group_id,
                    channel=channel,
                    deployment=deployment,
                    noise_seeds=noise_seeds,
                    system=system,
                )
            )
    return actions


def _validate_action_plan(actions: list[SurrogateAction]) -> None:
    action_ids = [action.action_id for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("action plan contains duplicate action ids")
    split_channels: dict[str, set[bytes]] = {}
    split_pairs: dict[str, set[tuple[bytes, bytes]]] = {}
    split_drops: dict[str, set[bytes]] = {}
    split_seeds: dict[str, set[int]] = {}
    for split in ("train", "validation", "test"):
        matching = [action for action in actions if action.split == split]
        split_channels[split] = {
            np.asarray(action.channel, dtype="<f4").tobytes() for action in matching
        }
        split_pairs[split] = {
            (
                np.asarray(action.channel, dtype="<f4").tobytes(),
                np.asarray(action.deployment, dtype="<i8").tobytes(),
            )
            for action in matching
        }
        split_drops[split] = {
            np.asarray(action.drop_probabilities, dtype="<f4").tobytes() for action in matching
        }
        split_seeds[split] = {seed for action in matching for seed in action.noise_seeds}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_channels[left] & split_channels[right]:
            raise ValueError(f"{left} and {right} contain an identical channel")
        if split_pairs[left] & split_pairs[right]:
            raise ValueError(f"{left} and {right} contain an identical channel/action pair")
        if split_drops[left] & split_drops[right]:
            raise ValueError(f"{left} and {right} contain an identical drop vector")
        if split_seeds[left] & split_seeds[right]:
            raise ValueError(f"{left} and {right} contain overlapping noise seeds")


def build_action_plan(
    *,
    system: SystemConfig,
    generation: DataGenerationConfig,
    config: SurrogateDatasetConfig,
    existing_ppo_cache: Path,
    existing_ppo_context: Path,
    plan_path: Path,
) -> dict[str, Any]:
    """Create or verify a deterministic action plan for resumable collection."""

    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "action_plan_schema_version": ACTION_PLAN_SCHEMA_VERSION,
        "system": system.to_dict(),
        "generation": asdict(generation),
        "dataset_config": asdict(config),
    }
    fingerprint = canonical_json_hash(metadata)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise ValueError("existing surrogate action plan has an incompatible configuration")
        _validate_imported_context_isolation(
            _import_ppo_training_actions(existing_ppo_cache, existing_ppo_context, system),
            existing["actions"],
        )
        return existing

    excluded = _load_existing_noise_seeds(existing_ppo_cache)
    train_rng = np.random.default_rng(config.training_noise_seed)
    training_seeds = _sample_seeds_excluding(
        train_rng,
        TRAIN_NOISE_SEED_RANGE,
        1024 * config.training_noise_samples,
        excluded["train"],
    ).reshape(1024, config.training_noise_samples)
    validation_seeds = _sample_seeds_excluding(
        np.random.default_rng(config.validation_noise_seed),
        VALIDATION_NOISE_SEED_RANGE,
        config.validation_noise_samples,
        excluded["validation"],
    )
    test_seeds = _sample_seeds_excluding(
        np.random.default_rng(config.test_noise_seed),
        TEST_NOISE_SEED_RANGE,
        config.test_noise_samples,
        excluded["test"],
    )

    action_rng = np.random.default_rng(config.action_seed)
    actions = _build_training_actions(action_rng, training_seeds, system, config)
    actions.extend(
        _build_grouped_evaluation_actions(
            split="validation",
            channel_seed=config.action_seed + 1,
            noise_seeds=validation_seeds,
            channel_count=config.validation_channels,
            system=system,
            config=config,
        )
    )
    actions.extend(
        _build_grouped_evaluation_actions(
            split="test",
            channel_seed=config.action_seed + 2,
            noise_seeds=test_seeds,
            channel_count=config.test_channels,
            system=system,
            config=config,
        )
    )
    _validate_action_plan(actions)
    imported = _import_ppo_training_actions(existing_ppo_cache, existing_ppo_context, system)
    _validate_imported_context_isolation(imported, [action.to_dict() for action in actions])
    payload = {
        **metadata,
        "config_fingerprint": fingerprint,
        "actions": [action.to_dict() for action in actions],
    }
    _atomic_write_json(plan_path, payload)
    return payload


def _validate_imported_context_isolation(
    imported: list[dict[str, Any]],
    planned_actions: list[dict[str, Any]],
) -> None:
    """Prove imported PPO contexts do not overlap held-out evaluation contexts."""

    evaluation = [action for action in planned_actions if action["split"] != "train"]
    imported_seeds = {
        int(seed) for row in imported for seed in np.asarray(row["noise_seeds"]).tolist()
    }
    planned_seeds = {
        int(seed)
        for action in planned_actions
        for seed in np.asarray(action["noise_seeds"]).tolist()
    }
    imported_channels = {
        np.asarray(row["channel"], dtype="<f4").tobytes() for row in imported
    }
    evaluation_channels = {
        np.asarray(row["channel"], dtype="<f4").tobytes() for row in evaluation
    }
    imported_pairs = {
        (
            np.asarray(row["channel"], dtype="<f4").tobytes(),
            np.asarray(row["deployment"], dtype="<i8").tobytes(),
        )
        for row in imported
    }
    evaluation_pairs = {
        (
            np.asarray(row["channel"], dtype="<f4").tobytes(),
            np.asarray(row["deployment"], dtype="<i8").tobytes(),
        )
        for row in evaluation
    }
    imported_drops = {
        np.asarray(row["drop_probabilities"], dtype="<f4").tobytes() for row in imported
    }
    evaluation_drops = {
        np.asarray(row["drop_probabilities"], dtype="<f4").tobytes() for row in evaluation
    }
    if imported_seeds & planned_seeds:
        raise ValueError("PPO cache and fresh action plan contain overlapping noise seeds")
    if imported_channels & evaluation_channels:
        raise ValueError("PPO training actions overlap validation/test channels")
    if imported_pairs & evaluation_pairs:
        raise ValueError("PPO training actions overlap validation/test channel/deployment pairs")
    if imported_drops & evaluation_drops:
        raise ValueError("PPO training actions overlap validation/test drop vectors")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line_number == len(lines):
                break
            raise
    return records


def _load_ppo_training_context(
    context_path: Path,
    cache_path: Path,
    system: SystemConfig,
) -> dict[bytes, tuple[np.ndarray, np.ndarray, float]]:
    """Load replay-verified PPO channels/deployments keyed by their drop vector."""

    if not context_path.exists():
        raise FileNotFoundError(f"PPO training context does not exist: {context_path}")
    metadata_path = context_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"PPO replay metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    context_sha256 = hashlib.sha256(context_path.read_bytes()).hexdigest()
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported PPO replay context metadata version")
    if metadata.get("history_exact") is not True or metadata.get("drop_set_exact") is not True:
        raise ValueError("PPO replay metadata does not prove exact reconstruction")
    if metadata.get("actions") != 1000 or metadata.get("unique_drop_vectors") != 1000:
        raise ValueError("PPO replay metadata has unexpected action counts")
    if metadata.get("context_sha256") != context_sha256:
        raise ValueError("PPO replay context SHA256 does not match its metadata")
    if metadata.get("ppl_cache_sha256") != cache_sha256:
        raise ValueError("PPO cache SHA256 does not match the replay metadata")
    with np.load(context_path) as data:
        channels = np.asarray(data["channels"], dtype=np.float32)
        deployments = np.asarray(data["deployments"], dtype=np.int64)
        probabilities = np.asarray(data["drop_probabilities"], dtype=np.float32)
    if channels.shape != (1000, system.num_uavs, system.num_uavs):
        raise ValueError(f"unexpected PPO context channel shape: {channels.shape}")
    if deployments.shape != (1000, system.num_layers):
        raise ValueError(f"unexpected PPO context deployment shape: {deployments.shape}")
    if probabilities.shape != (1000, system.num_layers - 1):
        raise ValueError(f"unexpected PPO context drop shape: {probabilities.shape}")
    context: dict[bytes, tuple[np.ndarray, np.ndarray, float]] = {}
    for channel, deployment, drops in zip(
        channels, deployments, probabilities, strict=True
    ):
        recomputed = boundary_drop_probabilities(deployment, channel, system)
        if not np.array_equal(recomputed, drops):
            raise ValueError("PPO replay context contains an inconsistent drop vector")
        key = drops.astype("<f4", copy=False).tobytes()
        if key in context:
            raise ValueError("PPO replay context contains duplicate drop vectors")
        latency = collaborative_latency(deployment, channel, system).total_seconds
        context[key] = (channel, deployment, latency)
    return context


def collect_surrogate_samples(
    *,
    plan: dict[str, Any],
    evaluator: TruePPLQualityEvaluator,
    sample_cache_path: Path,
    progress_interval: int = 50,
) -> dict[str, int]:
    """Evaluate every planned action/seed pair with durable per-seed progress."""

    metadata_path = sample_cache_path.with_suffix(sample_cache_path.suffix + ".meta.json")
    expected_metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "config_fingerprint": plan["config_fingerprint"],
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != expected_metadata:
            raise ValueError("surrogate sample cache metadata is incompatible")
    else:
        _atomic_write_json(metadata_path, expected_metadata)

    records = _load_jsonl(sample_cache_path)
    completed = {
        (str(record["action_id"]), int(record["noise_seed"])) for record in records
    }
    expected_keys: set[tuple[str, int]] = set()
    for action in plan["actions"]:
        for noise_seed in action["noise_seeds"]:
            key = (str(action["action_id"]), int(noise_seed))
            if key in expected_keys:
                raise ValueError(f"duplicate planned sample key: {key}")
            expected_keys.add(key)
    if unknown := completed - expected_keys:
        raise ValueError(f"sample cache contains unknown action/seed keys: {sorted(unknown)[:3]}")

    started_at = time.perf_counter()
    newly_completed = 0
    for action in plan["actions"]:
        probabilities = np.asarray(action["drop_probabilities"], dtype=np.float32)
        for noise_seed in action["noise_seeds"]:
            key = (str(action["action_id"]), int(noise_seed))
            if key in completed:
                continue
            before_forwards = evaluator.model_forwards
            sample_started = time.perf_counter()
            quality = float(
                evaluator.evaluate(
                    probabilities[None, :],
                    noise_seeds=np.asarray([noise_seed], dtype=np.int64),
                )[0]
            )
            duration = time.perf_counter() - sample_started
            cache_hit = evaluator.model_forwards == before_forwards
            record = {
                "format_version": DATASET_FORMAT_VERSION,
                "config_fingerprint": plan["config_fingerprint"],
                "action_id": action["action_id"],
                "split": action["split"],
                "source": action["source"],
                "group_id": action["group_id"],
                "channel": action["channel"],
                "deployment": action["deployment"],
                "drop_probabilities": action["drop_probabilities"],
                "latency_seconds": action["latency_seconds"],
                "noise_seed": int(noise_seed),
                "perplexity": evaluator.clean_perplexity * math.exp(quality),
                "log_ppl_ratio": quality,
                "evaluation_seconds": duration,
                "ppl_cache_hit": cache_hit,
            }
            _append_jsonl(sample_cache_path, record)
            completed.add(key)
            newly_completed += 1
            if progress_interval and newly_completed % progress_interval == 0:
                elapsed = time.perf_counter() - started_at
                remaining = len(expected_keys) - len(completed)
                rate = newly_completed / max(elapsed, 1e-9)
                eta = remaining / max(rate, 1e-9)
                print(
                    f"surrogate_samples={len(completed)}/{len(expected_keys)} "
                    f"new={newly_completed} eta_seconds={eta:.1f}",
                    flush=True,
                )
    return {
        "expected_samples": len(expected_keys),
        "completed_samples": len(completed),
        "new_samples": newly_completed,
    }


def _import_ppo_training_actions(
    cache_path: Path,
    context_path: Path,
    system: SystemConfig,
) -> list[dict[str, Any]]:
    """Aggregate only the current PPO cache's training-range multi-seed labels."""

    context = _load_ppo_training_context(context_path, cache_path, system)
    grouped: dict[bytes, dict[str, Any]] = {}
    for record in _load_jsonl(cache_path):
        noise_seed = int(record["noise_seed"])
        if noise_seed not in TRAIN_NOISE_SEED_RANGE:
            continue
        probabilities = np.asarray(record["drop_probabilities"], dtype=np.float32)
        key = probabilities.astype("<f4", copy=False).tobytes()
        entry = grouped.setdefault(
            key,
            {
                "drop_probabilities": probabilities,
                "noise_seeds": [],
                "qualities": [],
            },
        )
        entry["noise_seeds"].append(noise_seed)
        entry["qualities"].append(float(record["log_ppl_ratio"]))
    imported: list[dict[str, Any]] = []
    for index, entry in enumerate(grouped.values()):
        if len(entry["noise_seeds"]) != 4:
            raise ValueError("every imported PPO training action must contain exactly four seeds")
        key = np.asarray(entry["drop_probabilities"], dtype="<f4").tobytes()
        if key not in context:
            raise ValueError("PPO cache action is missing replay-verified channel/deployment context")
        channel, deployment, latency = context[key]
        imported.append(
            {
                "action_id": f"train-ppo-cache-{index:04d}",
                "split": "train",
                "source": "ppo_cache",
                "group_id": f"train-ppo-cache-{index:04d}",
                "channel": channel,
                "deployment": deployment,
                "drop_probabilities": entry["drop_probabilities"],
                "latency_seconds": latency,
                "noise_seeds": np.asarray(entry["noise_seeds"], dtype=np.int64),
                "qualities": np.asarray(entry["qualities"], dtype=np.float32),
                "has_context": True,
            }
        )
    if len(imported) != 1000:
        raise ValueError(f"expected 1000 PPO training actions, found {len(imported)}")
    if len(context) != len(imported):
        raise ValueError("PPO replay context contains actions absent from the training cache")
    return imported


def _aggregate_fresh_actions(
    plan: dict[str, Any],
    sample_records: list[dict[str, Any]],
) -> dict[SplitName, list[dict[str, Any]]]:
    records_by_action: dict[str, list[dict[str, Any]]] = {}
    for record in sample_records:
        records_by_action.setdefault(str(record["action_id"]), []).append(record)
    aggregated: dict[SplitName, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for action in plan["actions"]:
        action_records = records_by_action.get(str(action["action_id"]), [])
        expected_seeds = {int(seed) for seed in action["noise_seeds"]}
        observed_seeds = {int(record["noise_seed"]) for record in action_records}
        if observed_seeds != expected_seeds:
            raise RuntimeError(f"action {action['action_id']} has incomplete seed results")
        ordered = sorted(action_records, key=lambda record: int(record["noise_seed"]))
        aggregated[action["split"]].append(
            {
                **action,
                "channel": np.asarray(action["channel"], dtype=np.float32),
                "deployment": np.asarray(action["deployment"], dtype=np.int64),
                "drop_probabilities": np.asarray(
                    action["drop_probabilities"], dtype=np.float32
                ),
                "noise_seeds": np.asarray(
                    [record["noise_seed"] for record in ordered], dtype=np.int64
                ),
                "qualities": np.asarray(
                    [record["log_ppl_ratio"] for record in ordered], dtype=np.float32
                ),
                "has_context": True,
            }
        )
    return aggregated


def _validate_aggregated_splits(rows: dict[SplitName, list[dict[str, Any]]]) -> None:
    drop_keys: dict[str, set[bytes]] = {}
    seed_sets: dict[str, set[int]] = {}
    context_pairs: dict[str, set[tuple[bytes, bytes]]] = {}
    for split, split_rows in rows.items():
        drop_keys[split] = {
            np.asarray(row["drop_probabilities"], dtype="<f4").tobytes()
            for row in split_rows
        }
        seed_sets[split] = {
            int(seed) for row in split_rows for seed in np.asarray(row["noise_seeds"])
        }
        context_pairs[split] = {
            (
                np.asarray(row["channel"], dtype="<f4").tobytes(),
                np.asarray(row["deployment"], dtype="<i8").tobytes(),
            )
            for row in split_rows
            if bool(row["has_context"])
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if drop_keys[left] & drop_keys[right]:
            raise ValueError(f"aggregated {left}/{right} splits share a drop vector")
        if seed_sets[left] & seed_sets[right]:
            raise ValueError(f"aggregated {left}/{right} splits share a noise seed")
        if context_pairs[left] & context_pairs[right]:
            raise ValueError(f"aggregated {left}/{right} splits share a channel/action pair")


def _split_isolation_audit(
    rows: dict[SplitName, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Report leakage-critical context overlap and structural deployment reuse."""

    fields: dict[str, dict[str, set[Any]]] = {}
    for split, split_rows in rows.items():
        fields[split] = {
            "noise_seed": {
                int(seed) for row in split_rows for seed in np.asarray(row["noise_seeds"])
            },
            "channel": {
                np.asarray(row["channel"], dtype="<f4").tobytes() for row in split_rows
            },
            "deployment": {
                np.asarray(row["deployment"], dtype="<i8").tobytes() for row in split_rows
            },
            "channel_deployment_pair": {
                (
                    np.asarray(row["channel"], dtype="<f4").tobytes(),
                    np.asarray(row["deployment"], dtype="<i8").tobytes(),
                )
                for row in split_rows
            },
            "drop_vector": {
                np.asarray(row["drop_probabilities"], dtype="<f4").tobytes()
                for row in split_rows
            },
        }
    pairwise: dict[str, dict[str, int]] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        pairwise[f"{left}_vs_{right}"] = {
            name: len(fields[left][name] & fields[right][name])
            for name in fields[left]
        }
    leakage_fields = ("noise_seed", "channel", "channel_deployment_pair", "drop_vector")
    return {
        "pairwise_overlap_counts": pairwise,
        "leakage_fields": list(leakage_fields),
        "passed": all(
            values[field] == 0
            for values in pairwise.values()
            for field in leakage_fields
        ),
        "deployment_vector_note": (
            "A deployment vector can intentionally repeat on independent channels, especially "
            "for channel-independent compute-greedy. Leakage is therefore gated on the complete "
            "channel/deployment pair; raw deployment overlap remains reported for transparency."
        ),
    }


def _write_split_dataset(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"cannot write an empty surrogate split: {path}")
    seed_counts = {len(np.asarray(row["noise_seeds"])) for row in rows}
    if len(seed_counts) != 1:
        raise ValueError("all actions in one split must use the same number of noise seeds")
    qualities = [np.asarray(row["qualities"], dtype=np.float32) for row in rows]
    payload = {
        "action_ids": np.asarray([row["action_id"] for row in rows]),
        "sample_source": np.asarray([row["source"] for row in rows]),
        "group_ids": np.asarray([row["group_id"] for row in rows]),
        "channels": np.stack([np.asarray(row["channel"], dtype=np.float32) for row in rows]),
        "deployments": np.stack(
            [np.asarray(row["deployment"], dtype=np.int64) for row in rows]
        ),
        "drop_probabilities": np.stack(
            [np.asarray(row["drop_probabilities"], dtype=np.float32) for row in rows]
        ),
        "latency_seconds": np.asarray(
            [row["latency_seconds"] for row in rows], dtype=np.float32
        ),
        "noise_seeds": np.stack(
            [np.asarray(row["noise_seeds"], dtype=np.int64) for row in rows]
        ),
        "log_ppl_ratio": np.asarray(
            [values.mean(dtype=np.float64) for values in qualities], dtype=np.float32
        ),
        "log_ppl_ratio_std": np.asarray(
            [values.std(dtype=np.float64) for values in qualities], dtype=np.float32
        ),
        "noise_seed_count": np.asarray(
            [len(values) for values in qualities], dtype=np.int16
        ),
        "has_context": np.asarray([row["has_context"] for row in rows], dtype=np.bool_),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    source_counts: dict[str, int] = {}
    for source in payload["sample_source"].tolist():
        source_counts[str(source)] = source_counts.get(str(source), 0) + 1
    return {
        "path": str(path),
        "actions": len(rows),
        "noise_seeds_per_action": next(iter(seed_counts)),
        "source_counts": source_counts,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def aggregate_surrogate_datasets(
    *,
    plan: dict[str, Any],
    sample_cache_path: Path,
    existing_ppo_cache: Path,
    existing_ppo_context: Path,
    system: SystemConfig,
    output_directory: Path,
    quality_evaluator_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine fresh samples and isolated PPO training-cache actions into NPZ splits."""

    fresh = _aggregate_fresh_actions(plan, _load_jsonl(sample_cache_path))
    rows: dict[SplitName, list[dict[str, Any]]] = {
        "train": _import_ppo_training_actions(
            existing_ppo_cache, existing_ppo_context, system
        )
        + fresh["train"],
        "validation": fresh["validation"],
        "test": fresh["test"],
    }
    expected_counts = {"train": 2024, "validation": 128, "test": 128}
    actual_counts = {split: len(split_rows) for split, split_rows in rows.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(f"unexpected aggregate counts: {actual_counts}")
    _validate_aggregated_splits(rows)
    isolation_audit = _split_isolation_audit(rows)
    if not isolation_audit["passed"]:
        raise ValueError("surrogate split isolation audit failed")

    split_metadata = {}
    for split in ("train", "validation", "test"):
        split_metadata[split] = _write_split_dataset(
            output_directory / f"codellama_surrogate_multiseed_v2_{split}.npz",
            rows[split],
        )
    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "config_fingerprint": plan["config_fingerprint"],
        "generation": plan["generation"],
        "system": plan["system"],
        "dataset_config": plan["dataset_config"],
        "quality_evaluator": quality_evaluator_metadata,
        "ppo_training_context": {
            "path": str(existing_ppo_context),
            "sha256": hashlib.sha256(existing_ppo_context.read_bytes()).hexdigest(),
            "replay_metadata_path": str(existing_ppo_context.with_suffix(".json")),
            "replay_metadata_sha256": hashlib.sha256(
                existing_ppo_context.with_suffix(".json").read_bytes()
            ).hexdigest(),
            "replay_verified_actions": 1000,
        },
        "isolation_audit": isolation_audit,
        "splits": split_metadata,
    }
    manifest["dataset_fingerprint"] = canonical_json_hash(manifest)
    _atomic_write_json(
        output_directory / "codellama_surrogate_multiseed_v2_manifest.json",
        manifest,
    )
    return manifest


def collect_and_aggregate_surrogate_dataset(
    *,
    generation: DataGenerationConfig,
    system: SystemConfig,
    config: SurrogateDatasetConfig,
    existing_ppo_cache: Path,
    existing_ppo_context: Path,
    plan_path: Path,
    sample_cache_path: Path,
    ppl_cache_path: Path,
    output_directory: Path,
    device_name: str,
    progress_interval: int = 50,
) -> dict[str, Any]:
    """Run the complete resumable real-PPL collection and aggregation workflow."""

    plan = build_action_plan(
        system=system,
        generation=generation,
        config=config,
        existing_ppo_cache=existing_ppo_cache,
        existing_ppo_context=existing_ppo_context,
        plan_path=plan_path,
    )
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=device_name,
        cache_path=ppl_cache_path,
        progress_interval=progress_interval,
    )
    collection = collect_surrogate_samples(
        plan=plan,
        evaluator=evaluator,
        sample_cache_path=sample_cache_path,
        progress_interval=progress_interval,
    )
    manifest = aggregate_surrogate_datasets(
        plan=plan,
        sample_cache_path=sample_cache_path,
        existing_ppo_cache=existing_ppo_cache,
        existing_ppo_context=existing_ppo_context,
        system=system,
        output_directory=output_directory,
        quality_evaluator_metadata=evaluator.metadata(),
    )
    return {
        "collection": collection,
        "quality_evaluator": evaluator.metadata(),
        "dataset_manifest": manifest,
    }
