"""Replay cached PPO training to recover and verify its exact action contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from uav_rl.config import PPOConfig, SystemConfig
from uav_rl.experiment import estimate_latency_reference
from uav_rl.rl import DeploymentEnvironment, PPOTrainer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedQualityEvaluator:
    """Serve exact multi-seed labels from an existing true-PPL JSONL cache."""

    def __init__(self, cache_path: Path) -> None:
        self.lookup: dict[tuple[bytes, int], float] = {}
        self.training_drop_keys: set[bytes] = set()
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            drops = np.asarray(record["drop_probabilities"], dtype="<f4")
            seed = int(record["noise_seed"])
            key = drops.tobytes()
            self.lookup[(key, seed)] = float(record["log_ppl_ratio"])
            if seed < 1_000_000_000:
                self.training_drop_keys.add(key)

    def evaluate(
        self,
        drop_probabilities: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        if noise_seeds is None:
            raise ValueError("replay requires explicit noise seeds")
        probabilities = np.asarray(drop_probabilities, dtype=np.float32)
        raw_seeds = np.asarray(noise_seeds, dtype=np.int64)
        seeds = (
            np.broadcast_to(raw_seeds, (len(probabilities), len(raw_seeds)))
            if raw_seeds.ndim == 1
            else raw_seeds
        )
        output = []
        for drops, row_seeds in zip(probabilities, seeds, strict=True):
            drop_key = drops.astype("<f4", copy=False).tobytes()
            try:
                values = [self.lookup[(drop_key, int(seed))] for seed in row_seeds]
            except KeyError as error:
                raise ValueError(f"PPO replay diverged from the PPL cache: {error}") from error
            output.append(np.mean(values, dtype=np.float64))
        return np.asarray(output, dtype=np.float32)


class RecordingEnvironment(DeploymentEnvironment):
    """Record only training calls, identified by per-action 2-D seed arrays."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.training_channels: list[np.ndarray] = []
        self.training_deployments: list[np.ndarray] = []
        self.training_drops: list[np.ndarray] = []

    def evaluate(
        self,
        channels: np.ndarray,
        deployments: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        result = super().evaluate(channels, deployments, noise_seeds=noise_seeds)
        if noise_seeds is not None and np.asarray(noise_seeds).ndim == 2:
            self.training_channels.extend(np.asarray(channels, dtype=np.float32))
            self.training_deployments.extend(np.asarray(deployments, dtype=np.int64))
            self.training_drops.extend(
                np.asarray(result[1]["drop_probabilities"], dtype=np.float32)
            )
        return result


def reconstruct_context(cache_path: Path, state_path: Path, output_path: Path) -> dict[str, object]:
    """Require exact history/drop equality before publishing recovered contexts."""

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    raw_config = dict(state["ppo_config"])
    raw_config["system"] = SystemConfig(**raw_config["system"])
    config = PPOConfig(**raw_config)
    evaluator = CachedQualityEvaluator(cache_path)
    environment = RecordingEnvironment(
        config.system,
        evaluator,
        estimate_latency_reference(config.system, config.seed),
    )
    trainer = PPOTrainer(
        config,
        environment,
        teacher_quality_model=None,
        require_ppo_checkpoint=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary:
        temp = Path(temporary)
        history = trainer.train(temp / "best.pth", state_path=temp / "state.pth")
        channels = np.stack(environment.training_channels)
        deployments = np.stack(environment.training_deployments)
        drops = np.stack(environment.training_drops)
        replayed_keys = {
            row.astype("<f4", copy=False).tobytes() for row in drops
        }
        history_exact = history == state["history"]
        drops_exact = replayed_keys == evaluator.training_drop_keys
        if len(drops) != 1000 or len(replayed_keys) != 1000 or not history_exact or not drops_exact:
            raise RuntimeError("PPO replay failed exact history/drop-set verification")
        temporary_npz = temp / "context.npz"
        np.savez_compressed(
            temporary_npz,
            channels=channels,
            deployments=deployments,
            drop_probabilities=drops,
        )
        temporary_npz.replace(output_path)
    metadata = {
        "format_version": 1,
        "actions": 1000,
        "unique_drop_vectors": 1000,
        "history_exact": history_exact,
        "drop_set_exact": drops_exact,
        "ppl_cache": str(cache_path),
        "ppl_cache_sha256": _sha256(cache_path),
        "ppo_state": str(state_path),
        "ppo_state_sha256": _sha256(state_path),
        "context": str(output_path),
        "context_sha256": _sha256(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/cache/ppo_true_ppl_multiseed.jsonl"),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("artifacts/models/ppo_true_ppl_multiseed_state.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/ppo_true_ppl_multiseed_training_context.npz"),
    )
    args = parser.parse_args()
    print(json.dumps(reconstruct_context(args.cache, args.state, args.output), indent=2))


if __name__ == "__main__":
    main()
