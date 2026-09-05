"""Direct CodeLlama quality evaluation for surrogate-free policy training."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_rl.config import DataGenerationConfig
from uav_rl.models import activation_dropout


def _find_decoder_layers(model: Any) -> Any:
    """查找文本 decoder 的 layer 列表，兼容 Llama 和 Qwen3.5 包装结构。"""
    candidates = (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "transformer.h",
    )
    for path in candidates:
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue
        if hasattr(current, "__len__") and len(current) > 1:
            return current
    raise AttributeError(
        "无法找到 decoder layers；已尝试 model.layers、"
        "model.language_model.layers、language_model.layers 和 transformer.h"
    )


class TruePPLQualityEvaluator:
    """Evaluate deterministic activation-corruption PPL with a resident LLM.

    Each distinct (drop-probability vector, noise seed) pair is evaluated once. An
    optional JSONL cache persists results so completed LLM forwards survive interruption.
    """

    def __init__(
        self,
        generation: DataGenerationConfig,
        *,
        device_name: str = "cuda",
        cache_path: Path | None = None,
        progress_interval: int = 10,
        device_map: str | dict[str, int | str] | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from uav_rl.data.ppl_dataset import prepare_corpus, torch_dtype
        from uav_rl.metrics import compute_perplexity

        if progress_interval < 0:
            raise ValueError("progress_interval cannot be negative")
        self.generation = generation
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
        self.cache_path = cache_path
        self.progress_interval = progress_interval
        self.device_map = device_map
        self._compute_perplexity = compute_perplexity
        self._validate_cache_metadata(cache_path, generation)
        self._cache = self._load_cache(cache_path)
        self.model_forwards = 0
        self.cache_hits = 0

        print(f"loading_true_ppl_model={generation.model_id} device={self.device}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(generation.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 多卡模式下由 Accelerate 将模型分布到可见 GPU；此时不能再次调用
        # model.to(cuda)，否则会尝试把整个大模型移动到单张 GPU。
        model_kwargs: dict[str, Any] = {
            "dtype": torch_dtype(generation.dtype),
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            generation.model_id,
            **model_kwargs,
        )
        self.model.eval()
        print("true_ppl_model_loaded=true", flush=True)
        self.encoded = prepare_corpus(generation, tokenizer)
        print("true_ppl_corpus_loaded=true", flush=True)
        clean = self._compute_perplexity(
            self.model,
            self.encoded["input_ids"],
            self.encoded["attention_mask"],
            batch_size=generation.batch_size,
            device=self._input_device(),
        )
        self.clean_perplexity = clean.perplexity
        print(f"clean_perplexity={self.clean_perplexity:.6f}", flush=True)
        self.evaluated_sequences = clean.evaluated_sequences
        self.evaluated_tokens = clean.evaluated_tokens
        # 记录实际文本 decoder 的层数；Qwen3.5 是多模态包装模型，
        # 其文本层可能位于 model.language_model.layers。
        self.decoder_layers = _find_decoder_layers(self.model)
        self.num_boundaries = len(self.decoder_layers) - 1

    @staticmethod
    def _key(probabilities: np.ndarray, noise_seed: int) -> tuple[bytes, int]:
        return np.asarray(probabilities, dtype="<f4").tobytes(), noise_seed

    @classmethod
    def _load_cache(cls, path: Path | None) -> dict[tuple[bytes, int], float]:
        cache: dict[tuple[bytes, int], float] = {}
        if path is None or not path.exists():
            return cache
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                probabilities = np.asarray(record["drop_probabilities"], dtype=np.float32)
                noise_seed = int(record["noise_seed"])
                cache[cls._key(probabilities, noise_seed)] = float(record["log_ppl_ratio"])
            except json.JSONDecodeError as error:
                if line_number == len(lines):
                    break
                raise ValueError(f"invalid true-PPL cache record at line {line_number}") from error
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid true-PPL cache record at line {line_number}") from error
        return cache

    @staticmethod
    def _validate_cache_metadata(
        path: Path | None,
        generation: DataGenerationConfig,
    ) -> None:
        if path is None:
            return
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        expected = {"format_version": 2, "generation": asdict(generation)}
        if metadata_path.exists():
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError("true-PPL cache metadata does not match the requested corpus")
            return
        if path.exists() and path.stat().st_size:
            raise ValueError("non-empty true-PPL cache has no metadata file")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(metadata_path)

    def _append_cache(
        self,
        probabilities: np.ndarray,
        noise_seed: int,
        perplexity: float,
        quality: float,
    ) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "drop_probabilities": probabilities.tolist(),
            "noise_seed": noise_seed,
            "perplexity": perplexity,
            "log_ppl_ratio": quality,
        }
        with self.cache_path.open("a", encoding="utf-8") as cache_file:
            cache_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            cache_file.flush()
            os.fsync(cache_file.fileno())

    def _input_device(self) -> torch.device:
        """返回模型 embedding 所在的输入设备，兼容 device_map 多卡加载。"""
        if self.device_map is not None:
            return self.model.get_input_embeddings().weight.device
        return self.device

    def _cuda_rng_devices(self) -> list[int]:
        """返回当前进程可见的 CUDA 设备，保证多卡 noise seed 可复现。"""
        if self.device.type != "cuda":
            return []
        return list(range(torch.cuda.device_count()))

    def _release_unused_cuda_memory(self) -> None:
        """Return inactive allocator blocks between independent PPL forwards.

        A true-policy validation can run thousands of independent, dropout-hooked
        model forwards in one process. Releasing only inactive blocks avoids
        allocator fragmentation without altering the model, cache, or seeded PPL
        calculation.
        """

        if self.device.type == "cuda":
            for gpu_index in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()

    def _evaluate_one(self, probabilities: np.ndarray, noise_seed: int) -> float:
        key = self._key(probabilities, noise_seed)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        active_probabilities = {
            layer: float(probability)
            for layer, probability in enumerate(probabilities)
            if probability > 0.0
        }
        # 多卡模型的 activation 可能位于多张 GPU；所有可见 CUDA 设备都
        # 纳入 fork_rng，避免同一个 noise seed 在不同设备上产生不一致结果。
        cuda_devices = self._cuda_rng_devices()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(noise_seed)
            with activation_dropout(self.model, active_probabilities):
                result = self._compute_perplexity(
                    self.model,
                    self.encoded["input_ids"],
                    self.encoded["attention_mask"],
                    batch_size=self.generation.batch_size,
                    device=self._input_device(),
                )
        self._release_unused_cuda_memory()
        if not math.isfinite(result.perplexity):
            raise FloatingPointError("non-finite PPL in direct quality evaluation")
        quality = math.log(result.perplexity / self.clean_perplexity)
        self._cache[key] = quality
        self._append_cache(probabilities, noise_seed, result.perplexity, quality)
        self.model_forwards += 1
        if self.progress_interval and self.model_forwards % self.progress_interval == 0:
            print(
                f"true_ppl_forwards={self.model_forwards} cache_hits={self.cache_hits} "
                f"latest_ppl={result.perplexity:.4f}",
                flush=True,
            )
        return quality

    def evaluate(
        self,
        drop_probabilities: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        probabilities = np.asarray(drop_probabilities, dtype=np.float32)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_boundaries:
            raise ValueError(f"drop_probabilities must have shape (N, {self.num_boundaries})")
        if noise_seeds is None:
            seeds = np.full((len(probabilities), 1), self.generation.noise_seed, dtype=np.int64)
        else:
            raw_seeds = np.asarray(noise_seeds)
            if not np.issubdtype(raw_seeds.dtype, np.integer):
                raise ValueError("noise_seeds must contain integers")
            if raw_seeds.ndim == 1:
                seeds = np.broadcast_to(raw_seeds, (len(probabilities), len(raw_seeds)))
            elif raw_seeds.ndim == 2 and raw_seeds.shape[0] == len(probabilities):
                seeds = raw_seeds
            else:
                raise ValueError("noise_seeds must have shape (K,) or (N, K)")
            if seeds.shape[1] < 1:
                raise ValueError("noise_seeds must contain at least one seed per action")
            seeds = seeds.astype(np.int64, copy=False)
        return np.asarray(
            [
                np.mean(
                    [self._evaluate_one(row, int(noise_seed)) for noise_seed in row_seeds],
                    dtype=np.float64,
                )
                for row, row_seeds in zip(probabilities, seeds, strict=True)
            ],
            dtype=np.float32,
        )

    def metadata(self) -> dict[str, int | float | str]:
        """Return serializable corpus and cache statistics for experiment reports."""

        return {
            "model_id": self.generation.model_id,
            "clean_perplexity": self.clean_perplexity,
            "evaluated_sequences": self.evaluated_sequences,
            "evaluated_tokens": self.evaluated_tokens,
            "model_forwards": self.model_forwards,
            "cache_hits": self.cache_hits,
            "cached_entries": len(self._cache),
        }
