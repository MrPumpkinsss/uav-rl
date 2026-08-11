"""Generate deterministic activation-corruption PPL labels with CodeLlama."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.deployment import coverage_continuous_deployment, random_continuous_deployment
from uav_rl.metrics import compute_perplexity
from uav_rl.models import activation_dropout
from uav_rl.wireless import boundary_drop_probabilities, sample_channel


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _prepare_corpus(config: DataGenerationConfig, tokenizer: Any) -> dict[str, torch.Tensor]:
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=f"{config.dataset_split}[:{config.text_sample_limit}]",
    )
    texts = [text for text in dataset["text"] if text.strip()]
    return tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )


def generate_ppl_dataset(
    output_path: Path,
    generation: DataGenerationConfig,
    system: SystemConfig,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Generate and persist true CodeLlama PPL for random valid deployments."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    rng = np.random.default_rng(generation.seed)

    tokenizer = AutoTokenizer.from_pretrained(generation.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        generation.model_id,
        dtype=_torch_dtype(generation.dtype),
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    encoded = _prepare_corpus(generation, tokenizer)

    torch.manual_seed(generation.noise_seed)
    clean = compute_perplexity(
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
        batch_size=generation.batch_size,
        device=device,
    )

    channels = np.empty(
        (generation.num_samples, system.num_uavs, system.num_uavs), dtype=np.float32
    )
    deployments = np.empty((generation.num_samples, system.num_layers), dtype=np.int64)
    drop_probabilities = np.empty((generation.num_samples, system.num_layers - 1), dtype=np.float32)
    perplexities = np.empty(generation.num_samples, dtype=np.float32)
    durations = np.empty(generation.num_samples, dtype=np.float32)

    started_at = time.perf_counter()
    for index in range(generation.num_samples):
        channel = sample_channel(rng, system)
        deployment = random_continuous_deployment(rng, system)
        probabilities = boundary_drop_probabilities(deployment, channel, system)
        active_probabilities = {
            layer: float(probability)
            for layer, probability in enumerate(probabilities)
            if probability > 0.0
        }

        torch.manual_seed(generation.noise_seed)
        sample_started = time.perf_counter()
        with activation_dropout(model, active_probabilities):
            result = compute_perplexity(
                model,
                encoded["input_ids"],
                encoded["attention_mask"],
                batch_size=generation.batch_size,
                device=device,
            )
        if not math.isfinite(result.perplexity):
            raise FloatingPointError(f"non-finite noisy perplexity for generated sample {index}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        channels[index] = channel
        deployments[index] = deployment
        drop_probabilities[index] = probabilities
        perplexities[index] = result.perplexity
        durations[index] = time.perf_counter() - sample_started

        completed = index + 1
        if completed == 1 or completed % 10 == 0 or completed == generation.num_samples:
            elapsed = time.perf_counter() - started_at
            remaining = elapsed / completed * (generation.num_samples - completed)
            print(
                f"[{completed:4d}/{generation.num_samples}] "
                f"PPL={result.perplexity:.4f} elapsed={elapsed:.1f}s ETA={remaining:.1f}s",
                flush=True,
            )

    log_ppl_ratio = np.log(perplexities / clean.perplexity).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        channels=channels,
        deployments=deployments,
        drop_probabilities=drop_probabilities,
        perplexities=perplexities,
        log_ppl_ratio=log_ppl_ratio,
        evaluation_seconds=durations,
    )

    metadata: dict[str, Any] = {
        "generation": asdict(generation),
        "system": system.to_dict(),
        "clean_perplexity": clean.perplexity,
        "evaluated_tokens": clean.evaluated_tokens,
        "evaluated_sequences": clean.evaluated_sequences,
        "mean_evaluation_seconds": float(durations.mean()),
        "total_generation_seconds": time.perf_counter() - started_at,
        "ppl_min": float(perplexities.min()),
        "ppl_max": float(perplexities.max()),
        "ppl_mean": float(perplexities.mean()),
        "log_ratio_mean": float(log_ppl_ratio.mean()),
        "log_ratio_std": float(log_ppl_ratio.std()),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def append_tail_ppl_samples(
    dataset_path: Path,
    output_path: Path,
    generation: DataGenerationConfig,
    system: SystemConfig,
    *,
    candidate_pool_size: int,
    selection_mode: str = "tail",
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Append true-PPL samples selected from the high packet-loss tail.

    Candidates follow the original channel and deployment distributions. Selection is
    analytical, so CodeLlama is evaluated only for the requested tail samples.
    """

    if candidate_pool_size < generation.num_samples:
        raise ValueError("candidate pool must be at least as large as the append count")
    existing = np.load(dataset_path)
    required_keys = {
        "channels",
        "deployments",
        "drop_probabilities",
        "perplexities",
        "log_ppl_ratio",
        "evaluation_seconds",
    }
    if missing := required_keys.difference(existing.files):
        raise ValueError(f"dataset is missing required arrays: {sorted(missing)}")

    rng = np.random.default_rng(generation.seed)
    candidate_channels = np.empty(
        (candidate_pool_size, system.num_uavs, system.num_uavs), dtype=np.float32
    )
    candidate_deployments = np.empty((candidate_pool_size, system.num_layers), dtype=np.int64)
    candidate_probabilities = np.empty(
        (candidate_pool_size, system.num_layers - 1), dtype=np.float32
    )
    scores = np.empty(candidate_pool_size, dtype=np.float32)
    for index in range(candidate_pool_size):
        channel = sample_channel(rng, system)
        deployment = (
            coverage_continuous_deployment(rng, system)
            if selection_mode == "coverage"
            else random_continuous_deployment(rng, system)
        )
        probabilities = boundary_drop_probabilities(deployment, channel, system)
        candidate_channels[index] = channel
        candidate_deployments[index] = deployment
        candidate_probabilities[index] = probabilities
        # Independent boundary survival probabilities multiply, so their negative
        # log-survival is a natural analytical tail score.
        scores[index] = float(-np.log1p(-probabilities).sum())

    if selection_mode == "tail":
        selected = np.argsort(scores, kind="stable")[-generation.num_samples :][::-1]
        source_id = 1
    elif selection_mode == "coverage":
        if candidate_pool_size != generation.num_samples:
            raise ValueError("coverage mode requires one candidate per appended sample")
        selected = np.arange(generation.num_samples)
        source_id = 2
    else:
        raise ValueError(f"unknown selection mode: {selection_mode}")
    channels = candidate_channels[selected]
    deployments = candidate_deployments[selected]
    drop_probabilities = candidate_probabilities[selected]
    selected_scores = scores[selected]

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(generation.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        generation.model_id,
        dtype=_torch_dtype(generation.dtype),
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    encoded = _prepare_corpus(generation, tokenizer)
    clean = compute_perplexity(
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
        batch_size=generation.batch_size,
        device=device,
    )

    perplexities = np.empty(generation.num_samples, dtype=np.float32)
    durations = np.empty(generation.num_samples, dtype=np.float32)
    started_at = time.perf_counter()
    for index, probabilities in enumerate(drop_probabilities):
        active_probabilities = {
            layer: float(probability)
            for layer, probability in enumerate(probabilities)
            if probability > 0.0
        }
        torch.manual_seed(generation.noise_seed)
        sample_started = time.perf_counter()
        with activation_dropout(model, active_probabilities):
            result = compute_perplexity(
                model,
                encoded["input_ids"],
                encoded["attention_mask"],
                batch_size=generation.batch_size,
                device=device,
            )
        if not math.isfinite(result.perplexity):
            raise FloatingPointError(f"non-finite tail PPL for appended sample {index}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        perplexities[index] = result.perplexity
        durations[index] = time.perf_counter() - sample_started
        completed = index + 1
        if completed == 1 or completed % 10 == 0 or completed == generation.num_samples:
            print(
                f"[{completed:4d}/{generation.num_samples}] "
                f"tail_score={selected_scores[index]:.4f} PPL={result.perplexity:.4f}",
                flush=True,
            )

    log_ppl_ratio = np.log(perplexities / clean.perplexity).astype(np.float32)
    old_count = len(existing["perplexities"])
    old_source = (
        existing["sample_source"]
        if "sample_source" in existing.files
        else np.zeros(old_count, dtype=np.int8)
    )
    old_scores = (
        existing["tail_selection_score"]
        if "tail_selection_score" in existing.files
        else np.full(old_count, np.nan, dtype=np.float32)
    )
    combined: dict[str, np.ndarray] = {
        "channels": np.concatenate([existing["channels"], channels]),
        "deployments": np.concatenate([existing["deployments"], deployments]),
        "drop_probabilities": np.concatenate([existing["drop_probabilities"], drop_probabilities]),
        "perplexities": np.concatenate([existing["perplexities"], perplexities]),
        "log_ppl_ratio": np.concatenate([existing["log_ppl_ratio"], log_ppl_ratio]),
        "evaluation_seconds": np.concatenate([existing["evaluation_seconds"], durations]),
        "sample_source": np.concatenate(
            [old_source, np.full(generation.num_samples, source_id, dtype=np.int8)]
        ),
        "tail_selection_score": np.concatenate([old_scores, selected_scores]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)
    metadata = {
        "base_dataset": str(dataset_path),
        "base_samples": old_count,
        "tail_samples_appended": generation.num_samples,
        "total_samples": old_count + generation.num_samples,
        "candidate_pool_size": candidate_pool_size,
        "candidate_seed": generation.seed,
        "selection_mode": selection_mode,
        "noise_seed": generation.noise_seed,
        "clean_perplexity": clean.perplexity,
        "tail_score_min": float(selected_scores.min()),
        "tail_score_max": float(selected_scores.max()),
        "tail_ppl_min": float(perplexities.min()),
        "tail_ppl_max": float(perplexities.max()),
        "tail_ppl_mean": float(perplexities.mean()),
        "mean_evaluation_seconds": float(durations.mean()),
        "total_generation_seconds": time.perf_counter() - started_at,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
