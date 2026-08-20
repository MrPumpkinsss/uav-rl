"""True-CodeLlama evaluation of PPO and deployment baselines."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uav_rl.baselines import (
    compute_greedy_baseline,
    dynamic_programming_baseline,
    random_baseline,
    strong_link_baseline,
)
from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.data.ppl_dataset import prepare_corpus, torch_dtype
from uav_rl.metrics import compute_perplexity
from uav_rl.models import activation_dropout
from uav_rl.quality import SurrogateQualityEvaluator
from uav_rl.rl.environment import DeploymentEnvironment, generate_channels
from uav_rl.rl.oracle import four_segment_surrogate_oracle, full_surrogate_oracle
from uav_rl.rl.policy import ContinuousDeploymentActorCritic
from uav_rl.surrogate import load_surrogate
from uav_rl.wireless import boundary_drop_probabilities, collaborative_latency


def _load_policy(path: Path, system: SystemConfig) -> ContinuousDeploymentActorCritic:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hidden_dim = int(checkpoint["ppo_config"]["hidden_dim"])
    policy = ContinuousDeploymentActorCritic(system, hidden_dim)
    policy.load_state_dict(checkpoint["model_state"])
    return policy.eval()


def _policy_deployments(
    policy: ContinuousDeploymentActorCritic,
    channels: np.ndarray,
    system: SystemConfig,
) -> np.ndarray:
    scale = system.channel_gain_max - system.channel_gain_min
    states = ((channels - system.channel_gain_min) / scale).astype(np.float32)
    with torch.no_grad():
        return policy.sample(
            torch.from_numpy(states.reshape(len(states), -1)), deterministic=True
        ).actions.numpy()


def run_true_policy_benchmark(
    policy_path: Path,
    output_path: Path,
    *,
    channel_count: int,
    test_seed: int,
    random_seed: int,
    latency_reference: float,
    surrogate_path: Path,
    generation: DataGenerationConfig,
    system: SystemConfig,
    device_name: str = "cuda",
    method_names: tuple[str, ...] = (
        "ppo",
        "random",
        "compute_greedy",
        "strong_link",
        "dynamic_programming",
        "four_segment_surrogate_oracle",
        "full_surrogate_oracle",
    ),
) -> dict[str, Any]:
    """Evaluate all methods with true activation-corruption corpus PPL."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    known_methods = {
        "ppo",
        "random",
        "compute_greedy",
        "strong_link",
        "dynamic_programming",
        "four_segment_surrogate_oracle",
        "full_surrogate_oracle",
    }
    unknown_methods = set(method_names) - known_methods
    if unknown_methods:
        raise ValueError(f"unknown benchmark methods: {sorted(unknown_methods)}")
    channels = generate_channels(channel_count, test_seed, system)
    methods: dict[str, np.ndarray] = {}
    selection_seconds: dict[str, float] = {}

    def add_method(name: str, deployments: np.ndarray, started_at: float) -> None:
        methods[name] = deployments
        selection_seconds[name] = time.perf_counter() - started_at

    for name in method_names:
        started_at = time.perf_counter()
        if name == "ppo":
            policy = _load_policy(policy_path, system)
            deployments = _policy_deployments(policy, channels, system)
        elif name == "random":
            deployments = random_baseline(channels, random_seed, system)
        elif name == "compute_greedy":
            deployments = np.repeat(
                compute_greedy_baseline(channels, system)[None, :], channel_count, axis=0
            )
        elif name == "strong_link":
            deployments = strong_link_baseline(channels, system)
        elif name == "dynamic_programming":
            deployments = dynamic_programming_baseline(
                channels,
                system,
                latency_reference,
            )
        else:
            continue
        add_method(name, deployments, started_at)

    oracle_names = {
        "four_segment_surrogate_oracle",
        "full_surrogate_oracle",
    }
    requested_oracles = oracle_names.intersection(method_names)
    if requested_oracles:
        oracle_environment = DeploymentEnvironment(
            system,
            SurrogateQualityEvaluator(load_surrogate(surrogate_path, device)),
            latency_reference,
        )
        if "four_segment_surrogate_oracle" in requested_oracles:
            started_at = time.perf_counter()
            deployments = four_segment_surrogate_oracle(
                channels,
                oracle_environment,
                progress_label="four_segment_surrogate_oracle",
            )
            add_method("four_segment_surrogate_oracle", deployments, started_at)
        if "full_surrogate_oracle" in requested_oracles:
            started_at = time.perf_counter()
            deployments = full_surrogate_oracle(
                channels,
                oracle_environment,
                progress_label="full_surrogate_oracle",
            )
            add_method("full_surrogate_oracle", deployments, started_at)
        del oracle_environment
        if device.type == "cuda":
            torch.cuda.empty_cache()

    methods = {name: methods[name] for name in method_names}

    tokenizer = AutoTokenizer.from_pretrained(generation.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        generation.model_id,
        dtype=torch_dtype(generation.dtype),
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    encoded = prepare_corpus(generation, tokenizer)
    clean = compute_perplexity(
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
        batch_size=generation.batch_size,
        device=device,
    )

    cache: dict[bytes, float] = {}
    method_results: dict[str, Any] = {}
    total_evaluations = channel_count * len(methods)
    completed = 0
    model_forwards = 0
    started_at = time.perf_counter()
    for name, deployments in methods.items():
        ppls: list[float] = []
        latencies: list[float] = []
        for channel, deployment in zip(channels, deployments, strict=True):
            probabilities = boundary_drop_probabilities(deployment, channel, system)
            key = probabilities.tobytes()
            if key not in cache:
                active = {
                    layer: float(probability)
                    for layer, probability in enumerate(probabilities)
                    if probability > 0.0
                }
                torch.manual_seed(generation.noise_seed)
                with activation_dropout(model, active):
                    result = compute_perplexity(
                        model,
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        batch_size=generation.batch_size,
                        device=device,
                    )
                if not math.isfinite(result.perplexity):
                    raise FloatingPointError("non-finite PPL in true policy benchmark")
                cache[key] = result.perplexity
                model_forwards += 1
            ppls.append(cache[key])
            latencies.append(collaborative_latency(deployment, channel, system).total_seconds)
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == total_evaluations:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[{completed:4d}/{total_evaluations}] method={name} "
                    f"llm_evaluations={model_forwards} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        ppl_array = np.asarray(ppls, dtype=np.float64)
        latency_array = np.asarray(latencies, dtype=np.float64)
        log_ratio = np.log(ppl_array / clean.perplexity)
        rewards = -(
            system.quality_weight * log_ratio
            + (1.0 - system.quality_weight) * latency_array / latency_reference
        )
        method_results[name] = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "latency_mean_seconds": float(latency_array.mean()),
            "latency_std_seconds": float(latency_array.std()),
            "true_ppl_mean": float(ppl_array.mean()),
            "true_ppl_std": float(ppl_array.std()),
            "true_ppl_median": float(np.median(ppl_array)),
            "true_ppl_max": float(ppl_array.max()),
            "log_ppl_ratio_mean": float(log_ratio.mean()),
        }

    result = {
        "model_id": generation.model_id,
        "test_seed": test_seed,
        "random_seed": random_seed,
        "test_channels": channel_count,
        "clean_perplexity": clean.perplexity,
        "evaluated_tokens": clean.evaluated_tokens,
        "latency_reference_seconds": latency_reference,
        "quality_weight": system.quality_weight,
        "surrogate_path_for_oracles": (str(surrogate_path) if requested_oracles else None),
        "deployment_selection_seconds": selection_seconds,
        "requested_method_evaluations": total_evaluations,
        "evaluated_methods": list(methods),
        "unique_llm_evaluations": model_forwards,
        "elapsed_seconds": time.perf_counter() - started_at,
        "methods": method_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
