"""End-to-end compute-time benchmark for one perplexity evaluation."""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from uav_rl.metrics import PerplexityResult, compute_perplexity


@dataclass(frozen=True)
class PerplexityBenchmarkConfig:
    """Configuration defining a reproducible PPL timing experiment."""

    model_id: str = "codellama/CodeLlama-7b-hf"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "test"
    sample_limit: int = 50
    max_length: int = 512
    batch_size: int = 4
    measured_runs: int = 3
    device: str = "cuda"
    dtype: str = "float16"

    def validate(self) -> None:
        if self.sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.measured_runs < 1:
            raise ValueError("measured_runs must be positive")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")


@dataclass(frozen=True)
class PerplexityBenchmarkResult:
    """Serializable timing, metric, dataset, and hardware information."""

    config: dict[str, Any]
    perplexity: float
    evaluated_sequences: int
    evaluated_tokens: int
    batches_per_evaluation: int
    evaluation_seconds: list[float]
    mean_seconds: float
    median_seconds: float
    std_seconds: float
    tokens_per_second: float
    model_load_seconds: float
    data_prepare_seconds: float
    peak_memory_mib: float | None
    gpu_name: str | None
    torch_version: str
    transformers_version: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_and_tokenize(config: PerplexityBenchmarkConfig, tokenizer: Any) -> dict[str, Any]:
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=f"{config.dataset_split}[:{config.sample_limit}]",
    )
    texts = [text for text in dataset["text"] if text.strip()]
    if not texts:
        raise ValueError("the selected dataset slice contains no non-empty text")
    return tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_perplexity_benchmark(
    config: PerplexityBenchmarkConfig,
) -> PerplexityBenchmarkResult:
    """Measure the compute time of one complete PPL evaluation.

    Loading and tokenization are reported separately and excluded from evaluation time.
    A one-batch warm-up prevents kernel initialization from contaminating the first run.
    """

    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")

    model_load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=_resolve_dtype(config.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model_load_seconds = time.perf_counter() - model_load_started

    data_prepare_started = time.perf_counter()
    encoded = _load_and_tokenize(config, tokenizer)
    data_prepare_seconds = time.perf_counter() - data_prepare_started

    warmup_end = min(config.batch_size, encoded["input_ids"].size(0))
    compute_perplexity(
        model,
        encoded["input_ids"][:warmup_end],
        encoded["attention_mask"][:warmup_end],
        batch_size=config.batch_size,
        device=device,
    )
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    evaluation_seconds: list[float] = []
    metric_result: PerplexityResult | None = None
    for _ in range(config.measured_runs):
        _synchronize(device)
        started_at = time.perf_counter()
        metric_result = compute_perplexity(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            batch_size=config.batch_size,
            device=device,
        )
        _synchronize(device)
        evaluation_seconds.append(time.perf_counter() - started_at)

    assert metric_result is not None
    mean_seconds = statistics.fmean(evaluation_seconds)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )

    import transformers

    return PerplexityBenchmarkResult(
        config=asdict(config),
        perplexity=metric_result.perplexity,
        evaluated_sequences=metric_result.evaluated_sequences,
        evaluated_tokens=metric_result.evaluated_tokens,
        batches_per_evaluation=metric_result.batches,
        evaluation_seconds=evaluation_seconds,
        mean_seconds=mean_seconds,
        median_seconds=statistics.median(evaluation_seconds),
        std_seconds=statistics.pstdev(evaluation_seconds),
        tokens_per_second=metric_result.evaluated_tokens / mean_seconds,
        model_load_seconds=model_load_seconds,
        data_prepare_seconds=data_prepare_seconds,
        peak_memory_mib=peak_memory,
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        python_version=platform.python_version(),
    )
