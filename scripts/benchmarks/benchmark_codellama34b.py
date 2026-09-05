"""使用多 GPU + CPU offload 测试 CodeLlama-34B 的真实 clean PPL。"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uav_rl.config import DataGenerationConfig
from uav_rl.data.ppl_dataset import prepare_corpus
from uav_rl.metrics import compute_perplexity


def parse_args() -> argparse.Namespace:
    """解析 CodeLlama-34B PPL 测试所需的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="codellama/CodeLlama-34b-hf",
        help="Hugging Face 模型 ID，或本地模型目录。",
    )
    parser.add_argument(
        "--text-sample-limit",
        type=int,
        default=50,
        help="最多读取多少条 WikiText 文本。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="每条文本最多保留多少个 token。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="PPL 评估 batch size。使用 CPU offload 时建议为 1。",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="模型权重和计算使用的数据类型。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/codellama34b_clean_ppl.json"),
        help="测试结果 JSON 输出路径。",
    )
    return parser.parse_args()


def synchronize_visible_gpus() -> None:
    """同步所有可见 GPU，避免异步 CUDA 工作影响耗时统计。"""
    if not torch.cuda.is_available():
        return
    for gpu_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(gpu_index)


def filter_short_sequences(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """删除有效 token 少于两个的序列，避免无法计算 next-token loss。"""
    sequence_lengths = encoded["attention_mask"].sum(dim=1)
    valid_sequences = sequence_lengths > 1

    filtered = {
        key: value[valid_sequences]
        for key, value in encoded.items()
    }

    if filtered["input_ids"].size(0) == 0:
        raise RuntimeError(
            "过滤短序列后没有剩余数据，无法计算 clean PPL。"
        )

    return filtered


def load_model_and_tokenizer(model_id: str, dtype: torch.dtype):
    """使用 device_map=auto 加载模型，使权重尽可能分布到可见 GPU。"""
    print("loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )

    # Llama 通常没有单独的 padding token；PPL 批处理需要一个 pad token。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model with device_map=auto...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def print_device_map(model: torch.nn.Module) -> None:
    """打印模型模块的设备分布，并提示是否发生 CPU offload。"""
    device_map = getattr(model, "hf_device_map", {})
    cpu_modules: list[str] = []

    print("========== device map ==========")
    for module_name, device in device_map.items():
        print(f"{module_name}: {device}", flush=True)
        if str(device) == "cpu":
            cpu_modules.append(module_name)
    print("================================", flush=True)

    if cpu_modules:
        print(
            "CPU offload modules:",
            ", ".join(cpu_modules),
            flush=True,
        )
    else:
        print("CPU offload modules: none", flush=True)


def main() -> None:
    """加载 CodeLlama-34B，计算 clean PPL 并保存测试结果。"""
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法运行 CodeLlama-34B benchmark。")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    print("========== environment ==========")
    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA version: {torch.version.cuda}")
    print(f"visible GPU count: {torch.cuda.device_count()}")
    for gpu_index in range(torch.cuda.device_count()):
        print(
            f"GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}",
            flush=True,
        )
    print("==================================", flush=True)

    generation = DataGenerationConfig(
        model_id=args.model_id,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="test",
        text_sample_limit=args.text_sample_limit,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )

    load_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        model_dtype,
    )
    load_seconds = time.perf_counter() - load_start

    print("model loaded successfully", flush=True)
    print("number of layers:", len(model.model.layers), flush=True)
    print("model loading seconds:", load_seconds, flush=True)
    print_device_map(model)

    print("preparing WikiText corpus...", flush=True)
    encoded = prepare_corpus(generation, tokenizer)
    print(
        "original input shape:",
        tuple(encoded["input_ids"].shape),
        flush=True,
    )

    # WikiText 中可能存在空文本、标题或只有一个 token 的文本。
    encoded = filter_short_sequences(encoded)
    print(
        "filtered input shape:",
        tuple(encoded["input_ids"].shape),
        flush=True,
    )
    print(
        "sequence lengths:",
        encoded["attention_mask"].sum(dim=1).tolist(),
        flush=True,
    )

    # 对 device_map=auto 模型，输入应该放到 embedding 所在设备。
    input_device = model.get_input_embeddings().weight.device
    print("input device:", input_device, flush=True)

    print("warming up...", flush=True)
    with torch.no_grad():
        warmup = compute_perplexity(
            model,
            encoded["input_ids"][:1],
            encoded["attention_mask"][:1],
            batch_size=1,
            device=input_device,
        )
    synchronize_visible_gpus()
    print("warm-up PPL:", warmup.perplexity, flush=True)

    print("running clean PPL benchmark...", flush=True)
    synchronize_visible_gpus()
    benchmark_start = time.perf_counter()

    result = compute_perplexity(
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
        batch_size=args.batch_size,
        device=input_device,
    )

    synchronize_visible_gpus()
    benchmark_seconds = time.perf_counter() - benchmark_start

    if not math.isfinite(result.perplexity):
        raise FloatingPointError("clean PPL 不是有限值。")

    output = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "text_sample_limit": args.text_sample_limit,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_layers": len(model.model.layers),
        "clean_perplexity": result.perplexity,
        "mean_negative_log_likelihood": result.mean_negative_log_likelihood,
        "evaluated_sequences": result.evaluated_sequences,
        "evaluated_tokens": result.evaluated_tokens,
        "batches": result.batches,
        "model_loading_seconds": load_seconds,
        "benchmark_seconds": benchmark_seconds,
        "seconds_per_sequence": benchmark_seconds / result.evaluated_sequences,
        "seconds_per_token": benchmark_seconds / result.evaluated_tokens,
        "device_map": {
            name: str(device)
            for name, device in getattr(model, "hf_device_map", {}).items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("========== clean PPL result ==========")
    print(json.dumps(output, indent=2), flush=True)
    print(f"result saved to: {args.output}", flush=True)
    print("=======================================", flush=True)


if __name__ == "__main__":
    main()
