"""采集 Qwen3.5-4B 的真实 noisy-PPL surrogate 数据集。

默认 workload 为 12,288 个真实 noisy-PPL label，按约 3 秒/label 估计约 10.2 小时。
实际时长取决于 GPU、CPU offload、文本长度和 batch size；先用 --plan-only 检查数量。
脚本采用与 7B 相同的无线参数，但独立保存 Qwen3.5-4B 的数据、cache 和 manifest。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoConfig

from uav_rl.config import DataGenerationConfig, SystemConfig
from uav_rl.data.general_assignment_dataset import (
    GeneralAssignmentDatasetConfig,
    aggregate_general_assignment_cache,
    build_general_assignment_plan,
    collect_general_assignment_labels,
)
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.true_quality import TruePPLQualityEvaluator


DEFAULT_ROOT = Path("artifacts/qwen35_4b_surrogate_10h")


def parse_args() -> argparse.Namespace:
    """解析 Qwen3.5-4B 数据采集参数，所有输出路径都显式可修改。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-uavs", type=int, default=5)
    parser.add_argument("--max-layers-per-uav", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-sample-limit", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--action-seed", type=int, default=20260906)

    # 1024*4 + 256*16 + 256*16 = 12,288 个真实 noisy-PPL labels。
    parser.add_argument("--train-actions", type=int, default=1024)
    parser.add_argument("--validation-actions", type=int, default=256)
    parser.add_argument("--test-actions", type=int, default=256)
    parser.add_argument("--train-noise-samples", type=int, default=4)
    parser.add_argument("--validation-noise-samples", type=int, default=16)
    parser.add_argument("--test-noise-samples", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument(
        "--seconds-per-forward",
        type=float,
        default=3.0,
        help="仅用于估算总时长，不会改变采集过程。",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def infer_text_layers(model_id: str) -> int:
    """读取模型 config 中的文本层数，避免误把视觉层算进 deployment。"""
    config = AutoConfig.from_pretrained(model_id)
    text_config = getattr(config, "text_config", None)
    layer_count = getattr(text_config, "num_hidden_layers", None)
    if layer_count is None:
        layer_count = getattr(config, "num_hidden_layers", None)
    if layer_count is None:
        raise ValueError("模型 config 中没有 num_hidden_layers")
    return int(layer_count)


def build_configs(args: argparse.Namespace):
    """构建 Qwen3.5-4B 的生成、资源和 split 采样配置。"""
    num_layers = infer_text_layers(args.model_id)
    if num_layers != 32:
        raise ValueError(
            f"当前脚本按 Qwen3.5-4B 的 32 个文本 decoder layers 设计，实际为 {num_layers}"
        )
    system = SystemConfig(
        num_layers=num_layers,
        num_uavs=args.num_uavs,
        max_layers_per_uav=args.max_layers_per_uav,
    )
    resource = ResourceConstrainedConfig(system=system)
    generation = DataGenerationConfig(
        model_id=args.model_id,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="test",
        text_sample_limit=args.text_sample_limit,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=256,
        seed=args.action_seed,
        noise_seed=314159,
        dtype=args.dtype,
    )
    dataset = GeneralAssignmentDatasetConfig(
        action_seed=args.action_seed,
        training_noise_seed=args.action_seed + 101,
        validation_noise_seed=args.action_seed + 102,
        test_noise_seed=args.action_seed + 103,
        train_actions=args.train_actions,
        validation_actions=args.validation_actions,
        test_actions=args.test_actions,
        training_noise_samples=args.train_noise_samples,
        validation_noise_samples=args.validation_noise_samples,
        test_noise_samples=args.test_noise_samples,
    )
    return generation, resource, dataset


def main() -> None:
    """生成可恢复计划，执行真实 Qwen3.5-4B PPL，并聚合为 NPZ 数据集。"""
    args = parse_args()
    if args.seconds_per_forward <= 0:
        raise ValueError("--seconds-per-forward 必须为正数")
    generation, resource, dataset = build_configs(args)

    root = args.root
    plan_path = root / "plans" / "assignment_plan.json"
    sample_cache = root / "cache" / "labels.jsonl"
    ppl_cache = root / "cache" / "ppl.jsonl"
    output_directory = root / "data"
    log_directory = root / "logs"
    for directory in (plan_path.parent, sample_cache.parent, output_directory, log_directory):
        directory.mkdir(parents=True, exist_ok=True)

    plan = build_general_assignment_plan(
        config=resource,
        generation=generation,
        dataset=dataset,
        plan_path=plan_path,
        existing_cache_paths=(sample_cache,),
    )
    forwards = sum(len(action["noise_seeds"]) for action in plan["actions"])
    summary = {
        "model_id": generation.model_id,
        "num_layers": resource.system.num_layers,
        "num_boundaries": resource.system.num_layers - 1,
        "num_uavs": resource.system.num_uavs,
        "actions": {
            split: sum(action["split"] == split for action in plan["actions"])
            for split in ("train", "validation", "test")
        },
        "true_ppl_forwards": forwards,
        "estimated_hours": forwards * args.seconds_per_forward / 3600.0,
        "plan": str(plan_path),
        "sample_cache": str(sample_cache),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.plan_only:
        return

    # Qwen3.5 可能是 multimodal wrapper；TruePPLQualityEvaluator 会自动寻找文本 decoder layers。
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        device_map=args.device_map,
        cache_path=ppl_cache,
        progress_interval=args.progress_interval,
    )
    if evaluator.num_boundaries != resource.system.num_layers - 1:
        raise ValueError(
            f"boundary 维度不一致：evaluator={evaluator.num_boundaries}, "
            f"resource={resource.system.num_layers - 1}"
        )

    collection = collect_general_assignment_labels(
        plan,
        evaluator,
        sample_cache,
        progress_interval=args.progress_interval,
    )
    outputs = aggregate_general_assignment_cache(
        plan,
        sample_cache,
        output_directory,
        legacy_train_path=None,
        quality_metadata=evaluator.metadata(),
    )
    print(json.dumps({**summary, "collection": collection, "outputs": outputs}, default=str, indent=2), flush=True)


if __name__ == "__main__":
    main()
