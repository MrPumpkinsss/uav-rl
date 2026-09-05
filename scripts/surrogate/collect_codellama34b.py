"""采集 CodeLlama-34B 的真实 noisy-PPL surrogate 数据集。

本脚本面向 48 层 CodeLlama-34B，支持两张或多张 GPU 的
``device_map=auto`` 加载，并使用 JSONL cache 支持 SSH 中断恢复。
它采集的是 deployment、channel、boundary drop probabilities 和真实 PPL 标签，
不是保存完整 hidden-state activation tensor。
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


def parse_args() -> argparse.Namespace:
    """解析 34B 数据集采集参数，保证模型和输出目录都显式可审计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="codellama/CodeLlama-34b-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers 的 device_map；34B 多卡运行使用 auto。",
    )
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-uavs", type=int, default=5)
    parser.add_argument("--max-layers-per-uav", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-sample-limit", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--action-seed", type=int, default=20260905)
    parser.add_argument("--train-actions", type=int, default=384)
    parser.add_argument("--validation-actions", type=int, default=64)
    parser.add_argument("--test-actions", type=int, default=64)
    parser.add_argument("--train-noise-samples", type=int, default=4)
    parser.add_argument("--validation-noise-samples", type=int, default=16)
    parser.add_argument("--test-noise-samples", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/codellama34b_surrogate/data"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("artifacts/codellama34b_surrogate/plans/assignment_plan.json"),
    )
    parser.add_argument(
        "--sample-cache",
        type=Path,
        default=Path("artifacts/codellama34b_surrogate/cache/labels.jsonl"),
    )
    parser.add_argument(
        "--ppl-cache",
        type=Path,
        default=Path("artifacts/codellama34b_surrogate/cache/ppl.jsonl"),
    )
    return parser.parse_args()


def infer_layer_count(model_id: str) -> int:
    """从模型 config 读取 Transformer 层数，避免手工写错 boundary 维度。"""
    config = AutoConfig.from_pretrained(model_id)
    layer_count = getattr(config, "num_hidden_layers", None)
    if layer_count is None:
        raise ValueError("模型 config 中没有 num_hidden_layers")
    return int(layer_count)


def build_configs(args: argparse.Namespace) -> tuple[DataGenerationConfig, ResourceConstrainedConfig, GeneralAssignmentDatasetConfig]:
    """构建 34B 的模型、48 层资源和 train/validation/test 采样配置。"""
    detected_layers = infer_layer_count(args.model_id)
    num_layers = args.num_layers or detected_layers
    if num_layers != detected_layers:
        raise ValueError(
            f"--num-layers={num_layers} 与模型 config 的层数 {detected_layers} 不一致"
        )
    if args.num_uavs != 5:
        raise ValueError(
            "当前 resource profile 为 5 台 UAV；如需其他数量，应同时提供对应容量和能耗配置。"
        )

    max_layers = args.max_layers_per_uav or math.ceil(num_layers / args.num_uavs)
    system = SystemConfig(
        num_layers=num_layers,
        num_uavs=args.num_uavs,
        max_layers_per_uav=max_layers,
    )
    resource = ResourceConstrainedConfig(system=system)
    generation = DataGenerationConfig(
        model_id=args.model_id,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="test",
        dataset_arrow_file=None,
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
    """先生成可恢复的采样计划，再运行真实 34B noisy-PPL 并聚合 NPZ。"""
    args = parse_args()
    generation, resource, dataset = build_configs(args)

    # 先生成 plan；该阶段不加载 34B，可以先审查预计 action 和 forward 数量。
    plan = build_general_assignment_plan(
        config=resource,
        generation=generation,
        dataset=dataset,
        plan_path=args.plan,
        existing_cache_paths=(args.sample_cache,),
    )
    summary = {
        "model_id": generation.model_id,
        "num_layers": resource.system.num_layers,
        "num_boundaries": resource.system.num_layers - 1,
        "num_uavs": resource.system.num_uavs,
        "actions": {
            split: sum(action["split"] == split for action in plan["actions"])
            for split in ("train", "validation", "test")
        },
        "true_ppl_forwards": sum(
            len(action["noise_seeds"]) for action in plan["actions"]
        ),
        "plan": str(args.plan),
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.plan_only:
        return

    # 34B 使用 device_map=auto；evaluator 内部不会再调用 model.to(cuda)。
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        device_map=args.device_map,
        cache_path=args.ppl_cache,
        progress_interval=args.progress_interval,
    )
    if evaluator.num_boundaries != resource.system.num_layers - 1:
        raise ValueError(
            "模型 boundary 数与 resource configuration 不一致："
            f" evaluator={evaluator.num_boundaries},"
            f" resource={resource.system.num_layers - 1}"
        )

    # 每完成一个 action/noise-seed，立即追加 JSONL，SSH 中断后可继续运行。
    collection = collect_general_assignment_labels(
        plan,
        evaluator,
        args.sample_cache,
        progress_interval=args.progress_interval,
    )
    outputs = aggregate_general_assignment_cache(
        plan,
        args.sample_cache,
        args.output_directory,
        legacy_train_path=None,
        quality_metadata=evaluator.metadata(),
    )
    print(
        json.dumps(
            {**summary, "collection": collection, "outputs": outputs},
            default=str,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
