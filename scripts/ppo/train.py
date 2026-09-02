"""使用真实 CodeLlama PPL reward 训练或无损恢复 PPO。

每次运行都将 checkpoint、训练状态、真实 PPL cache、启动配置和 held-out
评估结果保存到同一个 ``--run-dir``，便于复现与审计。``--episodes`` 表示
目标总 episode 数，而不是本次新增数量；例如已有 1,000 episode 时，使用
``--resume --episodes 3000`` 会继续训练到总计 3,000 episode。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from uav_rl.config import DataGenerationConfig, PPOConfig, SystemConfig
from uav_rl.evaluation import STANDARD_METHODS, evaluate_methods
from uav_rl.experiment import estimate_latency_reference
from uav_rl.noise_seeds import test_noise_seeds, validation_noise_seeds
from uav_rl.rl import DeploymentEnvironment, PPOTrainer
from uav_rl.rl.environment import generate_channels
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    """解析真实 PPL PPO 的可复现实验参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs/ppo/next-round"),
        help="保存本次 PPO 所有 checkpoint、cache 和评估结果的目录。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="目标总 episode 数；恢复训练时增大该值即可延长训练。",
    )
    parser.add_argument("--rollout-size", type=int, default=128)
    parser.add_argument("--validation-channels", type=int, default=32)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=4,
        help="每 N 个 rollout 做一次 validation；默认 4，约每 512 episode 验证一次。",
    )
    parser.add_argument("--test-channels", type=int, default=64)
    parser.add_argument("--training-noise-samples", type=int, default=4)
    parser.add_argument("--validation-noise-samples", type=int, default=16)
    parser.add_argument("--test-noise-samples", type=int, default=16)
    parser.add_argument("--training-noise-seed", type=int, default=20260815)
    parser.add_argument("--validation-noise-seed", type=int, default=20260816)
    parser.add_argument("--test-noise-seed", type=int, default=20260817)
    parser.add_argument("--model", default="codellama/CodeLlama-7b-hf")
    parser.add_argument(
        "--dataset-arrow",
        type=Path,
        help="读取缓存的 Hugging Face Arrow split，避免重新解析数据集。",
    )
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="从 training_state.pth 精确恢复训练。")
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="跳过耗时的 held-out PPO/baseline 最终评估。",
    )
    return parser.parse_args()


def run_paths(run_directory: Path) -> dict[str, Path]:
    """返回一次可独立恢复的标准产物路径布局。"""

    return {
        "best_model": run_directory / "best_policy.pth",
        "state": run_directory / "training_state.pth",
        "ppl_cache": run_directory / "ppl_cache.jsonl",
        "launch_config": run_directory / "run_config.json",
        "result": run_directory / "evaluation.json",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """以原子替换方式写入可读的实验记录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def prepare_run_directory(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    """防止误覆盖旧实验，同时允许从完整训练状态精确恢复。"""

    if args.resume:
        if not paths["state"].is_file():
            raise FileNotFoundError(f"resume state does not exist: {paths['state']}")
        return

    occupied = [path for path in paths.values() if path.exists()]
    if occupied:
        rendered_paths = ", ".join(str(path) for path in occupied)
        raise FileExistsError(
            "run directory already contains artifacts; choose a new --run-dir or pass --resume: "
            f"{rendered_paths}"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """执行真实 PPL PPO 训练，并可在 held-out channel/seed 上完成最终评估。"""

    args = parse_args()
    # episode 数必须为正；其余参数由 PPOConfig 和 evaluator 在后续阶段继续校验。
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    # 先确定所有产物路径并检查目录，避免训练开始后才发现输出会覆盖旧结果。
    paths = run_paths(args.run_dir)
    prepare_run_directory(args, paths)
    print(
        f"starting_true_ppl_training run_dir={args.run_dir} target_episodes={args.episodes}",
        flush=True,
    )

    # 真实 PPL 训练关闭 teacher/behavior cloning，保证 reward 直接来自当前 deployment 的 LLM 评估。
    system = SystemConfig()
    config = replace(
        PPOConfig(system=system),
        training_episodes=args.episodes,
        rollout_size=args.rollout_size,
        teacher_channels=0,
        behavior_cloning_epochs=0,
        validation_channels=args.validation_channels,
        validation_interval=args.validation_interval,
        test_channels=args.test_channels,
        training_noise_samples=args.training_noise_samples,
        training_noise_seed=args.training_noise_seed,
        validation_noise_samples=args.validation_noise_samples,
        validation_noise_seed=args.validation_noise_seed,
        test_noise_samples=args.test_noise_samples,
        test_noise_seed=args.test_noise_seed,
    )
    # generation 配置与 run_config 一起保存；模型、token 数和 batch size 改变都会影响 PPL。
    generation = replace(
        DataGenerationConfig(),
        model_id=args.model,
        dataset_arrow_file=str(args.dataset_arrow) if args.dataset_arrow is not None else None,
        text_sample_limit=args.sample_limit,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    latency_reference = estimate_latency_reference(system, config.seed)
    # evaluator 内部维护 PPL cache；重复出现的 drop vector/noise seed 不会重复 forward。
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        cache_path=paths["ppl_cache"],
    )
    environment = DeploymentEnvironment(system, evaluator, latency_reference)
    trainer = PPOTrainer(
        config,
        environment,
        teacher_quality_model=None,
        require_ppo_checkpoint=True,
    )
    run_metadata = {
        "quality_backend": "true_ppl",
        "generation": asdict(generation),
        "latency_reference": latency_reference,
        "validation_noise_seeds": validation_noise_seeds(
            config.validation_noise_samples,
            config.validation_noise_seed,
        ).tolist(),
    }

    # resume 时保留原始 launch_config，避免把恢复运行误记成一套新的实验协议。
    if not args.resume:
        write_json(
            paths["launch_config"],
            {
                "format_version": 1,
                "purpose": "true-PPL PPO training",
                "ppo_config": asdict(config),
                "generation": asdict(generation),
                "run_metadata": run_metadata,
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
        )

    # 这里的计时只覆盖 PPO training，不把后面的 held-out evaluation 混入训练耗时。
    started_at = time.perf_counter()
    history = trainer.train(
        paths["best_model"],
        state_path=paths["state"],
        resume=args.resume,
        run_metadata=run_metadata,
    )
    training_seconds = time.perf_counter() - started_at

    result: dict[str, Any] = {
        "ppo_config": asdict(config),
        "generation": asdict(generation),
        "latency_reference_seconds": latency_reference,
        "training_seconds": training_seconds,
        "resumed": args.resume,
        "run_directory": str(args.run_dir),
        "training_state": str(paths["state"]),
        "best_model": str(paths["best_model"]),
        "ppl_cache": str(paths["ppl_cache"]),
        "quality_evaluator": evaluator.metadata(),
        "validation_noise_seeds": trainer.validation_noise_seeds.tolist(),
        "training_history": history,
    }

    if args.skip_final_evaluation:
        result["final_evaluation"] = {"skipped": True}
    else:
        test_channels = generate_channels(config.test_channels, config.test_seed, system)
        held_out_test_noise_seeds = test_noise_seeds(
            config.test_noise_samples,
            config.test_noise_seed,
        )
        result["test_noise_seeds"] = held_out_test_noise_seeds.tolist()
        result["methods"] = evaluate_methods(
            environment,
            trainer.model,
            test_channels,
            evaluator.clean_perplexity,
            random_seed=config.test_seed + 1,
            method_names=STANDARD_METHODS,
            noise_seeds=held_out_test_noise_seeds,
        )

    write_json(paths["result"], result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
