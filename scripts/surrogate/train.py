"""在 targeted multi-seed 标签上训练当前 surrogate ensemble。

脚本只训练和评估 surrogate，不会启动 PPO。训练完成后应先检查 acceptance
结果，再把 checkpoint 交给 PPO 训练入口。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.surrogate_training import EnsembleTrainingConfig
from uav_rl.tail_training import TailValidationCriteria, run_tail_validation_ablation


def parse_args() -> argparse.Namespace:
    """解析命令行参数，构造本次实验的运行配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据路径、模型配置和输出路径全部显式暴露，确保每次 surrogate 实验可审计。
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_train.npz"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_seed24_v5_validation.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_manifest.json"),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_tail_seed24_gated_expert_v5_metrics.json"),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=Path("artifacts/models/surrogate_targeted_ablation"),
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path("artifacts/results/surrogate_targeted_ablation"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_global.pth"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_validation.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_validation_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    """组织当前脚本的完整实验流程，包括加载、训练或评估和结果保存。"""
    args = parse_args()
    # 先解析参数，再构建/加载数据；这样 --help 和 plan-only 都不会触发真实模型推理。
    result = run_tail_validation_ablation(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        baseline_metrics_path=args.baseline_metrics,
        output_model_directory=args.model_directory,
        output_metrics_directory=args.result_directory,
        selected_checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        report_path=args.report,
        training_config=EnsembleTrainingConfig(epochs=args.epochs, patience=args.patience),
        criteria=TailValidationCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
