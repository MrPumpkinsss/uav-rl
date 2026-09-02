"""在冻结 validation 数据上训练或恢复 targeted tail residual surrogate。

residual 只学习相对于 global surrogate 的修正量，并由 hazard gate 控制修正强度。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.tail_residual import TailResidualTrainingConfig, train_tail_residual_surrogate
from uav_rl.tail_training import TailValidationCriteria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据路径、模型配置和输出路径全部显式暴露，确保每次 surrogate 实验可审计。
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--train", type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_train.npz"),
    )
    parser.add_argument(
        "--validation", type=Path,
        default=Path("artifacts/data/codellama_surrogate_tail_seed24_v5_validation.npz"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("artifacts/data/codellama_surrogate_targeted_manifest.json"),
    )
    parser.add_argument(
        "--global-validation-summary", type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_validation.json"),
    )
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_global.pth"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual.pth"),
    )
    parser.add_argument(
        "--state", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual_state.pth"),
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_residual_metrics.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/results/ppl_surrogate_targeted_residual_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 先解析参数，再构建/加载数据；这样 --help 和 plan-only 都不会触发真实模型推理。
    # residual 训练依赖已经冻结的 global checkpoint，不能直接把它当作独立 surrogate 训练。
    result = train_tail_residual_surrogate(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        global_validation_summary_path=args.global_validation_summary,
        base_checkpoint_path=args.base_checkpoint,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        report_path=args.report,
        state_path=args.state,
        training_config=TailResidualTrainingConfig(
            epochs=args.epochs,
            patience=args.patience,
        ),
        criteria=TailValidationCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
