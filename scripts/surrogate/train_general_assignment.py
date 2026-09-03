'''训练并评估 general-assignment surrogate。

脚本在 train split 上训练五成员 ensemble，用固定 validation split 选择模型，
最后只评估一次 held-out test split；整个过程不会启动 PPO。
'''

from __future__ import annotations

import argparse
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.surrogate_training import (
    EnsembleTrainingConfig,
    SurrogateAcceptanceCriteria,
    train_and_evaluate_ensemble,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，构造本次实验的运行配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据路径、模型配置和输出路径全部显式暴露，确保每次 surrogate 实验可审计。
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument('--train', type=Path, default=Path('artifacts/data/general_assignment_train.npz'))
    parser.add_argument('--validation', type=Path, default=Path('artifacts/data/general_assignment_validation.npz'))
    parser.add_argument('--test', type=Path, default=Path('artifacts/data/general_assignment_test.npz'))
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/data/general_assignment_manifest.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/models/ppl_surrogate_general_assignment_ensemble.pth'))
    parser.add_argument('--metrics', type=Path, default=Path('artifacts/results/ppl_surrogate_general_assignment_metrics.json'))
    parser.add_argument('--report', type=Path, default=Path('artifacts/results/ppl_surrogate_general_assignment_report.md'))
    parser.add_argument('--plots', type=Path, default=Path('artifacts/results/ppl_surrogate_general_assignment_plots'))
    return parser.parse_args()


def main() -> None:
    """组织当前脚本的完整实验流程，包括加载、训练或评估和结果保存。"""
    args = parse_args()
    # 先解析参数，再构建/加载数据；这样 --help 和 plan-only 都不会触发真实模型推理。
    # 训练函数内部先做数据 manifest/leakage 检查，再训练 ensemble 并评估 held-out test。
    result = train_and_evaluate_ensemble(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        dataset_manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        report_path=args.report,
        plot_directory=args.plots,
        training_config=EnsembleTrainingConfig(epochs=args.epochs, patience=args.patience),
        acceptance_criteria=SurrogateAcceptanceCriteria(),
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    print(result['acceptance'], flush=True)


if __name__ == '__main__':
    main()
