"""诊断 surrogate 数据覆盖范围；不会调用真实 PPL，也不会训练模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

from uav_rl.coverage_diagnostics import diagnose_surrogate_coverage


def parse_args() -> argparse.Namespace:
    """解析命令行参数，构造本次实验的运行配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据路径、模型配置和输出路径全部显式暴露，确保每次 surrogate 实验可审计。
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neighbors", type=int, default=8)
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
        "--checkpoint", type=Path,
        default=Path("artifacts/models/ppl_surrogate_targeted_residual.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/results/surrogate_data_coverage_diagnostic.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/results/surrogate_data_coverage_diagnostic.md"),
    )
    return parser.parse_args()


def main() -> None:
    """组织当前脚本的完整实验流程，包括加载、训练或评估和结果保存。"""
    args = parse_args()
    # 先解析参数，再构建/加载数据；这样 --help 和 plan-only 都不会触发真实模型推理。
    result = diagnose_surrogate_coverage(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        report_path=args.report,
        device_name=args.device,
        neighbors=args.neighbors,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
