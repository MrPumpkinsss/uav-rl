"""从 PPO training_state.pth 导出训练收敛曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """解析 checkpoint 和输出目录参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="PPO training_state.pth 路径。")
    parser.add_argument("--output-dir", type=Path, required=True, help="四张图的输出目录。")
    return parser.parse_args()


def _save_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
    output: Path,
    color: str,
) -> None:
    """绘制一条训练历史曲线并保存为 PNG。"""
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.plot(x, y, color=color, linewidth=1.4)
    axis.set_title(title)
    axis.set_xlabel("Completed training episodes")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    """读取 PPO 历史并生成 reward、validation、latency 和 PPL 四张图。"""
    args = parse_args()
    # training_state 同时保存配置和 history；这里只读取 CPU 张量，避免占用 GPU。
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    history = state["history"]
    rollout_size = int(state["ppo_config"]["rollout_size"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # reward、latency 和 log-PPL 每个点对应一次 rollout 的平均值。
    reward = np.asarray(history["reward"], dtype=np.float64)
    rollout_episodes = np.arange(1, len(reward) + 1) * rollout_size
    _save_curve(
        rollout_episodes, reward, title="PPO training reward convergence",
        ylabel="Mean rollout reward", output=args.output_dir / "01_training_reward.png", color="#1f77b4"
    )

    # validation 只在固定 validation channels 上定期计算，因此单独使用评估点序号。
    validation = np.asarray(history["validation_reward"], dtype=np.float64)
    validation_x = np.linspace(0, rollout_episodes[-1], len(validation)) if len(validation) else np.array([])
    _save_curve(
        validation_x, validation, title="PPO validation reward convergence",
        ylabel="Validation reward", output=args.output_dir / "02_validation_reward.png", color="#d62728"
    )

    latency = np.asarray(history["latency"], dtype=np.float64)
    _save_curve(
        rollout_episodes, latency, title="PPO latency during training",
        ylabel="Mean latency (s)", output=args.output_dir / "03_latency.png", color="#2ca02c"
    )

    log_ppl = np.asarray(history["log_ppl_ratio"], dtype=np.float64)
    _save_curve(
        rollout_episodes, log_ppl, title="PPO quality degradation during training",
        ylabel="Mean log-PPL ratio", output=args.output_dir / "04_log_ppl_ratio.png", color="#9467bd"
    )
    print(f"saved four figures to {args.output_dir}")


if __name__ == "__main__":
    main()
