"""Plot learning curves and held-out rewards from an RL comparison directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"ppo": "#1f77b4", "a2c": "#ff7f0e", "dqn": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _training_curve(history: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(history.get("training_reward", history.get("reward", [])), dtype=float)
    rollout_size = int(config["rollout_size"])
    target = int(config["training_episodes"])
    episodes = np.minimum(np.arange(1, len(rewards) + 1) * rollout_size, target)
    return episodes, rewards


def _validation_curve(history: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(history.get("validation_reward", []), dtype=float)
    rollout_size = int(config["rollout_size"])
    if len(values) == len(history.get("episodes", [])):
        episodes = np.asarray(history["episodes"], dtype=float)
    else:
        interval = int(
            config.get("validation_interval_rollouts", config.get("validation_interval", 1))
        )
        episodes = np.arange(1, len(values) + 1) * rollout_size * interval
    finite = np.isfinite(values)
    return episodes[finite], values[finite]


def _plot_seed_curves(
    axis: plt.Axes,
    curves: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    ylabel: str,
) -> None:
    for algorithm, series in curves.items():
        color = COLORS.get(algorithm)
        for x, y in series:
            axis.plot(x, y, color=color, alpha=0.22, linewidth=1.0)
        if not series:
            continue
        common_x = np.unique(np.concatenate([x for x, _ in series]))
        interpolated = np.stack([np.interp(common_x, x, y) for x, y in series])
        axis.plot(
            common_x,
            interpolated.mean(axis=0),
            color=color,
            linewidth=2.2,
            label=algorithm.upper(),
        )
    axis.set_xlabel("Environment episodes")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()


def main() -> None:
    args = parse_args()
    summary_path = args.comparison_dir / "comparison_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    validation_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in summary["rows"]:
        run_dir = Path(row["checkpoint"]).parent
        history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        algorithm = row["algorithm"]
        config = evaluation["algorithm_config"]
        training_curves.setdefault(algorithm, []).append(_training_curve(history, config))
        validation_curves.setdefault(algorithm, []).append(_validation_curve(history, config))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    _plot_seed_curves(axes[0], training_curves, ylabel="Surrogate training reward")
    _plot_seed_curves(axes[1], validation_curves, ylabel="Deterministic validation reward")
    figure.suptitle("UAV layer-assignment RL algorithm comparison")
    output = args.output or args.comparison_dir / "learning_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
