"""Compare PPO with authoritative heuristic/system baselines on one channel set.

This stage uses the frozen general-assignment surrogate for inexpensive screening.
It freezes the generated deployments so a later command can evaluate the exact
same actions with the true LLM.  The main table intentionally emphasizes model
partitioning and optimization baselines; A2C/DQN belong to the separate RL
algorithm-ablation pipeline under ``scripts/rl``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from uav_rl.system_baselines import (
    edge_shard_uav_baseline,
    hexgen_inspired_search_baseline,
    lingualinked_uav_baseline,
)
from uav_rl.resource_baselines import (
    jointdnn_multi_uav_baseline,
    neurosurgeon_best_split_baseline,
    petals_balanced_pipeline_baseline,
    pipeedge_uav_latency_baseline,
    random_feasible_baseline,
    surrogate_simulated_annealing_baseline,
)
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.policy_io import load_layerwise_policy, resource_config_from_dict
from uav_rl.surrogate import load_surrogate

METHOD_LABELS = {
    "ppo_deterministic": "Proposed PPO",
    "edge_shard_uav": "EdgeShard-UAV",
    "hexgen_inspired": "HexGen-inspired",
    "lingualinked_uav": "LinguaLinked-UAV",
    "jointdnn_multi_uav": "JointDNN-MUAV",
    "pipeedge_uav_latency": "PipeEdge-UAV",
    "petals_balanced": "Petals-balanced",
    "neurosurgeon_inspired": "Neurosurgeon-inspired",
    "simulated_annealing": "Simulated annealing",
    "random_feasible": "Random feasible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--surrogate",
        type=Path,
        default=Path(
            "artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/runs/system_baseline_comparison/authoritative_heuristics_2026-08-30"
        ),
    )
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--channel-seed", type=int, default=20260910)
    parser.add_argument("--edge-shard-plans-per-state", type=int, default=8)
    parser.add_argument("--hexgen-population", type=int, default=48)
    parser.add_argument("--hexgen-generations", type=int, default=48)
    parser.add_argument("--jointdnn-time-limit", type=float, default=1.0)
    parser.add_argument("--annealing-steps", type=int, default=1024)
    parser.add_argument("--random-seed", type=int, default=20260911)
    parser.add_argument("--surrogate-device", default="cuda")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metrics(
    deployments: np.ndarray,
    rewards: np.ndarray,
    details: dict[str, np.ndarray],
) -> dict[str, float]:
    boundaries = np.count_nonzero(deployments[:, 1:] != deployments[:, :-1], axis=1)
    ppl_ratio = np.exp(details["log_ppl_ratio"].astype(np.float64))
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_standard_error": float(rewards.std(ddof=1) / np.sqrt(len(rewards))),
        "reward_p10": float(np.quantile(rewards, 0.10)),
        "reward_p90": float(np.quantile(rewards, 0.90)),
        "log_ppl_ratio_mean": float(details["log_ppl_ratio"].mean()),
        "ppl_ratio_mean": float(ppl_ratio.mean()),
        "latency_mean_seconds": float(details["latency_seconds"].mean()),
        "latency_std_seconds": float(details["latency_seconds"].std()),
        "invalid_fraction": float(details["invalid"].mean()),
        "boundary_count_mean": float(boundaries.mean()),
        "boundary_count_p90": float(np.quantile(boundaries, 0.90)),
    }


def _write_report(output: Path, payload: dict) -> None:
    rows = payload["methods"]
    lines = [
        "# Frozen surrogate system-baseline comparison",
        "",
        "## Status",
        "",
        "This is a frozen **surrogate-screening** comparison. The exact deployments and channel "
        "matrices are saved for a later common-seed true-LLM evaluation; these numbers must not "
        "be presented as final true-PPL results.",
        "",
        "## Protocol",
        "",
        f"- Channels: `{payload['channels']}`",
        f"- Channel seed: `{payload['channel_seed']}`",
        f"- PPO checkpoint: `{payload['checkpoint']}`",
        f"- Frozen surrogate: `{payload['surrogate']}`",
        f"- EdgeShard plans retained per DP state: `{payload['edge_shard_plans_per_state']}`",
        f"- HexGen-inspired search: population `{payload['hexgen_population']}`, "
        f"generations `{payload['hexgen_generations']}` per channel",
        f"- JointDNN MILP time limit: `{payload['jointdnn_time_limit_seconds']}` seconds/channel",
        f"- Simulated-annealing steps: `{payload['annealing_steps']}` per channel",
        "",
        "## Results",
        "",
        "| Method | Reward up | Delta vs PPO [95% CI] | log-PPL ratio down | Latency (s) down | Boundaries | Decision ms/channel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["method_order"]:
        row = rows[name]
        lines.append(
            f"| {METHOD_LABELS[name]} | {row['reward_mean']:.6f} | "
            f"{row['paired_reward_difference_vs_ppo_mean']:+.6f} "
            f"[{row['paired_reward_difference_ci95_low']:+.6f}, "
            f"{row['paired_reward_difference_ci95_high']:+.6f}] | "
            f"{row['log_ppl_ratio_mean']:.6f} | "
            f"{row['latency_mean_seconds']:.6f} | {row['boundary_count_mean']:.3f} | "
            f"{row['decision_ms_per_channel']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "EdgeShard-UAV, HexGen-inspired, JointDNN-MUAV, PipeEdge-UAV, Petals-balanced "
            "and Neurosurgeon-inspired are explicit "
            "multi-UAV adaptations of prior system principles, not byte-for-byte reproductions of "
            "their original device/cloud implementations. Method names and paper text must retain "
            "the `-style` or `-inspired` qualification where appropriate.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _plot(output: Path, payload: dict) -> None:
    names = [name for name in payload["method_order"] if name != "random_feasible"]
    labels = [METHOD_LABELS[name] for name in names]
    rewards = [payload["methods"][name]["reward_mean"] for name in names]
    quality = [payload["methods"][name]["log_ppl_ratio_mean"] for name in names]
    latency = [payload["methods"][name]["latency_mean_seconds"] for name in names]
    decision = [payload["methods"][name]["decision_ms_per_channel"] for name in names]
    colors = ["#1f77b4"] + ["#7f7f7f"] * (len(names) - 1)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    panels = (
        (axes[0, 0], rewards, "Surrogate reward (higher is better)", False),
        (axes[0, 1], quality, "Predicted log-PPL ratio (lower is better)", False),
        (axes[1, 0], latency, "Latency in seconds (lower is better)", False),
        (axes[1, 1], decision, "Decision time in ms/channel (log scale)", True),
    )
    for axis, values, xlabel, logarithmic in panels:
        axis.barh(labels[::-1], values[::-1], color=colors[::-1])
        axis.set_xlabel(xlabel)
        if logarithmic:
            axis.set_xscale("log")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(
        "UAV LLM deployment: PPO vs primary system baselines\n"
        "Random-feasible lower bound is reported separately in the result table"
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (
        args.channels < 2
        or args.annealing_steps < 1
        or args.edge_shard_plans_per_state < 1
        or args.hexgen_population < 4
        or args.hexgen_generations < 1
        or args.jointdnn_time_limit <= 0.0
    ):
        raise ValueError("channel count and baseline budgets must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError("output directory is not empty; pass --allow-overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.run_dir / "best_policy.pth"
    if not checkpoint.is_file() or not args.surrogate.is_file():
        raise FileNotFoundError("PPO checkpoint and surrogate checkpoint are required")

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resource_config = resource_config_from_dict(checkpoint_payload["resource_config"])
    run_config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    latency_reference = float(run_config["latency_reference_seconds"])
    surrogate = load_surrogate(args.surrogate, device=torch.device(args.surrogate_device))
    environment = ResourceDeploymentEnvironment(resource_config, surrogate, latency_reference)
    policy, _ = load_layerwise_policy(checkpoint, environment, policy_device="cpu")
    channels = generate_resource_channels(args.channels, args.channel_seed, resource_config)

    generators: dict[str, Callable[[], np.ndarray]] = {
        "ppo_deterministic": lambda: policy.deployments(channels, deterministic=True),
        "edge_shard_uav": lambda: edge_shard_uav_baseline(
            channels, resource_config, plans_per_state=args.edge_shard_plans_per_state
        ),
        "hexgen_inspired": lambda: hexgen_inspired_search_baseline(
            channels,
            resource_config,
            environment,
            population_size=args.hexgen_population,
            generations=args.hexgen_generations,
            seed=args.random_seed + 2,
        ),
        "lingualinked_uav": lambda: lingualinked_uav_baseline(channels, resource_config),
        "jointdnn_multi_uav": lambda: jointdnn_multi_uav_baseline(
            channels,
            resource_config,
            latency_reference,
            time_limit_seconds=args.jointdnn_time_limit,
        ),
        "pipeedge_uav_latency": lambda: pipeedge_uav_latency_baseline(
            channels, resource_config, latency_reference
        ),
        "petals_balanced": lambda: petals_balanced_pipeline_baseline(
            channels, resource_config, latency_reference
        ),
        "neurosurgeon_inspired": lambda: neurosurgeon_best_split_baseline(
            channels, resource_config, latency_reference
        ),
        "simulated_annealing": lambda: surrogate_simulated_annealing_baseline(
            channels,
            resource_config,
            environment,
            steps=args.annealing_steps,
            seed=args.random_seed + 1,
        ),
        "random_feasible": lambda: random_feasible_baseline(
            channels, resource_config, seed=args.random_seed
        ),
    }

    methods: dict[str, dict[str, float]] = {}
    frozen_deployments: dict[str, np.ndarray] = {}
    reward_vectors: dict[str, np.ndarray] = {}
    for name, generate in generators.items():
        started = time.perf_counter()
        deployments = generate()
        decision_seconds = time.perf_counter() - started
        rewards, details = environment.evaluate(channels, deployments)
        row = _metrics(deployments, rewards, details)
        row["decision_seconds_total"] = decision_seconds
        row["decision_ms_per_channel"] = 1000.0 * decision_seconds / args.channels
        methods[name] = row
        reward_vectors[name] = rewards.copy()
        frozen_deployments[name] = deployments.astype(np.int16)
        print(
            f"method={name} reward={row['reward_mean']:.6f} "
            f"latency={row['latency_mean_seconds']:.6f} "
            f"decision_ms={row['decision_ms_per_channel']:.3f}",
            flush=True,
        )

    method_order = list(generators)
    ppo_reward = methods["ppo_deterministic"]["reward_mean"]
    ppo_vector = reward_vectors["ppo_deterministic"]
    for name in method_order:
        difference = reward_vectors[name] - ppo_vector
        standard_error = (
            float(difference.std(ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1
            else 0.0
        )
        methods[name]["reward_gap_to_ppo"] = methods[name]["reward_mean"] - ppo_reward
        methods[name]["paired_reward_difference_vs_ppo_mean"] = float(difference.mean())
        methods[name]["paired_reward_difference_ci95_low"] = float(
            difference.mean() - 1.96 * standard_error
        )
        methods[name]["paired_reward_difference_ci95_high"] = float(
            difference.mean() + 1.96 * standard_error
        )
        methods[name]["paired_win_rate_vs_ppo"] = float(np.mean(difference > 0.0))

    payload = {
        "format_version": 1,
        "stage": "frozen_surrogate_system_baseline_comparison",
        "not_true_model_evaluation": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "surrogate": str(args.surrogate),
        "surrogate_sha256": _sha256(args.surrogate),
        "resource_config": resource_config.to_dict(),
        "latency_reference_seconds": latency_reference,
        "channels": args.channels,
        "channel_seed": args.channel_seed,
        "edge_shard_plans_per_state": args.edge_shard_plans_per_state,
        "hexgen_population": args.hexgen_population,
        "hexgen_generations": args.hexgen_generations,
        "jointdnn_time_limit_seconds": args.jointdnn_time_limit,
        "annealing_steps": args.annealing_steps,
        "random_seed": args.random_seed,
        "method_order": method_order,
        "methods": methods,
        "deployment_archive": str(args.output_dir / "frozen_deployments.npz"),
        "channel_archive": str(args.output_dir / "channels.npy"),
    }
    np.save(args.output_dir / "channels.npy", channels)
    np.savez_compressed(args.output_dir / "frozen_deployments.npz", **frozen_deployments)
    _write_json(args.output_dir / "comparison_summary.json", payload)
    with (args.output_dir / "comparison_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["method", *methods[method_order[0]].keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in method_order:
            writer.writerow({"method": name, **methods[name]})
    _write_report(args.output_dir / "EXPERIMENT_REPORT.md", payload)
    _plot(args.output_dir / "comparison.png", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
