'''训练任意 layer-to-UAV PPO，并评估 surrogate 排名的 Top-K 候选。

训练阶段只使用冻结的 general-assignment surrogate，不在 PPO 更新过程中调用真实 LLM。
训练完成后，脚本对 held-out channel 生成多个可行 deployment，用 surrogate 排序，
再分别评估 surrogate-selected Top-1 和 Top-K 候选中的 true-model oracle。
'''

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from uav_rl.config import DataGenerationConfig, PPOConfig, SystemConfig
from uav_rl.experiment import estimate_latency_reference
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.data.general_assignment_dataset import sample_general_assignment
from uav_rl.resource_environment import ResourceDeploymentEnvironment, generate_resource_channels
from uav_rl.rl.layerwise_ppo import LayerwisePPOTrainer
from uav_rl.surrogate import load_surrogate
from uav_rl.true_quality import TruePPLQualityEvaluator


def _sha256(path: Path) -> str:
    """计算 surrogate 文件摘要，写入 run manifest 以支持之后复现实验。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/runs/surrogate_ppo/layerwise_topk'), help='当前 PPO run 的输出目录。')
    parser.add_argument('--surrogate', type=Path, default=Path('artifacts/models/ppl_surrogate_general_assignment_ensemble.pth'), help='冻结的 surrogate checkpoint。')
    parser.add_argument('--episodes', type=int, default=1000, help='PPO 训练 episode 总数；配合 --resume 时表示目标总数。')
    parser.add_argument('--rollout-size', type=int, default=128)
    parser.add_argument('--checkpoint-interval-episodes', type=int, default=500)
    parser.add_argument('--validation-interval', type=int, default=4)
    parser.add_argument('--validation-channels', type=int, default=32)
    parser.add_argument('--top-k', type=int, default=5, help='训练后保留的 PPO 候选数量。')
    parser.add_argument('--candidate-samples', type=int, default=20, help='每个信道生成候选时使用的 beam 宽度。')
    parser.add_argument('--teacher-channels', type=int, default=256)
    parser.add_argument('--behavior-cloning-epochs', type=int, default=20)
    parser.add_argument('--teacher-candidates', type=int, default=24)
    parser.add_argument('--true-channels', type=int, default=32)
    parser.add_argument('--true-noise-samples', type=int, default=4)
    parser.add_argument('--device', default='cuda', help='surrogate 和真实 LLM 使用的设备。')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    """原子写入小型实验记录，避免训练中断时损坏 run_config 或结果 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(args.episodes, args.rollout_size, args.checkpoint_interval_episodes, args.top_k, args.true_channels, args.true_noise_samples) < 1:
        raise ValueError('episode, rollout, checkpoint, Top-K, and validation values must be positive')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / 'training_state.pth'
    output_path = args.run_dir / 'best_policy.pth'
    candidate_directory = args.run_dir / 'candidate_policies'
    launch_path = args.run_dir / 'run_config.json'
    # surrogate 在整个训练期间保持冻结；PPO 只更新 policy，不更新质量模型。
    surrogate = load_surrogate(args.surrogate, device=torch.device(args.device))
    system = SystemConfig()
    resource_config = ResourceConstrainedConfig(system=system)
    latency_reference = estimate_latency_reference(system, 20260819)
    environment = ResourceDeploymentEnvironment(resource_config, surrogate, latency_reference)

    def teacher_provider(channels: np.ndarray) -> np.ndarray:
        """用有限候选 surrogate 搜索为 behavior cloning 生成教师 deployment。"""
        # 固定随机种子，保证教师数据不会因 Python 进程状态变化而漂移。
        rng = np.random.default_rng(20260825)
        deployments = []
        for channel in channels:
            # 每个 channel 独立采样一批满足资源约束的 assignment。
            candidates = []
            for _ in range(args.teacher_candidates):
                candidates.append(
                    sample_general_assignment(
                        rng,
                        channel,
                        resource_config,
                        target_boundaries=int(rng.integers(2, 5)),
                        max_attempts=20_000,
                    )
                )
            candidate_array = np.stack(candidates)
            repeated = np.repeat(channel[None, :, :], len(candidate_array), axis=0)
            # 教师只在候选集中选择 surrogate reward 最高者，不使用真实 PPL。
            rewards, _ = environment.evaluate(repeated, candidate_array)
            deployments.append(candidate_array[int(np.argmax(rewards))])
        return np.stack(deployments)
    config = replace(
        PPOConfig(system=system),
        training_episodes=args.episodes,
        rollout_size=args.rollout_size,
        teacher_channels=args.teacher_channels,
        behavior_cloning_epochs=args.behavior_cloning_epochs,
        validation_channels=args.validation_channels,
        validation_interval=args.validation_interval,
        training_noise_samples=4,
    )
    metadata = {
        'quality_backend': 'frozen_general_assignment_surrogate',
        'surrogate_checkpoint': str(args.surrogate),
        'surrogate_checkpoint_sha256': _sha256(args.surrogate),
        'top_k': args.top_k,
        'candidate_samples': args.candidate_samples,
        'teacher_channels': args.teacher_channels,
        'behavior_cloning_epochs': args.behavior_cloning_epochs,
        'teacher_candidates': args.teacher_candidates,
        'max_policy_boundaries': 4,
        'resource_config': resource_config.to_dict(),
    }
    _write_json(
        launch_path,
        {
            'format_version': 1,
            'policy_type': 'layerwise_general_assignment',
            'ppo_config': asdict(config),
            'run_metadata': metadata,
            'latency_reference_seconds': latency_reference,
        },
    )
    trainer = LayerwisePPOTrainer(
        config,
        resource_config,
        environment,
        teacher_action_provider=teacher_provider,
        max_policy_boundaries=4,
    )
    history = trainer.train(
        output_path,
        state_path=state_path,
        run_metadata=metadata,
        candidate_directory=candidate_directory,
        candidate_interval_episodes=args.checkpoint_interval_episodes,
        resume=args.resume,
    )

    # held-out channel 只用于训练后候选分析，不能与 teacher/training channels 混用。
    channels = generate_resource_channels(args.true_channels, 20260824, resource_config)
    topk_deployments, surrogate_rewards = trainer.top_k_deployments(
        channels, k=args.top_k, samples_per_channel=args.candidate_samples
    )
    generation = DataGenerationConfig()
    true_cache = args.run_dir / 'topk_true_ppl_cache.jsonl'
    evaluator = TruePPLQualityEvaluator(generation, device_name=args.device, cache_path=true_cache)
    true_environment = ResourceDeploymentEnvironment(resource_config, evaluator, latency_reference)
    # 将 (N, K, layers) 展平成 (N*K, layers)，便于复用统一的 environment API。
    flat_channels = np.repeat(channels, args.top_k, axis=0)
    flat_deployments = topk_deployments.reshape(args.true_channels * args.top_k, system.num_layers)
    noise_seeds = np.arange(4_100_000, 4_100_000 + args.true_noise_samples, dtype=np.int64)
    true_rewards, details = true_environment.evaluate(flat_channels, flat_deployments, noise_seeds=noise_seeds)
    # 恢复 channel/candidate 两个维度后，才能分别计算 Top-1 与 true Top-K oracle。
    true_rewards = true_rewards.reshape(args.true_channels, args.top_k)
    surrogate_top1 = true_rewards[:, 0]
    true_topk_oracle = true_rewards.max(axis=1)
    deterministic = trainer.deployments(channels, deterministic=True)
    deterministic_rewards, _ = true_environment.evaluate(channels, deterministic, noise_seeds=noise_seeds)
    result = {
        'format_version': 1,
        'run_dir': str(args.run_dir),
        'surrogate_checkpoint': str(args.surrogate),
        'training_target_episodes': args.episodes,
        'completed_history_points': len(history['reward']),
        'top_k': args.top_k,
        'candidate_samples': args.candidate_samples,
        'true_channels': args.true_channels,
        'true_noise_samples': args.true_noise_samples,
        'surrogate_top1_true_reward_mean': float(surrogate_top1.mean()),
        'topk_true_oracle_reward_mean': float(true_topk_oracle.mean()),
        'deterministic_true_reward_mean': float(deterministic_rewards.mean()),
        'topk_gain_over_deterministic': float(true_topk_oracle.mean() - deterministic_rewards.mean()),
        'top1_gain_over_deterministic': float(surrogate_top1.mean() - deterministic_rewards.mean()),
        'topk_selection_gap': float(true_topk_oracle.mean() - surrogate_top1.mean()),
        'invalid_fraction': float(details['invalid'].mean()),
        'surrogate_reward_mean_by_rank': surrogate_rewards.mean(axis=0).tolist(),
        'true_reward_mean_by_rank': true_rewards.mean(axis=0).tolist(),
        'true_reward_p10_by_rank': np.quantile(true_rewards, 0.10, axis=0).tolist(),
        'true_reward_p90_by_rank': np.quantile(true_rewards, 0.90, axis=0).tolist(),
        'evaluator': evaluator.metadata(),
    }
    _write_json(args.run_dir / 'topk_true_validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
