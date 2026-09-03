'''只为 surrogate 训练采集高 boundary 区域的额外真实 PPL 标签。'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from uav_rl.config import DataGenerationConfig
from uav_rl.data.general_assignment_dataset import _append_jsonl, _existing_seeds, _fresh_seeds, sample_general_assignment
from uav_rl.noise_seeds import TRAIN_NOISE_SEED_RANGE
from uav_rl.resource_assignment import ResourceConstrainedConfig, layerwise_drop_probabilities, layerwise_latency
from uav_rl.resource_environment import generate_resource_channels
from uav_rl.true_quality import TruePPLQualityEvaluator


def _sha256(path: Path) -> str:
    """计算文件 SHA256，用于确认输入文件未被替换。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    """解析命令行参数，构造本次实验的运行配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据路径、模型配置和输出路径全部显式暴露，确保每次 surrogate 实验可审计。
    parser.add_argument('--actions', type=int, default=128)
    parser.add_argument('--noise-samples', type=int, default=8)
    parser.add_argument('--action-seed', type=int, default=20260830)
    parser.add_argument('--noise-seed', type=int, default=2026083001)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--progress-interval', type=int, default=25)
    parser.add_argument('--reference-manifest', type=Path, default=Path('artifacts/data/general_assignment_manifest.json'))
    parser.add_argument('--existing-cache', type=Path, default=Path('artifacts/cache/general_assignment_surrogate_labels.jsonl'))
    parser.add_argument('--ppl-cache', type=Path, default=Path('artifacts/cache/general_assignment_surrogate_ppl.jsonl'))
    parser.add_argument('--sample-cache', type=Path, default=Path('artifacts/cache/general_assignment_high_boundary_augmented.jsonl'))
    parser.add_argument('--plan', type=Path, default=Path('artifacts/data/general_assignment_high_boundary_augmentation_plan.json'))
    parser.add_argument('--output-train', type=Path, default=Path('artifacts/data/general_assignment_train_high_augmented.npz'))
    parser.add_argument('--output-manifest', type=Path, default=Path('artifacts/data/general_assignment_high_augmented_manifest.json'))
    return parser.parse_args()


def _build_plan(args: argparse.Namespace, config: ResourceConstrainedConfig, generation: DataGenerationConfig) -> dict:
    """根据已有数据集和增强参数构造新的采样计划。"""
    rng = np.random.default_rng(args.action_seed)
    channels = generate_resource_channels(args.actions, args.action_seed, config)
    actions = []
    for index, channel in enumerate(channels):
        deployment = sample_general_assignment(
            rng,
            channel,
            config,
            target_boundaries=int(rng.integers(9, 14)),
            max_attempts=20_000,
        )
        drops = layerwise_drop_probabilities(deployment, channel, config)
        actions.append({
            'action_id': f'high-augmentation-train-{index:04d}',
            'split': 'train',
            'source': 'high_boundary_augmentation',
            'group_id': f'high-augmentation-train-{index:04d}',
            'channel': channel.tolist(),
            'deployment': deployment.tolist(),
            'drop_probabilities': drops.tolist(),
            'latency_seconds': float(layerwise_latency(deployment, channel, config).total_seconds),
            'noise_seeds': [],
        })
    excluded = _existing_seeds((args.existing_cache, args.sample_cache))
    seeds = _fresh_seeds(np.random.default_rng(args.noise_seed), TRAIN_NOISE_SEED_RANGE, args.actions * args.noise_samples, excluded)
    for action, row in zip(actions, seeds.reshape(args.actions, args.noise_samples), strict=True):
        action['noise_seeds'] = row.tolist()
    payload = {
        'format_version': 1,
        'purpose': 'high_boundary_surrogate_train_augmentation',
        'resource_config': config.to_dict(),
        'generation': generation.__dict__,
        'actions': actions,
    }
    payload['config_fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return payload


def main() -> None:
    """组织当前脚本的完整实验流程，包括加载、训练或评估和结果保存。"""
    args = parse_args()
    # 先解析参数，再构建/加载数据；这样 --help 和 plan-only 都不会触发真实模型推理。
    if min(args.actions, args.noise_samples, args.progress_interval) < 1:
        raise ValueError('actions, noise samples, and progress interval must be positive')
    # 增广数据必须沿用原数据集的 generation 配置，避免模型或语料不一致。
    reference = json.loads(args.reference_manifest.read_text(encoding='utf-8'))
    generation = DataGenerationConfig(**reference['generation'])
    config = ResourceConstrainedConfig()
    # 已有 plan 必须验证 resource config；否则不能把旧计划混入当前数据集。
    if args.plan.exists():
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        if plan.get('actions') is None or plan.get('resource_config') != config.to_dict():
            raise ValueError('existing high-boundary plan is incompatible with the requested resource config')
    else:
        plan = _build_plan(args, config, generation)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    args.sample_cache.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.sample_cache.exists():
        for line in args.sample_cache.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row['action_id'], int(row['noise_seed'])))
    # evaluator cache 负责去重相同 drop vector/noise seed，避免重复昂贵的真实 PPL forward。
    evaluator = TruePPLQualityEvaluator(generation, device_name=args.device, cache_path=args.ppl_cache, progress_interval=args.progress_interval)
    if abs(evaluator.clean_perplexity - float(reference['quality_evaluator']['clean_perplexity'])) > 1e-5:
        raise ValueError('clean PPL changed from the reference dataset')
    expected = sum(len(a['noise_seeds']) for a in plan['actions'])
    newly = 0
    # 逐 action、逐 noise seed 写入 JSONL，进程被中断时只需继续未完成的组合。
    for action in plan['actions']:
        probabilities = np.asarray(action['drop_probabilities'], dtype=np.float32)
        for seed in action['noise_seeds']:
            key = (action['action_id'], int(seed))
            if key in completed:
                continue
            started = time.perf_counter()
            quality = float(evaluator.evaluate(probabilities[None, :], noise_seeds=np.asarray([seed], dtype=np.int64))[0])
            _append_jsonl(args.sample_cache, {
                'format_version': 1,
                'config_fingerprint': plan['config_fingerprint'],
                **action,
                'noise_seed': int(seed),
                'perplexity': evaluator.clean_perplexity * math.exp(quality),
                'log_ppl_ratio': quality,
                'evaluation_seconds': time.perf_counter() - started,
            })
            completed.add(key)
            newly += 1
            if newly % args.progress_interval == 0:
                print(f'high_boundary_samples={len(completed)}/{expected}', flush=True)
    with np.load('artifacts/data/general_assignment_train.npz') as old:
        base = {key: np.array(old[key]) for key in old.files}
    grouped = {}
    for line in args.sample_cache.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row['action_id'], []).append(row)
    rows = []
    for action in plan['actions']:
        labels = np.asarray([row['log_ppl_ratio'] for row in grouped[action['action_id']]], dtype=np.float32)
        rows.append({
            'action_ids': action['action_id'], 'sample_source': action['source'], 'group_ids': action['group_id'],
            'channels': np.asarray(action['channel'], dtype=np.float32), 'deployments': np.asarray(action['deployment'], dtype=np.int64),
            'drop_probabilities': np.asarray(action['drop_probabilities'], dtype=np.float32), 'latency_seconds': np.float32(action['latency_seconds']),
            'log_ppl_ratio': np.float32(labels.mean()), 'log_ppl_ratio_std': np.float32(labels.std()), 'noise_seed_count': np.int32(len(labels)), 'has_context': np.bool_(True),
        })
    augmented = {key: np.concatenate([base[key], np.asarray([row[key] for row in rows])], axis=0) for key in base}
    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_train, **augmented)
    manifest = json.loads(args.reference_manifest.read_text(encoding='utf-8'))
    manifest['splits']['train']['path'] = str(args.output_train)
    manifest['splits']['train']['actions'] = int(len(augmented['action_ids']))
    manifest['splits']['train']['sha256'] = _sha256(args.output_train)
    manifest['quality_evaluator'] = evaluator.metadata()
    manifest['augmentation'] = {'source': str(args.sample_cache), 'actions': args.actions, 'noise_samples': args.noise_samples, 'completed': len(completed), 'config_fingerprint': plan['config_fingerprint']}
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'expected': expected, 'completed': len(completed), 'new': newly, 'output_train': str(args.output_train), 'output_manifest': str(args.output_manifest)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
