'''本脚本构建或恢复用于论文 layer-to-UAV assignment 的真实 PPL 数据集。

脚本不会启动 PPO。每个完成的 action/noise-seed 标签都会先写入 JSONL，
因此即使中断，也可以从 cache 恢复而不重复执行真实 CodeLlama forward。
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import DataGenerationConfig
from uav_rl.data.general_assignment_dataset import (
    GeneralAssignmentDatasetConfig,
    aggregate_general_assignment_cache,
    build_general_assignment_plan,
    collect_general_assignment_labels,
)
from uav_rl.resource_assignment import ResourceConstrainedConfig
from uav_rl.true_quality import TruePPLQualityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # 数据路径、模型配置和输出路径全部显式暴露，确保每次 surrogate 实验可审计。
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--progress-interval', type=int, default=25)
    parser.add_argument(
        '--reference-manifest',
        type=Path,
        default=Path('artifacts/data/codellama_surrogate_tail_seed24_v5_manifest.json'),
    )
    parser.add_argument(
        '--legacy-train',
        type=Path,
        default=Path('artifacts/data/codellama_surrogate_tail_seed24_v5_train.npz'),
    )
    parser.add_argument(
        '--plan',
        type=Path,
        default=Path('artifacts/data/general_assignment_surrogate_plan.json'),
    )
    parser.add_argument(
        '--sample-cache',
        type=Path,
        default=Path('artifacts/cache/general_assignment_surrogate_labels.jsonl'),
    )
    parser.add_argument(
        '--ppl-cache',
        type=Path,
        default=Path('artifacts/cache/general_assignment_surrogate_ppl.jsonl'),
    )
    parser.add_argument('--output-directory', type=Path, default=Path('artifacts/data'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 先解析参数，再构建/加载数据；这样 --help 和 plan-only 都不会触发真实模型推理。
    # reference manifest 定义模型、语料和 clean PPL，必须作为所有 label 的共同基准。
    reference = json.loads(args.reference_manifest.read_text(encoding='utf-8'))
    generation = DataGenerationConfig(**reference['generation'])
    plan = build_general_assignment_plan(
        config=ResourceConstrainedConfig(),
        generation=generation,
        dataset=GeneralAssignmentDatasetConfig(),
        plan_path=args.plan,
        existing_cache_paths=(args.sample_cache,),
    )
    summary = {
        'config_fingerprint': plan['config_fingerprint'],
        'actions': {
            split: sum(action['split'] == split for action in plan['actions'])
            for split in ('train', 'validation', 'test')
        },
        'true_ppl_forwards': sum(len(action['noise_seeds']) for action in plan['actions']),
    }
    # plan-only 只生成并检查采样计划，不加载 LLM，便于先审阅预计的计算量。
    if args.plan_only:
        print(json.dumps(summary, indent=2), flush=True)
        return

    # 真正进入标签采集后才加载 LLM；evaluator 自带 JSONL cache，可从中断处恢复。
    evaluator = TruePPLQualityEvaluator(
        generation,
        device_name=args.device,
        cache_path=args.ppl_cache,
        progress_interval=args.progress_interval,
    )
    expected = reference['quality_evaluator']
    if abs(evaluator.clean_perplexity - float(expected['clean_perplexity'])) > 1e-5:
        raise ValueError('clean PPL changed from the reference labels')
    if evaluator.evaluated_sequences != int(expected['evaluated_sequences']):
        raise ValueError('evaluated sequence count changed from the reference labels')
    if evaluator.evaluated_tokens != int(expected['evaluated_tokens']):
        raise ValueError('evaluated token count changed from the reference labels')

    collection = collect_general_assignment_labels(
        plan, evaluator, args.sample_cache, progress_interval=args.progress_interval
    )
    outputs = aggregate_general_assignment_cache(
        plan,
        args.sample_cache,
        args.output_directory,
        legacy_train_path=args.legacy_train,
        quality_metadata=evaluator.metadata(),
    )
    print(json.dumps({**summary, 'collection': collection, 'outputs': outputs}, default=str, indent=2))


if __name__ == '__main__':
    main()
