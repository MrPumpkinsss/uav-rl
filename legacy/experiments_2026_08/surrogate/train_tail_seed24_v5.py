'''Train seed24 global and tail-gated ensembles using validation-only selection.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.surrogate_training import EnsembleTrainingConfig
from uav_rl.tail_expert import TailExpertTrainingConfig, train_tail_gated_surrogate
from uav_rl.tail_training import TailValidationCriteria, run_tail_validation_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=250)
    parser.add_argument(
        '--train', type=Path,
        default=Path('artifacts/data/codellama_surrogate_tail_seed24_v5_train.npz'),
    )
    parser.add_argument(
        '--validation', type=Path,
        default=Path(
            'artifacts/data/codellama_surrogate_tail_seed24_v5_validation.npz'
        ),
    )
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'artifacts/data/codellama_surrogate_tail_seed24_v5_manifest.json'
        ),
    )
    parser.add_argument(
        '--baseline-metrics', type=Path,
        default=Path(
            'artifacts/results/ppl_surrogate_multiseed_ensemble_v2_metrics.json'
        ),
    )
    parser.add_argument(
        '--global-checkpoint', type=Path,
        default=Path(
            'artifacts/models/ppl_surrogate_tail_seed24_global_v5_selected.pth'
        ),
    )
    parser.add_argument(
        '--gated-checkpoint', type=Path,
        default=Path(
            'artifacts/models/ppl_surrogate_tail_seed24_gated_expert_v5.pth'
        ),
    )
    parser.add_argument(
        '--summary', type=Path,
        default=Path('artifacts/results/ppl_surrogate_tail_seed24_v5_pipeline.json'),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    criteria = TailValidationCriteria()
    system = SystemConfig()
    global_result = run_tail_validation_ablation(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        baseline_metrics_path=args.baseline_metrics,
        output_model_directory=Path('artifacts/models/tail_seed24_v5_ablation'),
        output_metrics_directory=Path('artifacts/results/tail_seed24_v5_ablation'),
        selected_checkpoint_path=args.global_checkpoint,
        summary_path=Path(
            'artifacts/results/ppl_surrogate_tail_seed24_v5_validation.json'
        ),
        report_path=Path(
            'artifacts/results/ppl_surrogate_tail_seed24_v5_validation_report.md'
        ),
        training_config=EnsembleTrainingConfig(
            epochs=args.epochs,
            patience=args.patience,
        ),
        criteria=criteria,
        system=system,
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    gated_result = train_tail_gated_surrogate(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest_path=args.manifest,
        baseline_metrics_path=args.baseline_metrics,
        base_checkpoint_path=args.global_checkpoint,
        checkpoint_path=args.gated_checkpoint,
        metrics_path=Path(
            'artifacts/results/ppl_surrogate_tail_seed24_gated_expert_v5_metrics.json'
        ),
        report_path=Path(
            'artifacts/results/ppl_surrogate_tail_seed24_gated_expert_v5_report.md'
        ),
        training_config=TailExpertTrainingConfig(),
        criteria=criteria,
        system=system,
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    summary = {
        'format_version': 5,
        'selection_stage': 'validation_only',
        'fresh_final_test_generated': False,
        'ppo_started': False,
        'global': {
            'passed': global_result['passed'],
            'selected_variant': global_result['selected_variant'],
            'checkpoint': global_result['selected_checkpoint'],
        },
        'tail_gated': {
            'passed': gated_result['passed'],
            'checkpoint': gated_result['checkpoint'],
            'overall_mae': gated_result['validation_metrics']['mae'],
            'tail_mae': gated_result['tail_metrics']['mae'],
            'tail_spearman': gated_result['tail_metrics']['spearman'],
            'validation_gate': gated_result['validation_gate'],
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
