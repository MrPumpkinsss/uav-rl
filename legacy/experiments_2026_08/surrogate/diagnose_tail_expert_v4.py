'''Write consumed-data diagnostics for the validation-selected v4 tail expert.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_rl.config import SystemConfig
from uav_rl.tail_expert import diagnose_tail_gated_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('artifacts/models/ppl_surrogate_tail_gated_expert_v4.pth'),
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path(
            'artifacts/data/codellama_surrogate_tail_v3_seed16_manifest.json'
        ),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(
            'artifacts/results/ppl_surrogate_tail_gated_expert_v4_diagnostics.json'
        ),
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=Path(
            'artifacts/results/'
            'ppl_surrogate_tail_gated_expert_v4_diagnostic_report.md'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = diagnose_tail_gated_surrogate(
        checkpoint_path=args.checkpoint,
        seed16_manifest_path=args.manifest,
        output_path=args.output,
        report_path=args.report,
        system=SystemConfig(),
        latency_reference_seconds=1.3077757414751234,
        device_name=args.device,
    )
    concise = {
        'not_a_final_acceptance_test': result['not_a_final_acceptance_test'],
        'checkpoint_sha256': result['checkpoint_sha256'],
        'diagnostics': {
            name: {
                'overall_mae': values['metrics']['mae'],
                'tail_mae': values['tail_metrics']['mae'],
                'tail_spearman': values['tail_metrics']['spearman'],
            }
            for name, values in result['diagnostics'].items()
        },
    }
    print(json.dumps(concise, indent=2), flush=True)


if __name__ == '__main__':
    main()
