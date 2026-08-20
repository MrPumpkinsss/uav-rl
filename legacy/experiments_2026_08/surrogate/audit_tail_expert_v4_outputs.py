'''Audit v4 tail-expert provenance, consistency, and stop conditions.'''

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_MEMBERS = 5
FINAL_ARTIFACTS = (
    Path('artifacts/data/codellama_surrogate_tail_v3_final_test_plan.json'),
    Path('artifacts/data/codellama_surrogate_tail_v3_test.npz'),
    Path('artifacts/data/codellama_surrogate_tail_v3_final_manifest.json'),
    Path('artifacts/cache/surrogate_tail_v3_final_test.jsonl'),
    Path('artifacts/cache/surrogate_tail_v3_final_test_ppl.jsonl'),
    Path('artifacts/results/ppl_surrogate_tail_ensemble_v3_metrics.json'),
    Path('artifacts/results/ppl_surrogate_tail_ensemble_v3_report.md'),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def audit_outputs(
    *,
    manifest_path: Path,
    base_checkpoint_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    diagnostic_path: Path,
) -> dict[str, Any]:
    '''Return deterministic checks for the failed v4 validation gate.'''

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_hash = _sha256(checkpoint_path)
    manifest_hash = manifest.get('dataset_fingerprint') or _sha256(manifest_path)
    base_hash = _sha256(base_checkpoint_path)
    data_files = payload.get('data_files', {})

    checks: dict[str, bool] = {
        'checkpoint_format_v4': payload.get('format_version') == 4,
        'checkpoint_model_type': payload.get('model_type')
        == 'tail_gated_ensemble',
        'checkpoint_validation_only': payload.get('selection_stage')
        == 'validation_only',
        'base_has_five_members': len(payload.get('base_model_states', []))
        == EXPECTED_MEMBERS,
        'expert_has_five_members': len(payload.get('expert_model_states', []))
        == EXPECTED_MEMBERS,
        'member_metrics_has_five_entries': len(payload.get('member_metrics', []))
        == EXPECTED_MEMBERS,
        'base_checkpoint_hash_matches': payload.get('base_checkpoint', {}).get(
            'sha256'
        )
        == base_hash,
        'manifest_fingerprint_matches': payload.get('data_manifest_hash')
        == manifest_hash,
        'metrics_checkpoint_hash_matches': metrics.get('checkpoint_sha256')
        == checkpoint_hash,
        'metrics_manifest_fingerprint_matches': metrics.get('data_manifest_hash')
        == manifest_hash,
        'metrics_gate_failed': metrics.get('passed') is False,
        'checkpoint_gate_failed': payload.get('validation_gate', {}).get('passed')
        is False,
        'gate_parameters_match': payload.get('gate') == metrics.get('selected_gate'),
        'validation_metrics_match': _same(
            payload.get('validation_metrics', {}),
            metrics.get('validation_metrics', {}),
        ),
        'tail_metrics_match': _same(
            payload.get('tail_metrics', {}), metrics.get('tail_metrics', {})
        ),
        'validation_gate_matches': _same(
            payload.get('validation_gate', {}), metrics.get('validation_gate', {})
        ),
        'diagnostic_is_consumed_only': diagnostic.get(
            'not_a_final_acceptance_test'
        )
        is True,
        'diagnostic_checkpoint_hash_matches': diagnostic.get('checkpoint_sha256')
        == checkpoint_hash,
        'no_final_test_artifacts': not any(path.exists() for path in FINAL_ARTIFACTS),
    }

    split_hashes: dict[str, str] = {}
    for split in ('train', 'validation'):
        entry = data_files.get(split, {})
        path = Path(str(entry.get('path', '')))
        exists = path.is_file()
        actual_hash = _sha256(path) if exists else ''
        split_hashes[split] = actual_hash
        checks[f'{split}_file_exists'] = exists
        checks[f'{split}_checkpoint_hash_matches'] = (
            exists and entry.get('sha256') == actual_hash
        )
        checks[f'{split}_manifest_hash_matches'] = (
            exists
            and manifest.get('splits', {}).get(split, {}).get('sha256')
            == actual_hash
        )

    return {
        'format_version': 4,
        'stage': 'development_audit',
        'passed': all(checks.values()),
        'checks': checks,
        'input_sha256': {
            'manifest': _sha256(manifest_path),
            'base_checkpoint': base_hash,
            'checkpoint': checkpoint_hash,
            'metrics': _sha256(metrics_path),
            'diagnostic': _sha256(diagnostic_path),
            **{f'{split}_split': value for split, value in split_hashes.items()},
        },
        'validation_gate': metrics.get('validation_gate'),
        'final_artifacts_checked': [str(path) for path in FINAL_ARTIFACTS],
        'notes': [
            'Process liveness is checked separately because process IDs are not '
            'stable artifact metadata.',
            'The failed gate intentionally prevents final-test generation and PPO.',
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path(
            'artifacts/data/codellama_surrogate_tail_v3_seed16_manifest.json'
        ),
    )
    parser.add_argument(
        '--base-checkpoint',
        type=Path,
        default=Path(
            'artifacts/models/ppl_surrogate_tail_seed16_ensemble_v3_selected.pth'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('artifacts/models/ppl_surrogate_tail_gated_expert_v4.pth'),
    )
    parser.add_argument(
        '--metrics',
        type=Path,
        default=Path(
            'artifacts/results/ppl_surrogate_tail_gated_expert_v4_metrics.json'
        ),
    )
    parser.add_argument(
        '--diagnostic',
        type=Path,
        default=Path(
            'artifacts/results/ppl_surrogate_tail_gated_expert_v4_diagnostics.json'
        ),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(
            'artifacts/results/ppl_surrogate_tail_gated_expert_v4_audit.json'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_outputs(
        manifest_path=args.manifest,
        base_checkpoint_path=args.base_checkpoint,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        diagnostic_path=args.diagnostic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
