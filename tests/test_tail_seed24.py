'''Tests for the resumable tail seed24 extension plan.'''

from __future__ import annotations

import json
from pathlib import Path

from uav_rl.data.tail_seed24 import TailSeed24Config, build_tail_seed24_plan
from uav_rl.noise_seeds import TRAIN_NOISE_SEED_RANGE, VALIDATION_NOISE_SEED_RANGE


def _action(action_id: str, split: str, source: str, seeds: list[int]) -> dict:
    index = int(action_id.rsplit('-', 1)[-1])
    return {
        'action_id': action_id,
        'split': split,
        'source': source,
        'group_id': f'{split}-{index}',
        'channel': [[float(index + 1)]],
        'deployment': [index % 5],
        'drop_probabilities': [index / 10000.0],
        'latency_seconds': 1.0,
        'noise_seeds': seeds,
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding='utf-8')


def test_seed24_plan_is_unique_resumable_and_reaches_target(tmp_path: Path) -> None:
    v2_actions = [
        _action(f'v2-train-{index}', 'train', 'tail', list(range(index * 4, index * 4 + 4)))
        for index in range(128)
    ]
    v2_actions.extend(
        _action(
            f'v2-validation-{index}',
            'validation',
            'tail',
            list(range(1_000_000_000 + index * 16, 1_000_000_016 + index * 16)),
        )
        for index in range(16)
    )
    v3_actions = [
        _action(
            f'v3-train-{index}',
            'train',
            'tail_hazard',
            list(range(10_000 + index * 8, 10_008 + index * 8)),
        )
        for index in range(512)
    ]
    v3_actions.extend(
        _action(
            f'v3-validation-{index}',
            'validation',
            'tail_boundary',
            list(range(1_100_000_000 + index * 16, 1_100_000_016 + index * 16)),
        )
        for index in range(64)
    )
    seed16_actions = [
        {**action, 'noise_seeds': list(range(100_000 + index * 8, 100_008 + index * 8))}
        for index, action in enumerate(v3_actions[:512])
    ]
    paths = {
        'v2': tmp_path / 'v2.json',
        'v3': tmp_path / 'v3.json',
        'seed16': tmp_path / 'seed16.json',
        'manifest': tmp_path / 'manifest.json',
        'output': tmp_path / 'seed24.json',
    }
    _write(paths['v2'], {'actions': v2_actions})
    _write(paths['v3'], {'actions': v3_actions})
    _write(paths['seed16'], {'actions': seed16_actions})
    _write(
        paths['manifest'],
        {
            'stage': 'development_seed_extension',
            'dataset_fingerprint': 'fixture',
        },
    )
    kwargs = {
        'v2_plan_path': paths['v2'],
        'v3_plan_path': paths['v3'],
        'seed16_plan_path': paths['seed16'],
        'seed16_manifest_path': paths['manifest'],
        'plan_path': paths['output'],
        'config': TailSeed24Config(),
    }
    first = build_tail_seed24_plan(**kwargs)
    second = build_tail_seed24_plan(**kwargs)
    assert first == second
    assert first['audit']['passed'] is True
    assert first['audit']['new_samples'] == 7296
    seeds = [seed for action in first['actions'] for seed in action['noise_seeds']]
    assert len(seeds) == len(set(seeds))
    for action in first['actions']:
        valid_range = (
            TRAIN_NOISE_SEED_RANGE
            if action['split'] == 'train'
            else VALIDATION_NOISE_SEED_RANGE
        )
        assert all(seed in valid_range for seed in action['noise_seeds'])


def test_gate_fallback_prefers_fewer_failed_checks() -> None:
    from uav_rl.tail_expert import _gate_selection_key

    criteria = {
        'maximum_tail_mae': 0.12,
        'minimum_tail_spearman': 0.9,
        'maximum_overall_mae': 0.1,
        'maximum_non_tail_source_regression': 0.02,
    }
    unbalanced = {
        'name': 'unbalanced',
        'tail_metrics': {'mae': 0.1315, 'spearman': 0.929},
        'validation_metrics': {'mae': 0.1480},
        'validation_gate': {
            'passed': False,
            'checks': {
                'tail_mae': False,
                'tail_spearman': True,
                'overall_mae': False,
                'non_tail_regression': False,
            },
            'criteria': criteria,
            'maximum_non_tail_source_mae_regression': 0.0959,
        },
    }
    balanced = {
        'name': 'balanced',
        'tail_metrics': {'mae': 0.1351, 'spearman': 0.923},
        'validation_metrics': {'mae': 0.1149},
        'validation_gate': {
            'passed': False,
            'checks': {
                'tail_mae': False,
                'tail_spearman': True,
                'overall_mae': False,
                'non_tail_regression': True,
            },
            'criteria': criteria,
            'maximum_non_tail_source_mae_regression': 0.0123,
        },
    }

    assert min([unbalanced, balanced], key=_gate_selection_key)['name'] == 'balanced'
