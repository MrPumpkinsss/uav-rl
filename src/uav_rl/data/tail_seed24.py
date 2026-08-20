'''Resumable tail-only seed extension from mixed 4/16-seed labels to 24.'''

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_rl.data.surrogate_dataset import (
    _aggregate_fresh_actions,
    _atomic_write_json,
    _load_jsonl,
    _write_split_dataset,
    canonical_json_hash,
)
from uav_rl.data.tail_dataset import _sample_unique_excluding
from uav_rl.noise_seeds import TRAIN_NOISE_SEED_RANGE, VALIDATION_NOISE_SEED_RANGE


@dataclass(frozen=True)
class TailSeed24Config:
    '''Raise every tail train/validation action to 24 independent seeds.'''

    training_noise_seed: int = 2026081701
    validation_noise_seed: int = 2026081702
    target_noise_samples: int = 24

    def __post_init__(self) -> None:
        if self.target_noise_samples < 17:
            raise ValueError('target must exceed the existing 16-seed labels')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_tail(action: dict[str, Any]) -> bool:
    source = str(action['source'])
    return source == 'tail' or source.startswith('tail_')


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _identity(action: dict[str, Any]) -> dict[str, Any]:
    fields = (
        'action_id',
        'split',
        'source',
        'group_id',
        'channel',
        'deployment',
        'drop_probabilities',
        'latency_seconds',
    )
    return {field: action[field] for field in fields}


def _full_tail_actions(
    v2_plan: dict[str, Any],
    v3_plan: dict[str, Any],
    seed16_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    seed16_by_id = {
        str(action['action_id']): action for action in seed16_plan['actions']
    }
    actions: list[dict[str, Any]] = []
    for action in v2_plan['actions']:
        if action['split'] in ('train', 'validation') and _is_tail(action):
            actions.append({**_identity(action), 'noise_seeds': action['noise_seeds']})
    for action in v3_plan['actions']:
        if action['split'] not in ('train', 'validation'):
            continue
        seeds = list(action['noise_seeds'])
        if action['split'] == 'train':
            action_id = str(action['action_id'])
            extension = seed16_by_id.get(action_id)
            if extension is None:
                raise ValueError(f'missing seed16 action {action_id}')
            seeds.extend(extension['noise_seeds'])
        actions.append({**_identity(action), 'noise_seeds': seeds})
    if len({str(action['action_id']) for action in actions}) != len(actions):
        raise ValueError('tail seed24 inputs contain duplicate action ids')
    return actions


def build_tail_seed24_plan(
    *,
    v2_plan_path: Path,
    v3_plan_path: Path,
    seed16_plan_path: Path,
    seed16_manifest_path: Path,
    plan_path: Path,
    config: TailSeed24Config,
) -> dict[str, Any]:
    '''Create or validate an immutable plan containing only new seed evaluations.'''

    v2_plan = _load(v2_plan_path)
    v3_plan = _load(v3_plan_path)
    seed16_plan = _load(seed16_plan_path)
    seed16_manifest = _load(seed16_manifest_path)
    if seed16_manifest.get('stage') != 'development_seed_extension':
        raise ValueError('seed24 requires the completed seed16 development manifest')
    full_actions = _full_tail_actions(v2_plan, v3_plan, seed16_plan)
    excluded = {
        int(seed)
        for plan in (v2_plan, v3_plan, seed16_plan)
        for action in plan['actions']
        for seed in action['noise_seeds']
    }
    metadata = {
        'format_version': 5,
        'stage': 'tail_seed24_extension',
        'config': asdict(config),
        'parents': {
            'v2_plan': {'path': str(v2_plan_path), 'sha256': _sha256(v2_plan_path)},
            'v3_plan': {'path': str(v3_plan_path), 'sha256': _sha256(v3_plan_path)},
            'seed16_plan': {
                'path': str(seed16_plan_path),
                'sha256': _sha256(seed16_plan_path),
            },
            'seed16_manifest': {
                'path': str(seed16_manifest_path),
                'sha256': _sha256(seed16_manifest_path),
                'dataset_fingerprint': seed16_manifest['dataset_fingerprint'],
            },
        },
    }
    fingerprint = canonical_json_hash(metadata)
    if plan_path.exists():
        existing = _load(plan_path)
        if existing.get('config_fingerprint') != fingerprint:
            raise ValueError('existing tail seed24 plan is incompatible')
        return existing

    new_actions: list[dict[str, Any]] = []
    for split, rng_seed, seed_range in (
        ('train', config.training_noise_seed, TRAIN_NOISE_SEED_RANGE),
        ('validation', config.validation_noise_seed, VALIDATION_NOISE_SEED_RANGE),
    ):
        split_actions = [action for action in full_actions if action['split'] == split]
        rng = np.random.default_rng(rng_seed)
        for existing_count in sorted(
            {len(action['noise_seeds']) for action in split_actions}
        ):
            group = [
                action
                for action in split_actions
                if len(action['noise_seeds']) == existing_count
            ]
            additional = config.target_noise_samples - existing_count
            if additional < 1:
                raise ValueError('seed24 parent already meets or exceeds the target')
            sampled = _sample_unique_excluding(
                rng, seed_range, (len(group), additional), excluded
            )
            excluded.update(int(seed) for seed in sampled.reshape(-1))
            for action, seeds in zip(group, sampled, strict=True):
                new_actions.append(
                    {**_identity(action), 'noise_seeds': seeds.astype(np.int64).tolist()}
                )

    new_seeds = [
        int(seed) for action in new_actions for seed in action['noise_seeds']
    ]
    full_by_id = {str(action['action_id']): action for action in full_actions}
    train_actions = sum(action['split'] == 'train' for action in new_actions)
    validation_actions = sum(
        action['split'] == 'validation' for action in new_actions
    )
    audit = {
        'train_actions': train_actions,
        'validation_actions': validation_actions,
        'new_samples': len(new_seeds),
        'new_seeds_unique': len(new_seeds) == len(set(new_seeds)),
        'target_seed_count': all(
            len(full_by_id[str(action['action_id'])]['noise_seeds'])
            + len(action['noise_seeds'])
            == config.target_noise_samples
            for action in new_actions
        ),
        'expected_action_counts': train_actions == 640 and validation_actions == 80,
    }
    audit['passed'] = all(
        audit[key]
        for key in ('new_seeds_unique', 'target_seed_count', 'expected_action_counts')
    )
    if not audit['passed']:
        raise ValueError('new tail seed24 plan failed its audit')
    payload = {
        **metadata,
        'config_fingerprint': fingerprint,
        'audit': audit,
        'actions': new_actions,
    }
    _atomic_write_json(plan_path, payload)
    return payload


def _write_combined_summary(
    *, base_path: Path, tail_path: Path, output_path: Path
) -> dict[str, Any]:
    fields = (
        'action_ids',
        'sample_source',
        'group_ids',
        'channels',
        'deployments',
        'drop_probabilities',
        'latency_seconds',
        'log_ppl_ratio',
        'log_ppl_ratio_std',
        'noise_seed_count',
        'has_context',
    )
    with np.load(base_path, allow_pickle=False) as base_data:
        base = {field: np.asarray(base_data[field]) for field in fields}
    with np.load(tail_path, allow_pickle=False) as tail_data:
        tail = {field: np.asarray(tail_data[field]) for field in fields}
    base_sources = base['sample_source'].astype(str)
    keep = np.asarray(
        [source != 'tail' and not source.startswith('tail_') for source in base_sources]
    )
    payload = {
        field: np.concatenate([base[field][keep], tail[field]], axis=0)
        for field in fields
    }
    ids = payload['action_ids'].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError('combined seed24 summary contains duplicate action ids')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        'path': str(output_path),
        'sha256': _sha256(output_path),
        'actions': len(ids),
        'sources': {
            source: int(np.sum(payload['sample_source'].astype(str) == source))
            for source in sorted(set(payload['sample_source'].astype(str)))
        },
        'noise_seed_count_min': int(payload['noise_seed_count'].min()),
        'noise_seed_count_max': int(payload['noise_seed_count'].max()),
    }


def _split_isolation(actions: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, dict[str, set[Any]]] = {}
    for split in ('train', 'validation'):
        rows = [action for action in actions if action['split'] == split]
        fields[split] = {
            'noise_seed': {
                int(seed) for action in rows for seed in action['noise_seeds']
            },
            'drop_vector': {
                np.asarray(action['drop_probabilities'], dtype='<f4').tobytes()
                for action in rows
            },
            'channel_deployment_pair': {
                (
                    np.asarray(action['channel'], dtype='<f4').tobytes(),
                    np.asarray(action['deployment'], dtype='<i8').tobytes(),
                )
                for action in rows
            },
        }
    overlap = {
        name: len(fields['train'][name] & fields['validation'][name])
        for name in fields['train']
    }
    return {'overlap_counts': overlap, 'passed': all(value == 0 for value in overlap.values())}


def aggregate_tail_seed24_dataset(
    *,
    v2_plan_path: Path,
    v3_plan_path: Path,
    seed16_plan_path: Path,
    seed16_manifest_path: Path,
    extension_plan_path: Path,
    v2_cache_path: Path,
    v3_cache_path: Path,
    seed16_cache_path: Path,
    extension_cache_path: Path,
    output_directory: Path,
    quality_evaluator_metadata: dict[str, Any],
) -> dict[str, Any]:
    '''Aggregate existing and new records into immutable seed24 train/validation.'''

    v2_plan = _load(v2_plan_path)
    v3_plan = _load(v3_plan_path)
    seed16_plan = _load(seed16_plan_path)
    parent = _load(seed16_manifest_path)
    extension_plan = _load(extension_plan_path)
    if extension_plan['parents']['seed16_manifest']['sha256'] != _sha256(
        seed16_manifest_path
    ):
        raise ValueError('seed16 manifest changed after seed24 planning')

    existing = _full_tail_actions(v2_plan, v3_plan, seed16_plan)
    extension_by_id = {
        str(action['action_id']): action for action in extension_plan['actions']
    }
    combined: list[dict[str, Any]] = []
    for action in existing:
        action_id = str(action['action_id'])
        extension = extension_by_id.get(action_id)
        if extension is None:
            raise ValueError(f'missing seed24 extension for {action_id}')
        seeds = [
            *(int(seed) for seed in action['noise_seeds']),
            *(int(seed) for seed in extension['noise_seeds']),
        ]
        if len(seeds) != 24 or len(seeds) != len(set(seeds)):
            raise ValueError(f'invalid seed24 aggregate for {action_id}')
        combined.append({**_identity(action), 'noise_seeds': seeds})
    if len(extension_by_id) != len(combined):
        raise ValueError('seed24 plan contains unexpected action ids')

    records = []
    for path in (
        v2_cache_path,
        v3_cache_path,
        seed16_cache_path,
        extension_cache_path,
    ):
        records.extend(_load_jsonl(path))
    record_keys = [
        (str(record['action_id']), int(record['noise_seed'])) for record in records
    ]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError('seed24 parent caches contain duplicate action/seed records')
    rows = _aggregate_fresh_actions({'actions': combined}, records)
    if len(rows['train']) != 640 or len(rows['validation']) != 80:
        raise RuntimeError('seed24 aggregation produced unexpected tail action counts')
    isolation = _split_isolation(combined)
    if not isolation['passed']:
        raise ValueError('seed24 train/validation actions are not isolated')

    train_tail_path = output_directory / 'codellama_surrogate_tail_seed24_v5_train_tail.npz'
    validation_tail_path = (
        output_directory / 'codellama_surrogate_tail_seed24_v5_validation_tail.npz'
    )
    train_tail = _write_split_dataset(train_tail_path, rows['train'])
    validation_tail = _write_split_dataset(validation_tail_path, rows['validation'])
    if train_tail['noise_seeds_per_action'] != 24:
        raise ValueError('seed24 train tail shard did not reach 24 seeds/action')
    if validation_tail['noise_seeds_per_action'] != 24:
        raise ValueError('seed24 validation tail shard did not reach 24 seeds/action')

    v2_train = Path(parent['base_v2']['splits']['train']['path'])
    v2_validation = Path(parent['base_v2']['splits']['validation']['path'])
    train = _write_combined_summary(
        base_path=v2_train,
        tail_path=train_tail_path,
        output_path=output_directory / 'codellama_surrogate_tail_seed24_v5_train.npz',
    )
    validation = _write_combined_summary(
        base_path=v2_validation,
        tail_path=validation_tail_path,
        output_path=(
            output_directory / 'codellama_surrogate_tail_seed24_v5_validation.npz'
        ),
    )
    if train['actions'] != 2536 or validation['actions'] != 192:
        raise ValueError('seed24 combined split counts changed unexpectedly')

    manifest = {
        'format_version': 3,
        'workflow_version': 5,
        'stage': 'development_tail_seed24',
        'not_a_final_acceptance_test': True,
        'generation': parent['generation'],
        'system': parent['system'],
        'quality_evaluator': quality_evaluator_metadata,
        'extension_config': extension_plan['config'],
        'parents': extension_plan['parents'],
        'extension_plan': {
            'path': str(extension_plan_path),
            'sha256': _sha256(extension_plan_path),
            'config_fingerprint': extension_plan['config_fingerprint'],
        },
        'extension_cache': {
            'path': str(extension_cache_path),
            'sha256': _sha256(extension_cache_path),
            'records': len(_load_jsonl(extension_cache_path)),
        },
        'tail_splits': {'train': train_tail, 'validation': validation_tail},
        'splits': {'train': train, 'validation': validation},
        'isolation_audit': isolation,
        'diagnostic_test': parent['diagnostic_test'],
    }
    manifest['dataset_fingerprint'] = canonical_json_hash(manifest)
    manifest_path = output_directory / 'codellama_surrogate_tail_seed24_v5_manifest.json'
    _atomic_write_json(manifest_path, manifest)
    return manifest
