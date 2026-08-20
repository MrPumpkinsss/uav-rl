import numpy as np

from uav_rl.data.general_assignment_dataset import (
    GeneralAssignmentDatasetConfig,
    build_general_assignment_plan,
    sample_general_assignment,
)
from uav_rl.config import DataGenerationConfig
from uav_rl.resource_assignment import (
    ResourceConstrainedConfig,
    resource_usage,
    validate_layerwise_deployment,
)
from uav_rl.resource_environment import generate_resource_channels


def test_sampler_keeps_exact_boundaries_and_allows_repeated_uavs() -> None:
    config = ResourceConstrainedConfig()
    channel = generate_resource_channels(1, 20260819, config)[0]
    rng = np.random.default_rng(7)
    repeated = False
    for boundary_count in range(2, 15):
        deployment = sample_general_assignment(
            rng,
            channel,
            config,
            target_boundaries=boundary_count,
            max_attempts=2_000,
        )
        validate_layerwise_deployment(deployment, config, channel=channel)
        assert int(np.count_nonzero(deployment[:-1] != deployment[1:])) == boundary_count
        runs = deployment[np.r_[True, deployment[1:] != deployment[:-1]]]
        repeated |= len(set(runs.tolist())) < len(runs)
        usage = resource_usage(deployment, config, channel=channel)
        assert np.all(usage.memory_units <= np.asarray(config.uav_memory_capacity_units) + 1e-9)
        assert np.all(usage.total_energy_joule <= np.asarray(config.uav_energy_budget_joule) + 1e-9)
    assert repeated


def test_general_plan_is_deterministic_and_split_seed_isolated(tmp_path) -> None:
    config = ResourceConstrainedConfig()
    dataset = GeneralAssignmentDatasetConfig(
        train_actions=4,
        validation_actions=2,
        test_actions=2,
        training_noise_samples=2,
        validation_noise_samples=2,
        test_noise_samples=2,
    )
    generation = DataGenerationConfig(model_id='test', text_sample_limit=2)
    first = build_general_assignment_plan(
        config=config,
        generation=generation,
        dataset=dataset,
        plan_path=tmp_path / 'plan.json',
    )
    second = build_general_assignment_plan(
        config=config,
        generation=generation,
        dataset=dataset,
        plan_path=tmp_path / 'plan.json',
    )
    assert first == second
    splits = {
        split: {
            seed
            for row in first['actions']
            if row['split'] == split
            for seed in row['noise_seeds']
        }
        for split in ('train', 'validation', 'test')
    }
    assert not (splits['train'] & splits['validation'])
    assert not (splits['train'] & splits['test'])
    assert not (splits['validation'] & splits['test'])
