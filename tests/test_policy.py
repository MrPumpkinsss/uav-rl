import torch

from uav_rl.config import SystemConfig
from uav_rl.deployment import validate_deployment
from uav_rl.rl.policy import ContinuousDeploymentActorCritic


def test_policy_actions_are_valid_and_log_probability_recomputes() -> None:
    torch.manual_seed(123)
    config = SystemConfig(num_layers=8, max_layers_per_uav=3)
    policy = ContinuousDeploymentActorCritic(config, hidden_dim=32)
    states = torch.rand(6, config.num_uavs * config.num_uavs)

    sampled = policy.sample(states)
    evaluated = policy.evaluate(states, sampled.actions)

    for deployment in sampled.actions.numpy():
        validate_deployment(deployment, config)
    assert torch.allclose(sampled.log_probabilities, evaluated.log_probabilities)
    assert sampled.actions.shape == (6, config.num_layers)
