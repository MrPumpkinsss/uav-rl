"""Configuration records for fair layerwise RL baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class A2CConfig:
    """Synchronous advantage actor-critic hyperparameters."""

    seed: int = 20260830
    hidden_dim: int = 512
    learning_rate: float = 3e-4
    rollout_size: int = 128
    training_episodes: int = 20_000
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    validation_channels: int = 256
    validation_seed: int = 20260901
    validation_interval_rollouts: int = 4
    max_boundaries: int = 4

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.rollout_size, self.training_episodes) < 1:
            raise ValueError("network size and training budgets must be positive")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.max_boundaries < 0:
            raise ValueError("max_boundaries cannot be negative")


@dataclass(frozen=True)
class DQNConfig:
    """Masked Double-DQN hyperparameters for the discrete UAV action space."""

    seed: int = 20260830
    hidden_dim: int = 512
    learning_rate: float = 1e-4
    rollout_size: int = 128
    training_episodes: int = 20_000
    gamma: float = 1.0
    replay_capacity: int = 200_000
    replay_warmup_transitions: int = 2_048
    batch_size: int = 256
    gradient_steps_per_rollout: int = 256
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 12_000
    max_grad_norm: float = 10.0
    validation_channels: int = 256
    validation_seed: int = 20260901
    validation_interval_rollouts: int = 4
    max_boundaries: int = 4

    def __post_init__(self) -> None:
        integer_values = (
            self.hidden_dim,
            self.rollout_size,
            self.training_episodes,
            self.replay_capacity,
            self.replay_warmup_transitions,
            self.batch_size,
            self.gradient_steps_per_rollout,
            self.target_update_interval,
            self.epsilon_decay_episodes,
        )
        if min(integer_values) < 1:
            raise ValueError("all DQN sizes and budgets must be positive")
        if self.replay_capacity < self.replay_warmup_transitions:
            raise ValueError("replay_capacity must cover the warm-up transitions")
        if not 0.0 < self.learning_rate or not 0.0 < self.gamma <= 1.0:
            raise ValueError("learning_rate and gamma must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon schedule must satisfy 0 <= end <= start <= 1")
        if self.max_boundaries < 0:
            raise ValueError("max_boundaries cannot be negative")
