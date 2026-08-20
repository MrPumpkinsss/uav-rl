"""Typed experiment configuration shared by data generation and RL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SystemConfig:
    """UAV collaborative-inference parameters in normalized physical units."""

    num_layers: int = 32
    num_uavs: int = 5
    max_layers_per_uav: int = 8
    compute_seconds_per_layer: float = 0.020
    compute_speed: tuple[float, ...] = (1.0, 1.25, 0.9, 1.4, 1.1)
    activation_size_mbit: float = 4.0
    total_bandwidth_mhz: float = 20.0
    transmit_power: float = 1.0
    noise_power: float = 1.0
    decoding_threshold: float = 1.0
    channel_gain_min: float = 2.0
    channel_gain_max: float = 20.0
    quality_weight: float = 0.5

    def __post_init__(self) -> None:
        if len(self.compute_speed) != self.num_uavs:
            raise ValueError("compute_speed must contain one value per UAV")
        if self.num_layers > self.num_uavs * self.max_layers_per_uav:
            raise ValueError("UAV capacity is insufficient for all model layers")
        if not 0.0 <= self.quality_weight <= 1.0:
            raise ValueError("quality_weight must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DataGenerationConfig:
    """Reproducible CodeLlama PPL dataset settings."""

    model_id: str = "codellama/CodeLlama-7b-hf"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "test"
    dataset_arrow_file: str | None = None
    text_sample_limit: int = 50
    max_length: int = 512
    batch_size: int = 4
    num_samples: int = 256
    seed: int = 20260810
    noise_seed: int = 314159
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class PPOConfig:
    """Contextual-bandit PPO hyperparameters."""

    seed: int = 20260810
    hidden_dim: int = 512
    learning_rate: float = 3e-5
    rollout_size: int = 128
    training_episodes: int = 8192
    update_epochs: int = 6
    minibatch_size: int = 64
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5
    teacher_channels: int = 8192
    teacher_seed: int = 20260814
    behavior_cloning_epochs: int = 60
    behavior_cloning_learning_rate: float = 3e-4
    teacher_relative_rewards: bool = False
    online_behavior_cloning_coefficient: float = 0.0
    validation_channels: int = 256
    validation_seed: int = 20260813
    validation_interval: int = 4
    test_channels: int = 256
    test_seed: int = 20260811
    training_noise_samples: int = 4
    training_noise_seed: int = 20260815
    validation_noise_samples: int = 16
    validation_noise_seed: int = 20260816
    test_noise_samples: int = 16
    test_noise_seed: int = 20260817
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self) -> None:
        experiment_seeds = {self.teacher_seed, self.validation_seed, self.test_seed}
        if len(experiment_seeds) != 3:
            raise ValueError("teacher, validation, and test seeds must be distinct")
        noise_generator_seeds = {
            self.training_noise_seed,
            self.validation_noise_seed,
            self.test_noise_seed,
        }
        if len(noise_generator_seeds) != 3:
            raise ValueError("training, validation, and test noise generator seeds must be distinct")
        if min(
            self.training_noise_samples,
            self.validation_noise_samples,
            self.test_noise_samples,
        ) < 1:
            raise ValueError("all noise sample counts must be positive")
        if self.online_behavior_cloning_coefficient < 0.0:
            raise ValueError("online behavior-cloning coefficient cannot be negative")
