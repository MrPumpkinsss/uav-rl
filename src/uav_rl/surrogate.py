"""Trainable PPL reward surrogate for activation packet loss."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class PPLSurrogate(nn.Module):
    """Predict log(PPL_noisy / PPL_clean) from boundary drop rates."""

    def __init__(self, num_boundaries: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_boundaries = num_boundaries
        engineered_features = 5
        feature_count = num_boundaries + engineered_features
        self.register_buffer("feature_mean", torch.zeros(feature_count))
        self.register_buffer("feature_scale", torch.ones(feature_count))
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _engineer_features(self, drop_probabilities: torch.Tensor) -> torch.Tensor:
        if drop_probabilities.shape[-1] != self.num_boundaries:
            raise ValueError(
                f"expected {self.num_boundaries} boundary probabilities, "
                f"got {drop_probabilities.shape[-1]}"
            )
        total = drop_probabilities.sum(dim=-1, keepdim=True)
        maximum = drop_probabilities.max(dim=-1, keepdim=True).values
        square_sum = drop_probabilities.square().sum(dim=-1, keepdim=True)
        boundary_fraction = (drop_probabilities > 0).float().mean(dim=-1, keepdim=True)
        cumulative_hazard = -torch.log1p(-drop_probabilities.clamp_max(1.0 - 1e-6)).sum(
            dim=-1, keepdim=True
        )
        return torch.cat(
            [
                drop_probabilities,
                total,
                maximum,
                square_sum,
                boundary_fraction,
                cumulative_hazard,
            ],
            dim=-1,
        )

    def forward(self, drop_probabilities: torch.Tensor) -> torch.Tensor:
        features = self._engineer_features(drop_probabilities)
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.network(normalized).squeeze(-1)


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    seed: int = 20260810
    hidden_dim: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 1500
    validation_fraction: float = 0.2
    patience: int = 250


def train_surrogate(
    dataset_path: Path,
    output_path: Path,
    config: SurrogateTrainingConfig,
) -> dict[str, float]:
    """Fit the surrogate with a fixed train/validation split and early stopping."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    data = np.load(dataset_path)
    features = torch.from_numpy(data["drop_probabilities"]).float()
    targets = torch.from_numpy(data["log_ppl_ratio"]).float()

    generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(features.size(0), generator=generator)
    validation_size = max(1, round(features.size(0) * config.validation_fraction))
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    train_x, train_y = features[training_indices], targets[training_indices]
    validation_x, validation_y = features[validation_indices], targets[validation_indices]

    model = PPLSurrogate(features.size(1), config.hidden_dim)
    with torch.no_grad():
        engineered_train = model._engineer_features(train_x)
        model.feature_mean.copy_(engineered_train.mean(dim=0))
        # A boundary can be absent from a finite training split yet appear in a
        # valid PPO action. Keep sparse probability dimensions on a physical scale
        # instead of amplifying them by an epsilon-sized standard deviation.
        model.feature_scale.copy_(engineered_train.std(dim=0).clamp_min(0.02))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        prediction = model(train_x)
        loss = loss_function(prediction, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(validation_x), validation_y).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prediction = model(train_x)
        validation_prediction = model(validation_x)

    def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
        mae = torch.mean(torch.abs(prediction - target)).item()
        denominator = torch.sum((target - target.mean()) ** 2)
        r2 = 1.0 - torch.sum((prediction - target) ** 2) / denominator
        return mae, float(r2.item())

    train_mae, train_r2 = metrics(train_prediction, train_y)
    validation_mae, validation_r2 = metrics(validation_prediction, validation_y)
    checkpoint: dict[str, Any] = {
        "model_state": model.state_dict(),
        "num_boundaries": features.size(1),
        "hidden_dim": config.hidden_dim,
        "training_config": asdict(config),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    result = {
        "train_mae_log_ratio": train_mae,
        "train_r2": train_r2,
        "validation_mae_log_ratio": validation_mae,
        "validation_r2": validation_r2,
        "best_validation_mse": best_loss,
        "epochs_completed": float(epoch + 1),
        "training_samples": float(train_x.size(0)),
        "validation_samples": float(validation_x.size(0)),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def load_surrogate(path: Path, device: torch.device | None = None) -> PPLSurrogate:
    """Load a trained, evaluation-only surrogate."""

    target_device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    model = PPLSurrogate(checkpoint["num_boundaries"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(target_device).eval()
