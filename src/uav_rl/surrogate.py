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


class PPLSurrogateEnsemble(nn.Module):
    """Average independent PPL surrogates and expose epistemic uncertainty."""

    def __init__(self, models: list[PPLSurrogate]) -> None:
        super().__init__()
        if not models:
            raise ValueError("a surrogate ensemble must contain at least one model")
        num_boundaries = models[0].num_boundaries
        if any(model.num_boundaries != num_boundaries for model in models):
            raise ValueError("all ensemble members must use the same boundary count")
        self.num_boundaries = num_boundaries
        self.models = nn.ModuleList(models)

    def predict_with_uncertainty(
        self,
        drop_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.stack(
            [model(drop_probabilities) for model in self.models],
            dim=0,
        )
        return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)

    def forward(self, drop_probabilities: torch.Tensor) -> torch.Tensor:
        mean, _ = self.predict_with_uncertainty(drop_probabilities)
        return mean


class TailGatedSurrogate(nn.Module):
    """Blend a frozen global ensemble with a tail-only expert ensemble."""

    def __init__(
        self,
        base_model: PPLSurrogateEnsemble,
        expert_models: list[PPLSurrogate],
        *,
        hazard_threshold: float = 0.4,
        hazard_temperature: float = 0.03,
    ) -> None:
        super().__init__()
        if not expert_models:
            raise ValueError("a tail-gated surrogate needs at least one expert")
        if len(base_model.models) != len(expert_models):
            raise ValueError("base and expert ensembles must have equal member counts")
        if hazard_temperature <= 0.0:
            raise ValueError("hazard temperature must be positive")
        self.base_model = base_model
        self.expert_models = nn.ModuleList(expert_models)
        self.num_boundaries = base_model.num_boundaries
        self.register_buffer(
            "hazard_threshold", torch.tensor(float(hazard_threshold))
        )
        self.register_buffer(
            "hazard_temperature", torch.tensor(float(hazard_temperature))
        )
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def gate(self, drop_probabilities: torch.Tensor) -> torch.Tensor:
        hazard = -torch.log1p(
            -drop_probabilities.clamp_max(1.0 - 1e-6)
        ).sum(dim=-1)
        return torch.sigmoid(
            (hazard - self.hazard_threshold) / self.hazard_temperature
        )

    def predict_with_uncertainty(
        self,
        drop_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_predictions = torch.stack(
            [model(drop_probabilities) for model in self.base_model.models], dim=0
        )
        expert_predictions = torch.stack(
            [model(drop_probabilities) for model in self.expert_models], dim=0
        )
        gate = self.gate(drop_probabilities).unsqueeze(0)
        blended = base_predictions + gate * (expert_predictions - base_predictions)
        return blended.mean(dim=0), blended.std(dim=0, unbiased=False)

    def forward(self, drop_probabilities: torch.Tensor) -> torch.Tensor:
        mean, _ = self.predict_with_uncertainty(drop_probabilities)
        return mean


class TailResidualSurrogate(nn.Module):
    """Apply a hazard-gated residual correction to a frozen global ensemble.

    Residual members predict a correction to the paired global member rather
    than a second complete PPL estimate. This keeps the global prediction
    intact outside the tail region while targeted data corrects tail bias.
    """

    def __init__(
        self,
        base_model: PPLSurrogateEnsemble,
        residual_models: list[PPLSurrogate],
        *,
        hazard_threshold: float = 0.4,
        hazard_temperature: float = 0.03,
    ) -> None:
        super().__init__()
        if not residual_models:
            raise ValueError("a tail-residual surrogate needs at least one member")
        if len(base_model.models) != len(residual_models):
            raise ValueError("base and residual ensembles must have equal member counts")
        if hazard_temperature <= 0.0:
            raise ValueError("hazard temperature must be positive")
        self.base_model = base_model
        self.residual_models = nn.ModuleList(residual_models)
        self.num_boundaries = base_model.num_boundaries
        self.register_buffer("hazard_threshold", torch.tensor(float(hazard_threshold)))
        self.register_buffer("hazard_temperature", torch.tensor(float(hazard_temperature)))
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def gate(self, drop_probabilities: torch.Tensor) -> torch.Tensor:
        hazard = -torch.log1p(
            -drop_probabilities.clamp_max(1.0 - 1e-6)
        ).sum(dim=-1)
        return torch.sigmoid(
            (hazard - self.hazard_threshold) / self.hazard_temperature
        )

    def predict_with_uncertainty(
        self,
        drop_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_predictions = torch.stack(
            [model(drop_probabilities) for model in self.base_model.models], dim=0
        )
        residual_predictions = torch.stack(
            [model(drop_probabilities) for model in self.residual_models], dim=0
        )
        corrected = base_predictions + (
            self.gate(drop_probabilities).unsqueeze(0) * residual_predictions
        )
        return corrected.mean(dim=0), corrected.std(dim=0, unbiased=False)

    def forward(self, drop_probabilities: torch.Tensor) -> torch.Tensor:
        mean, _ = self.predict_with_uncertainty(drop_probabilities)
        return mean


SurrogateModel = (
    PPLSurrogate | PPLSurrogateEnsemble | TailGatedSurrogate | TailResidualSurrogate
)


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


def load_surrogate(path: Path, device: torch.device | None = None) -> SurrogateModel:
    """Load a single model, ensemble, tail-gated, or residual checkpoint."""

    target_device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    if checkpoint.get("model_type") == "tail_residual_ensemble":
        base_models = []
        for state in checkpoint["base_model_states"]:
            model = PPLSurrogate(
                checkpoint["num_boundaries"], checkpoint["base_hidden_dim"]
            )
            model.load_state_dict(state)
            base_models.append(model)
        residual_models = []
        for state in checkpoint["residual_model_states"]:
            model = PPLSurrogate(
                checkpoint["num_boundaries"], checkpoint["residual_hidden_dim"]
            )
            model.load_state_dict(state)
            residual_models.append(model)
        return TailResidualSurrogate(
            PPLSurrogateEnsemble(base_models),
            residual_models,
            hazard_threshold=checkpoint["gate"]["hazard_threshold"],
            hazard_temperature=checkpoint["gate"]["hazard_temperature"],
        ).to(target_device).eval()
    if checkpoint.get("model_type") == "tail_gated_ensemble":
        base_models = []
        for state in checkpoint["base_model_states"]:
            model = PPLSurrogate(
                checkpoint["num_boundaries"], checkpoint["base_hidden_dim"]
            )
            model.load_state_dict(state)
            base_models.append(model)
        expert_models = []
        for state in checkpoint["expert_model_states"]:
            model = PPLSurrogate(
                checkpoint["num_boundaries"], checkpoint["expert_hidden_dim"]
            )
            model.load_state_dict(state)
            expert_models.append(model)
        return TailGatedSurrogate(
            PPLSurrogateEnsemble(base_models),
            expert_models,
            hazard_threshold=checkpoint["gate"]["hazard_threshold"],
            hazard_temperature=checkpoint["gate"]["hazard_temperature"],
        ).to(target_device).eval()
    if checkpoint.get("format_version") == 2:
        models = []
        for state in checkpoint["model_states"]:
            model = PPLSurrogate(checkpoint["num_boundaries"], checkpoint["hidden_dim"])
            model.load_state_dict(state)
            models.append(model)
        return PPLSurrogateEnsemble(models).to(target_device).eval()
    model = PPLSurrogate(checkpoint["num_boundaries"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(target_device).eval()
