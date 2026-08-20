"""Tests for engineered PPL surrogate features and checkpoint loading."""

from pathlib import Path

import torch

from uav_rl.surrogate import (
    PPLSurrogate,
    PPLSurrogateEnsemble,
    TailGatedSurrogate,
    TailResidualSurrogate,
    load_surrogate,
)


def test_surrogate_accepts_boundary_probabilities() -> None:
    model = PPLSurrogate(num_boundaries=31, hidden_dim=16)
    prediction = model(torch.zeros(4, 31))

    assert prediction.shape == (4,)
    assert torch.isfinite(prediction).all()


def test_surrogate_checkpoint_preserves_feature_scaling(tmp_path: Path) -> None:
    model = PPLSurrogate(num_boundaries=31, hidden_dim=16)
    model.feature_mean.fill_(0.25)
    model.feature_scale.fill_(0.5)
    checkpoint = {
        "model_state": model.state_dict(),
        "num_boundaries": 31,
        "hidden_dim": 16,
    }
    path = tmp_path / "surrogate.pth"
    torch.save(checkpoint, path)

    loaded = load_surrogate(path)

    assert torch.equal(loaded.feature_mean, model.feature_mean)
    assert torch.equal(loaded.feature_scale, model.feature_scale)


def test_ensemble_returns_member_mean_and_uncertainty() -> None:
    first = PPLSurrogate(num_boundaries=3, hidden_dim=4)
    second = PPLSurrogate(num_boundaries=3, hidden_dim=4)
    for parameter in first.parameters():
        parameter.data.zero_()
    for parameter in second.parameters():
        parameter.data.zero_()
    first.network[-1].bias.data.fill_(1.0)
    second.network[-1].bias.data.fill_(3.0)
    ensemble = PPLSurrogateEnsemble([first, second])

    mean, uncertainty = ensemble.predict_with_uncertainty(torch.zeros(2, 3))

    assert torch.equal(mean, torch.full((2,), 2.0))
    assert torch.equal(uncertainty, torch.full((2,), 1.0))
    assert torch.equal(ensemble(torch.zeros(2, 3)), mean)


def test_version_two_ensemble_checkpoint_loads(tmp_path: Path) -> None:
    models = [PPLSurrogate(num_boundaries=3, hidden_dim=4) for _ in range(2)]
    checkpoint = {
        "format_version": 2,
        "model_states": [model.state_dict() for model in models],
        "num_boundaries": 3,
        "hidden_dim": 4,
    }
    path = tmp_path / "ensemble.pth"
    torch.save(checkpoint, path)

    loaded = load_surrogate(path)

    assert isinstance(loaded, PPLSurrogateEnsemble)
    assert len(loaded.models) == 2


def test_tail_gated_checkpoint_loads_and_blends_members(tmp_path: Path) -> None:
    base_models = [PPLSurrogate(num_boundaries=3, hidden_dim=4) for _ in range(2)]
    expert_models = [PPLSurrogate(num_boundaries=3, hidden_dim=4) for _ in range(2)]
    for model in [*base_models, *expert_models]:
        for parameter in model.parameters():
            parameter.data.zero_()
    for model in base_models:
        model.network[-1].bias.data.fill_(1.0)
    for model in expert_models:
        model.network[-1].bias.data.fill_(3.0)
    checkpoint = {
        "format_version": 4,
        "model_type": "tail_gated_ensemble",
        "base_model_states": [model.state_dict() for model in base_models],
        "expert_model_states": [model.state_dict() for model in expert_models],
        "num_boundaries": 3,
        "base_hidden_dim": 4,
        "expert_hidden_dim": 4,
        "gate": {"hazard_threshold": 0.0, "hazard_temperature": 1.0},
    }
    path = tmp_path / "tail_gated.pth"
    torch.save(checkpoint, path)

    loaded = load_surrogate(path)
    assert isinstance(loaded, TailGatedSurrogate)
    prediction = loaded(torch.zeros(2, 3))
    assert torch.allclose(prediction, torch.full((2,), 2.0))



def test_tail_residual_checkpoint_loads_and_corrects_members(tmp_path: Path) -> None:
    base_models = [PPLSurrogate(num_boundaries=3, hidden_dim=4) for _ in range(2)]
    residual_models = [PPLSurrogate(num_boundaries=3, hidden_dim=4) for _ in range(2)]
    for model in [*base_models, *residual_models]:
        for parameter in model.parameters():
            parameter.data.zero_()
    for model in base_models:
        model.network[-1].bias.data.fill_(1.0)
    for model in residual_models:
        model.network[-1].bias.data.fill_(3.0)
    checkpoint = {
        "format_version": 1,
        "model_type": "tail_residual_ensemble",
        "base_model_states": [model.state_dict() for model in base_models],
        "residual_model_states": [model.state_dict() for model in residual_models],
        "num_boundaries": 3,
        "base_hidden_dim": 4,
        "residual_hidden_dim": 4,
        "gate": {"hazard_threshold": 0.0, "hazard_temperature": 1.0},
    }
    path = tmp_path / "tail_residual.pth"
    torch.save(checkpoint, path)

    loaded = load_surrogate(path)
    assert isinstance(loaded, TailResidualSurrogate)
    prediction = loaded(torch.zeros(2, 3))
    expected = 1.0 + 3.0 * torch.sigmoid(torch.tensor(0.0))
    assert torch.allclose(prediction, torch.full((2,), expected))
