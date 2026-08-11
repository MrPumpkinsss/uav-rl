"""Tests for engineered PPL surrogate features and checkpoint loading."""

from pathlib import Path

import torch

from uav_rl.surrogate import PPLSurrogate, load_surrogate


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
