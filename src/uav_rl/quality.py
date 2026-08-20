"""Quality backends consumed by the deployment reward environment."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from uav_rl.surrogate import SurrogateModel


@runtime_checkable
class QualityEvaluator(Protocol):
    """Map boundary packet-drop probabilities to log PPL ratios."""

    def evaluate(
        self,
        drop_probabilities: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one log PPL ratio for every row in ``drop_probabilities``."""

        ...


class SurrogateQualityEvaluator:
    """Adapt a trained PyTorch PPL surrogate to the common quality interface."""

    def __init__(self, model: SurrogateModel) -> None:
        self.model = model.eval()

    def evaluate(
        self,
        drop_probabilities: np.ndarray,
        *,
        noise_seeds: np.ndarray | None = None,
    ) -> np.ndarray:
        del noise_seeds
        device = next(self.model.parameters()).device
        inputs = torch.from_numpy(np.asarray(drop_probabilities, dtype=np.float32)).to(device)
        with torch.no_grad():
            predictions = self.model(inputs).clamp_min(0.0)
        return predictions.cpu().numpy().astype(np.float32, copy=False)
