"""Scoped activation corruption matching cross-UAV packet loss."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def activation_dropout(
    model: Any,
    boundary_probabilities: dict[int, float],
) -> Iterator[None]:
    """Apply activation dropout at selected decoder layers and always remove hooks."""

    handles: list[Any] = []

    def make_hook(probability: float):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            mask = torch.rand_like(hidden) >= probability
            noisy_hidden = hidden * mask.to(hidden.dtype)
            return (noisy_hidden, *output[1:]) if isinstance(output, tuple) else noisy_hidden

        return hook

    try:
        layers = model.model.layers
        for layer, probability in boundary_probabilities.items():
            if probability <= 0.0:
                continue
            if not 0 <= layer < len(layers) - 1:
                raise ValueError(f"invalid model boundary layer: {layer}")
            if probability > 1.0:
                raise ValueError("drop probability cannot exceed one")
            handles.append(layers[layer].register_forward_hook(make_hook(probability)))
        yield
    finally:
        for handle in handles:
            handle.remove()
