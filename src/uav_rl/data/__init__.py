"""Offline datasets used to decouple PPO training from CodeLlama inference."""

from __future__ import annotations

from typing import Any


def generate_ppl_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazily import heavy Hugging Face dependencies for PPL generation."""

    from .ppl_dataset import generate_ppl_dataset as implementation

    return implementation(*args, **kwargs)


__all__ = ["generate_ppl_dataset"]
