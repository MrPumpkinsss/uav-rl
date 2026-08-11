"""Offline datasets used to decouple PPO training from CodeLlama inference."""

from .ppl_dataset import generate_ppl_dataset

__all__ = ["generate_ppl_dataset"]
