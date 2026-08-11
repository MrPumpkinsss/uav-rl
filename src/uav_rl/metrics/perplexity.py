"""Token-weighted perplexity for causal language models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PerplexityResult:
    """Aggregated causal language-model evaluation result."""

    perplexity: float
    mean_negative_log_likelihood: float
    evaluated_tokens: int
    evaluated_sequences: int
    batches: int


def compute_perplexity(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> PerplexityResult:
    """Compute corpus PPL by weighting every predicted token equally.

    Padding tokens and the first token of each sequence are excluded because a causal
    language model does not predict them.
    """

    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    if input_ids.ndim != 2:
        raise ValueError("input tensors must have shape [sequences, tokens]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    total_negative_log_likelihood = 0.0
    total_tokens = 0
    batch_count = 0

    with torch.inference_mode():
        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            batch_attention_mask = attention_mask[start:end]
            batch_length = int(batch_attention_mask.sum(dim=1).max().item())
            batch_input_ids = input_ids[start:end, :batch_length].to(device)
            batch_attention_mask = batch_attention_mask[:, :batch_length].to(device)
            logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=False,
            ).logits

            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = batch_input_ids[:, 1:].contiguous()
            valid_tokens = batch_attention_mask[:, 1:].contiguous().bool()
            token_losses = F.cross_entropy(
                shifted_logits.transpose(1, 2),
                shifted_labels,
                reduction="none",
            )
            valid_losses = token_losses[valid_tokens]
            if not torch.isfinite(valid_losses).all():
                raise FloatingPointError(
                    f"non-finite token loss in corpus batch starting at sequence {start}"
                )

            total_negative_log_likelihood += valid_losses.sum().item()
            total_tokens += int(valid_tokens.sum().item())
            batch_count += 1

    if total_tokens == 0:
        raise ValueError("at least one next-token target is required to compute perplexity")

    mean_nll = total_negative_log_likelihood / total_tokens
    return PerplexityResult(
        perplexity=math.exp(mean_nll),
        mean_negative_log_likelihood=mean_nll,
        evaluated_tokens=total_tokens,
        evaluated_sequences=input_ids.size(0),
        batches=batch_count,
    )
