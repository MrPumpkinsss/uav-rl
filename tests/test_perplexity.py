import math
from types import SimpleNamespace

import pytest
import torch

from uav_rl.benchmarks import PerplexityBenchmarkConfig
from uav_rl.metrics import compute_perplexity


class FixedLogitModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def __call__(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self.logits)


def test_perplexity_weights_valid_tokens_and_ignores_padding() -> None:
    input_ids = torch.tensor([[0, 1, 0], [1, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    logits = torch.zeros(2, 3, 2)

    result = compute_perplexity(
        FixedLogitModel(logits),
        input_ids,
        attention_mask,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert result.evaluated_tokens == 3
    assert result.perplexity == pytest.approx(2.0)
    assert result.mean_negative_log_likelihood == pytest.approx(math.log(2.0))


def test_perplexity_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_perplexity(
            FixedLogitModel(torch.empty(0)),
            torch.ones(1, 3, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.long),
            batch_size=1,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_limit", 0),
        ("max_length", 1),
        ("batch_size", 0),
        ("measured_runs", 0),
        ("dtype", "int8"),
    ],
)
def test_invalid_benchmark_config_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        PerplexityBenchmarkConfig(**{field: value}).validate()
