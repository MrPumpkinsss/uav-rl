"""Performance benchmarks used by the UAV inference experiments."""

from .perplexity import PerplexityBenchmarkConfig, run_perplexity_benchmark

__all__ = ["PerplexityBenchmarkConfig", "run_perplexity_benchmark"]
