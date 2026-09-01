"""System-level baselines for UAV collaborative LLM inference.

The modules in this package keep recent LLM deployment adaptations separate from
legacy DNN baselines in :mod:`uav_rl.resource_baselines`.
"""

from uav_rl.system_baselines.edge_shard_uav import edge_shard_uav_baseline
from uav_rl.system_baselines.exact_grouped_oracle import (
    ExactOracleResult,
    exact_grouped_reward_oracle,
)
from uav_rl.system_baselines.hexgen_search import hexgen_inspired_search_baseline
from uav_rl.system_baselines.lingualinked_uav import lingualinked_uav_baseline

__all__ = [
    "ExactOracleResult",
    "edge_shard_uav_baseline",
    "exact_grouped_reward_oracle",
    "hexgen_inspired_search_baseline",
    "lingualinked_uav_baseline",
]
