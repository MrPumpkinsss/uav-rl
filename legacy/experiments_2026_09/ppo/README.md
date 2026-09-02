# Archived PPO scripts (2026-09)

These scripts were removed from the active command surface on September 2, 2026.
They remain here for historical reproducibility only and are not imported or
maintained as current experiment entry points.

| Archived script | Why archived | Current replacement |
| --- | --- | --- |
| `train_segment.py` | Older contiguous-segment PPO action space | `scripts/ppo/train_layerwise_topk.py` |
| `train_surrogate.py` | Older surrogate PPO trainer and environment | `scripts/ppo/train_layerwise_topk.py` |
| `compare_true_baselines.py` | Development comparison over previously consumed evidence | `scripts/baselines/evaluate_frozen_system_baselines_true.py` |
| `compare_general_assignment_baselines.py` | Superseded by the frozen system-baseline pipeline | `scripts/baselines/compare_system_baselines.py` |
| `diagnose_local_ranking.py` | One-off local ranking diagnostic | `scripts/surrogate/diagnose_coverage.py` and frozen benchmark reports |

The archived files may depend on older `uav_rl` APIs and are intentionally
excluded from current tests and Ruff checks. Do not use them for new runs.
