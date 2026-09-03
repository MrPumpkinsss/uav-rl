# Script entry points

Set the source checkout path first:

```powershell
$env:PYTHONPATH = "src"
```

## System baselines

| Script | Purpose |
| --- | --- |
| `baselines/compare_system_baselines.py` | Freeze and compare the current PPO and system baselines, including EdgeShard-UAV and HexGen-inspired, with the common surrogate. |
| `baselines/evaluate_frozen_system_baselines_true.py` | Evaluate those exact frozen deployments with the matching true LLM and shared noise seeds. |

## PPO

| Script | Purpose |
| --- | --- |
| `ppo/train.py` | Train or resume PPO directly with the true CodeLlama PPL backend. |
| `ppo/train_layerwise_topk.py` | Train the current arbitrary layer-to-UAV PPO and export original Top-K candidates. |
| `ppo/validate_true_policy.py` | Select frozen candidates using independent true CodeLlama PPL. |

The older segment/surrogate PPO scripts and the previous baseline-comparison scripts are archived under `legacy/experiments_2026_09/ppo/`. The current PPO entry point is `ppo/train_layerwise_topk.py`; direct true-PPL training is `ppo/train.py`.

## RL algorithm baselines

| Script | Purpose |
| --- | --- |
| `rl/compare_algorithms.py` | Train PPO, A2C and masked Double-DQN from scratch under one common surrogate protocol. |
| `rl/evaluate_algorithms_true.py` | Evaluate frozen RL checkpoints on one common true-LLM channel/noise set. |
| `rl/plot_algorithm_comparison.py` | Plot per-seed and mean training/validation learning curves. |

The controlled RL table is separate from the teacher-warm-started best-system PPO table. See
`docs/RL_BASELINE_PROTOCOL.md` for the fairness rules and output layout.

## Surrogate

| Script | Purpose |
| --- | --- |
| `surrogate/collect_general_assignment.py` | Generate resumable general-assignment true-PPL labels. |
| `surrogate/collect_high_boundary_augmentation.py` | Collect targeted high-boundary labels. |
| `surrogate/train_general_assignment.py` | Train the general-assignment ensemble. |
| `surrogate/train.py` | Train the current global surrogate ensemble. |
| `surrogate/train_residual.py` | Train the targeted tail residual ensemble. |
| `surrogate/diagnose_coverage.py` | Inspect coverage and disagreement without adding labels. |
| `surrogate/extend_labels.py` | Resume targeted label extension. |

## Other

| Script | Purpose |
| --- | --- |
| `benchmarks/measure_ppl.py` | Measure one true CodeLlama PPL run. |
| `maintenance/verify_legacy_archive.py` | Verify archive paths, sizes, and SHA256 values. |

Use a fresh descriptive run directory for a new experiment. Do not encode
version numbers in source filenames; put variants in `artifacts/runs/` instead.
