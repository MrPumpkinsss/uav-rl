# Script entry points

Set the source checkout path first:

```powershell
$env:PYTHONPATH = "src"
```

## System baselines

| Script | Purpose |
| --- | --- |
| `baselines/compare_system_baselines.py` | Freeze and compare PPO, JointDNN-MUAV, PipeEdge-UAV, Petals-balanced, Neurosurgeon-inspired, simulated annealing, and random feasible deployments with the common surrogate. |
| `baselines/evaluate_frozen_system_baselines_true.py` | Evaluate those exact frozen deployments with the matching true LLM and shared noise seeds. |

## PPO

| Script | Purpose |
| --- | --- |
| `ppo/train.py` | Train or resume PPO with the configured quality backend. |
| `ppo/train_layerwise_topk.py` | Train the current arbitrary layer-to-UAV PPO and export original Top-K candidates. |
| `ppo/train_segment.py` | Train the older contiguous-segment policy family. |
| `ppo/train_surrogate.py` | Train PPO with a frozen surrogate reward and export frozen candidates. |
| `ppo/validate_true_policy.py` | Select frozen candidates using independent true CodeLlama PPL. |
| `ppo/compare_true_baselines.py` | Compare a selected policy with standard baselines on consumed validation evidence. |
| `ppo/compare_general_assignment_baselines.py` | Run the resource-aware common-seed baseline comparison. |
| `ppo/diagnose_local_ranking.py` | Diagnose local surrogate ranking around Strong-link actions. |

The Diverse Top-K experiment is archived and is not a current PPO entry point.

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
