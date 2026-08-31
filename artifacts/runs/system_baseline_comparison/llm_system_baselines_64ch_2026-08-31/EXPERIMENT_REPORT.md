# Frozen surrogate system-baseline comparison

## Status

This is a frozen **surrogate-screening** comparison. The exact deployments and channel matrices are saved for a later common-seed true-LLM evaluation; these numbers must not be presented as final true-PPL results.

## Protocol

- Channels: `64`
- Channel seed: `20260910`
- PPO checkpoint: `artifacts\runs\surrogate_ppo\layerwise_topk_high_augmented_2026-08-20\best_policy.pth`
- Frozen surrogate: `artifacts\models\ppl_surrogate_general_assignment_high_augmented_ensemble.pth`
- EdgeShard plans retained per DP state: `8`
- HexGen-inspired search: population `48`, generations `48` per channel
- JointDNN MILP time limit: `1.0` seconds/channel
- Simulated-annealing steps: `1024` per channel

## Results

| Method | Reward up | Delta vs PPO [95% CI] | log-PPL ratio down | Latency (s) down | Boundaries | Decision ms/channel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Proposed PPO | -0.383815 | +0.000000 [+0.000000, +0.000000] | 0.207401 | 0.735128 | 2.062 | 10.708 |
| EdgeShard-UAV | -0.424044 | -0.040229 [-0.057753, -0.022705] | 0.315048 | 0.699451 | 2.000 | 265.963 |
| HexGen-inspired | -0.372926 | +0.010890 [-0.004879, +0.026658] | 0.204905 | 0.709825 | 2.000 | 793.982 |
| JointDNN-MUAV | -0.398647 | -0.014832 [-0.032614, +0.002951] | 0.261025 | 0.703687 | 2.000 | 1827.435 |
| PipeEdge-UAV | -0.431887 | -0.048072 [-0.065950, -0.030194] | 0.330299 | 0.700022 | 2.000 | 112.735 |
| Petals-balanced | -0.802524 | -0.418709 [-0.444753, -0.392664] | 0.531086 | 1.409244 | 4.000 | 112.684 |
| Neurosurgeon-inspired | -0.388179 | -0.004363 [-0.020974, +0.012247] | 0.238366 | 0.705948 | 2.000 | 209.051 |
| Simulated annealing | -0.373881 | +0.009935 [-0.005966, +0.025835] | 0.203512 | 0.714160 | 2.000 | 2973.176 |
| Random feasible | -3.477950 | -3.094135 [-3.634770, -2.553500] | 2.838805 | 5.402419 | 8.469 | 8.896 |

## Interpretation boundary

EdgeShard-UAV, HexGen-inspired, JointDNN-MUAV, PipeEdge-UAV, Petals-balanced and Neurosurgeon-inspired are explicit multi-UAV adaptations of prior system principles, not byte-for-byte reproductions of their original device/cloud implementations. Method names and paper text must retain the `-style` or `-inspired` qualification where appropriate.
