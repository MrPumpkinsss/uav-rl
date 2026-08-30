# RL algorithm baseline protocol

## Purpose

The UAV layer-partition experiment now has two deliberately separate result tables:

1. **Controlled RL comparison:** PPO vs A2C vs DQN, all trained from scratch with the same
   state, discrete UAV action space, resource mask, boundary cap, surrogate reward, channel
   seeds, episode budget and held-out evaluation channels.
2. **Best-system comparison:** the strongest deployable PPO checkpoint (which may use teacher
   warm-starting) vs CoEdge, Neurosurgeon, dynamic programming, MILP proxy, search methods and
   random feasible assignments.

Do not merge these tables. Teacher initialization is a system improvement, not an intrinsic
property of PPO, and would make an algorithm-only comparison unfair.

## Why these RL baselines

- **PPO:** current primary on-policy clipped actor-critic method.
- **A2C:** an unclipped synchronous actor-critic control. It shares the PPO actor/value
  architecture, so the comparison isolates the clipped PPO update as much as practical.
- **DQN:** a value-based discrete-action baseline. The implementation uses action masking,
  replay, a target network and Double-DQN action selection. A one-shot DQN over all complete
  assignments is not used because the action count is exponential (`num_uavs ** num_layers`).

SAC, TD3 and DDPG are not primary baselines here because their standard forms target continuous
actions, while each decision in this project is a discrete UAV index. They can become relevant
only after defining a continuous relaxation and a projection rule, which would change the
optimization problem.

## Recommended paper baseline matrix

The main experiment should contain both learned and non-learned references:

| Group | Methods | Role |
| --- | --- | --- |
| RL, controlled | PPO, A2C, masked Double-DQN | Compare on-policy clipped, on-policy unclipped and off-policy value-based learning. |
| Simple sanity checks | Random feasible, compute greedy, strong-link | Establish lower bounds and expose reward/constraint bugs. |
| Prior partition heuristics | CoEdge-style adaptive partition, Neurosurgeon best split | Compare with recognizable edge-partition strategies. |
| Optimization/search | Dynamic programming proxy, MILP proxy, simulated annealing, genetic search | Separate policy-learning gains from per-instance optimization budget. |
| Diagnostic upper bounds | Top-K true oracle, optional exhaustive oracle on tiny cases | Diagnose candidate-generation and surrogate-ranking regret; never call these deployable methods. |

For readability, the main paper table can include PPO/A2C/DQN plus Random, CoEdge, Neurosurgeon,
MILP-proxy and the strongest search baseline. Put the remaining methods in an ablation or appendix.

## Common MDP

- One episode corresponds to one channel matrix.
- The episode has one step per model layer.
- The action is the UAV assigned to the current layer.
- Observation: normalized channel matrix, layer progress, normalized memory use, normalized
  energy use and previous-UAV one-hot state.
- Infeasible actions are masked before sampling or maximization.
- Reward is evaluated after the complete assignment:

  `-(quality_weight * log_ppl_ratio + latency_weight * normalized_latency)`

- Default boundary freeze threshold: 4. This matches the current layerwise PPO implementation. It is not a strict cap: when staying would violate memory/energy feasibility, another switch remains legal. Therefore every table must report both realized boundary count and threshold-exceedance fraction.
- The controlled comparison uses surrogate reward for training and screening. Any paper result
  must subsequently be confirmed on the same true-LLM channel/noise set.

## Run the comparison

```powershell
$env:PYTHONPATH = "src"
python scripts/rl/compare_algorithms.py `
  --algorithms ppo a2c dqn `
  --seeds 20260830 20260831 20260902 `
  --episodes 20000 `
  --evaluation-channels 512 `
  --output-dir artifacts/runs/rl_algorithm_comparison/ppo_a2c_dqn_20k_3seed
```

For the final paper, increase the budget only if all algorithms receive the same increase. Report
mean and standard deviation across at least three seeds, learning curves, invalid fraction,
boundary count, reward, log-PPL ratio, latency and wall-clock training time.

## Output layout

```text
artifacts/runs/rl_algorithm_comparison/<experiment_name>/
├── ppo/seed_<seed>/
├── a2c/seed_<seed>/
├── dqn/seed_<seed>/
├── comparison_summary.json
└── comparison_table.csv
```

Each seed directory contains a best validation checkpoint, complete training history and a
held-out surrogate evaluation record. These files are inputs to a later common-seed true-LLM
validation stage; surrogate-only metrics must be labelled as such.

## Method references

- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518, 2015.
- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, ICML 2016.
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
