# Project structure

This repository keeps current implementation, executable entry points, generated artifacts, and historical experiments separate.

## Current implementation

```text
src/uav_rl/
├── config.py                  # shared dataclasses and experiment configuration
├── resource_assignment.py     # memory/energy-constrained layer assignments
├── resource_environment.py    # surrogate/true reward environment
├── surrogate.py               # surrogate model definitions and checkpoint loading
├── surrogate_training.py      # ensemble training and validation
├── true_quality.py            # direct CodeLlama PPL evaluator with JSONL cache
├── data/                      # dataset generation, aggregation, and label utilities
├── metrics/                   # PPL and evaluation metrics
├── models/                    # activation-dropout model hooks
├── benchmarks/                # reusable benchmark evaluators
├── system_baselines/          # EdgeShard-UAV, HexGen-inspired, grouped oracle
└── rl/                        # shared RL state, PPO, policy I/O, and oracle code
    └── algorithms/            # A2C and masked Double-DQN baselines
```

## Executable entry points

```text
scripts/
├── baselines/                 # system/heuristic screening and frozen true evaluation
├── benchmarks/                # one-shot measurement commands
├── ppo/                       # production PPO training and heuristic comparison
├── rl/                        # controlled PPO/A2C/DQN training and true evaluation
├── surrogate/                 # surrogate data collection, diagnostics, and training
└── maintenance/               # archive/hash and repository maintenance checks
```

The canonical current PPO entry point is `scripts/ppo/train.py`. Surrogate
training commands are research utilities and are not imported by the PPO runtime.

## Generated artifacts

```text
artifacts/
├── data/                      # NPZ datasets and manifests
├── cache/                     # resumable JSONL caches
├── models/                    # trained surrogate checkpoints
├── runs/                      # self-contained PPO and evaluation runs
├── results/                   # metrics and reports
├── logs/                      # scheduled-task stdout/stderr
└── archive/                   # completed or superseded artifacts
```

Artifacts are not source code. A run directory should contain its configuration,
checkpoints, cache paths, metrics, and reports together.

## Historical code

`legacy/` is read-only historical material. Nothing under `legacy/` is a current
runtime entry point. Superseded scripts and experiments are moved there with a
manifest containing their original path, new path, size, and SHA256.

## Naming rules

- Use descriptive names based on purpose: `train_layerwise_topk.py`,
  `compare_general_assignment_baselines.py`, and `collect_general_assignment.py`.
- Keep one active command per purpose; do not create `v2`, `v3`, or date suffixes
  for source files.
- Put run dates and experiment variants in `artifacts/runs/<family>/<run-name>`.
- Put superseded source under `legacy/experiments_<period>/<family>/`.
- Keep experimental alternatives out of the active import path until they pass a
  true-model validation gate.

## Recent LLM baseline package

```text
src/uav_rl/system_baselines/
├── edge_shard_uav.py          # subset DP for device choice and contiguous shards
├── hexgen_search.py           # constrained topology-aware evolutionary search
└── exact_grouped_oracle.py    # exact enumeration inside a grouped action set

scripts/baselines/
├── compare_system_baselines.py
├── evaluate_frozen_system_baselines_true.py
└── run_exact_grouped_oracle.py
```

Algorithm definitions and reporting qualifications are fixed in
`docs/RECENT_LLM_BASELINES.md`.
