# UAV-RL

面向 UAV 协同 LLM 推理的可复现实验库。项目将链路质量、分层部署、激活 dropout 和延迟建模组合为一个强化学习环境；当前正式训练路径直接使用 CodeLlama perplexity（PPL）作为质量信号。

## 当前状态

下一轮 PPO 应使用**真实 PPL**训练，不使用 surrogate reward。

此前的多-seed surrogate 研究已经完整保留，但最新验证仍未通过预先设定的绝对误差门槛：tail Spearman 达标，整体和 tail MAE 未达标。因此这些结果只用于后续 surrogate 研究，不能作为新的 PPO 验收训练入口。

## 目录结构

```text
uav-rl/
├── src/uav_rl/                 # 当前实现：环境、PPO、真实 PPL 评估、数据与指标
│   ├── benchmarks/             # PPL 和历史策略评估实现
│   ├── data/                   # 数据集与可恢复 JSONL/NPZ 聚合逻辑
│   ├── metrics/                # PPL 指标
│   ├── models/                 # activation dropout 模型组件
│   └── rl/                     # 环境、actor-critic、oracle 和 PPO
├── scripts/                    # 仅保留当前可执行入口，见下表
│   ├── benchmarks/measure_ppl.py
│   ├── maintenance/verify_legacy_archive.py
│   └── ppo/train.py
├── tests/                      # 当前实现的自动测试
├── legacy/                     # 只读历史追溯区，不是当前运行入口
│   ├── prototype_v0/
│   ├── surrogate_v1_fixed_seed/
│   └── experiments_2026_08/    # 已结束的 v2–v5 surrogate / PPO 试验脚本
└── artifacts/                  # 不纳入 Git 的运行产物与缓存
    ├── archive/                # 已归档的历史产物
    ├── runs/ppo/<run-name>/    # 新 PPO 训练的完整自包含目录
    ├── cache/                  # 既有研究 cache（保留原位）
    ├── data/                   # 既有研究数据（保留原位）
    ├── models/                 # 既有 checkpoint（保留原位）
    └── results/                # 既有评估与报告（保留原位）
```

## 当前脚本入口

| 用途 | 命令 |
| --- | --- |
| 预估一次完整 PPL 的耗时 | `python scripts/benchmarks/measure_ppl.py --device cuda` |
| 训练或无损恢复真实 PPL PPO | `python scripts/ppo/train.py ...` |
| 仅补充当前 surrogate 的高方差训练标签 | `python scripts/surrogate/extend_labels.py --device cuda` |
| 重训并在 validation 上选择当前 surrogate ensemble | `python scripts/surrogate/train.py --device cuda` |
| 校验历史归档的 SHA256 | `python scripts/maintenance/verify_legacy_archive.py` |

`scripts/` 中不再保留版本号实验脚本。所有 v2–v5 数据生成、surrogate 训练、ablation、诊断、调度和审计代码都在 [legacy/experiments_2026_08/](legacy/experiments_2026_08/README.md)，并附有用途说明。

## 环境

建议 Python 3.10 或 3.11，并安装与本机 CUDA 匹配的 PyTorch。

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider
python -m ruff check src scripts tests
```

真实 PPL 的默认实验配置是：

- 模型：`codellama/CodeLlama-7b-hf`
- 语料：WikiText-2 test 的 27 个有效序列、1689 个 next-token
- 最大序列长度：512
- 默认 batch size：4
- 默认精度：BF16
- reward 标签：多 noise seeds 下 `log(PPL_noisy / PPL_clean)` 的均值

训练、验证和测试使用彼此隔离的 channel、deployment/drop-vector 和 noise seeds。每个完成的真实 PPL 计算会立即写进 JSONL；恢复时已完成键会被复用，而不是重算。

## 下一轮 PPO

先测一次真实 PPL 耗时，确认 GPU、模型和数据集路径正常：

```powershell
$env:PYTHONPATH = "src"
python scripts/benchmarks/measure_ppl.py --device cuda --dtype bfloat16
```

开始一个 1,000 episode 的真实 PPL PPO 训练：

```powershell
$env:PYTHONPATH = "src"
python scripts/ppo/train.py `
  --run-dir artifacts/runs/ppo/true-ppl-round-01 `
  --episodes 1000 `
  --device cuda
```

如需把同一实验扩展到 3,000 episodes，`--episodes` 表示**最终总数**，不是额外训练量：

```powershell
python scripts/ppo/train.py `
  --run-dir artifacts/runs/ppo/true-ppl-round-01 `
  --episodes 3000 `
  --device cuda `
  --resume
```

该恢复流程保存并恢复模型、optimizer、最佳 checkpoint、Python/NumPy/PyTorch RNG、channel RNG、noise RNG、完整 reward history 和运行元数据。除 `training_episodes` 外，任何训练配置或元数据变化都会被拒绝，因此恢复是无损的。

每次训练只会在自己的 `run-dir` 中写入：

```text
artifacts/runs/ppo/true-ppl-round-01/
├── run_config.json       # 启动时固定的配置与产物路径
├── ppl_cache.jsonl       # 完成即落盘的真实 PPL cache
├── training_state.pth    # 无损恢复状态
├── best_policy.pth       # 最佳验证 reward 的策略
└── evaluation.json       # 训练后在独立测试 seeds 上的 PPO / baseline 对比
```

默认训练每 4 个 rollout 做一次 validation（rollout size 默认 128，即约每 512 episodes），避免在长任务中频繁触发昂贵检查。默认训练时每个动作采用 4 个 noise seeds；validation/test 分别采用 16 个从未用于训练的 seeds。

如果只希望完成训练与保存状态、暂不执行昂贵的 held-out baseline 比较，可添加 `--skip-final-evaluation`。之后以相同 `run-dir --resume` 再运行即可执行完整收尾评估。

## 历史研究与归档

[legacy/README.md](legacy/README.md) 描述所有历史代码的范围。原始原型、fixed-seed surrogate，以及已完成的 surrogate v2–v5 实验均不应作为新的训练入口。

早期归档的路径、大小和 SHA256 记录在 [legacy/archive_manifest_2026-08-16.json](legacy/archive_manifest_2026-08-16.json)。可用以下命令验证：

```powershell
python scripts/maintenance/verify_legacy_archive.py
```

其他项目中的 Qwen3/28-layer 数据不属于本仓库，未复制、未移动、未登记。
