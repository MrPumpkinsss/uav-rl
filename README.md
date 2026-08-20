# UAV-RL

可复现的 UAV 协同 LLM 推理实验库。当前代码、脚本、实验产物和历史归档分层保存，避免把实验变体混入正式运行入口。

## 当前入口

| 用途 | 命令 |
| --- | --- |
| 真实 CodeLlama PPO | `scripts/ppo/train.py` |
| layerwise surrogate PPO + 原始 Top-K | `scripts/ppo/train_layerwise_topk.py` |
| frozen candidate 真实模型验证 | `scripts/ppo/validate_true_policy.py` |
| common-seed baseline 对比 | `scripts/ppo/compare_general_assignment_baselines.py` |
| surrogate 数据采集 | `scripts/surrogate/collect_general_assignment.py` |
| surrogate ensemble 训练 | `scripts/surrogate/train_general_assignment.py` |
| coverage / tail 诊断 | `scripts/surrogate/diagnose_coverage.py` |
| 单次真实 PPL 耗时 | `scripts/benchmarks/measure_ppl.py` |
| 归档校验 | `scripts/maintenance/verify_legacy_archive.py` |

```powershell
$env:PYTHONPATH = "src"
```

## 目录结构

```text
src/uav_rl/       当前 Python 实现
  data/           数据集、JSONL cache、label 聚合
  metrics/        PPL 和评估指标
  models/         activation dropout 组件
  benchmarks/     可复用 benchmark evaluator
  rl/             PPO、actor-critic、环境和 oracle
scripts/          当前可执行入口
  ppo/            PPO 训练、验证和 baseline
  surrogate/      surrogate 数据、诊断和训练
  benchmarks/     一次性测量
  maintenance/    仓库维护
tests/            单元测试和恢复性测试
artifacts/        数据、cache、模型、runs、结果、日志和 archive
legacy/           只读历史代码和已结束实验
docs/             项目结构和命名说明
```

更完整的结构说明见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 当前默认实验

当前 common-seed 真实验证使用 32 个 channels、4 个 activation-noise seeds 和 CodeLlama-7b：

- 原始 Top-K PPO Top-1 reward：`-0.400744`
- 原始 Top-K true oracle reward：`-0.391821`
- deterministic PPO reward：`-0.428221`
- 最强非 PPO proxy beam baseline：`-0.419458`

结果位于 `artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.*`。

Diverse Top-K 已完成验证但没有带来提升，代码和产物归档在：

```text
legacy/experiments_2026_08/ppo/diverse_topk/
artifacts/archive/2026-08-20/diverse_topk/
```

后续默认使用原始 Top-K，不使用 Diverse Top-K。

## 检查

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider
python -m ruff check src scripts tests
```

真实 PPL 任务会在每个 `(drop_vector, noise_seed)` 完成后写入 JSONL；中断后可从 cache 无损恢复。

## 命名规则

- 当前源码不使用 `v2`、`v3` 或日期后缀区分版本。
- 实验变体放在 `artifacts/runs/<family>/<run-name>`。
- 被替代的源码放在 `legacy/experiments_<period>/<family>/`。
- 旧文件移动前保留路径、大小和 SHA256 manifest。
- `legacy/README_previous.md` 是之前根 README 的历史备份，不是当前入口。

## 详细实验说明

这一节面向第一次接触本项目的读者，说明数据、surrogate、PPO 和真实模型验证之间的边界。

### 研究对象和 action 表示

当前场景有 32 个 LLM layers 和 5 个 UAV。一个 action 是长度为 32 的整数向量：

```text
deployment[layer] = uav_id
```

channel 是 `5 x 5` 的 UAV 链路增益矩阵。相邻 layer 分配给不同 UAV 时产生一个 boundary，因此共有 31 个可能的 boundary。每个 boundary 的 drop probability 由发送 UAV、接收 UAV 和 channel 计算得到。

`src/uav_rl/resource_assignment.py` 负责任意 layer-to-UAV assignment 的资源约束，包括 memory、计算能耗、通信能耗和 hover 能耗。当前 resource profile 是可复现实验场景参数，不是实际硬件规格；参数会写入数据 manifest 和 PPO run config。

`src/uav_rl/resource_environment.py` 统一计算 reward：先检查 assignment 是否可行，再得到 31 维 drop vector、计算/通信 latency 和质量项。训练时质量项来自 surrogate，真实验证时质量项来自冻结的 CodeLlama。

```text
reward = -(quality_weight * log(PPL_noisy / PPL_clean)
           + (1 - quality_weight) * normalized_latency)
```

不合法 assignment 的 reward 为 `-100`。

## Surrogate 数据集

### 真实标签如何产生

数据生成入口是 `scripts/surrogate/collect_general_assignment.py`，底层实现为 `src/uav_rl/data/general_assignment_dataset.py`。每一个 action 在多个 activation-noise seed 下调用真实 CodeLlama，得到 `log(PPL_noisy / PPL_clean)`，再聚合成该 action 的平均值。

当前 manifest 中记录的真实模型配置为：

- `codellama/CodeLlama-7b-hf`；
- WikiText-2 raw test split；
- 最多 50 个文本样本、最大长度 512；
- 实际评估 27 个序列、1689 个 token；
- clean PPL 约为 `12.87591`；
- 推理 dtype 为 `bfloat16`。

真实 PPL 很慢，所以生成器在每完成一个 `(action_id, noise_seed)` 后立即写入 JSONL。中断后再次启动会跳过已经完成的键；这也是长时间真实模型任务可以无损恢复的原因。

### 数据字段和 split

每个聚合 action 至少包含：`action_id`、`sample_source`、`group_id`、`channels`、`deployments`、`drop_probabilities`、`latency_seconds`、`log_ppl_ratio`、`log_ppl_ratio_std` 和 `noise_seed_count`。

生成器会检查 train/validation/test 之间不存在 noise seed、channel/deployment pair 或 drop vector 重叠。聚合后的当前数据规模以 `artifacts/data/general_assignment_manifest.json` 为准：

| split | action 数 | 当前 seed 数 | 用途 |
| --- | ---: | ---: | --- |
| train | 2920 | 4 或 24 | surrogate 参数训练 |
| validation | 64 | 16 | early stopping 和模型选择 |
| test | 64 | 16 | 最终评估，只运行一次 |

当前 train 集包含已保存的 PPO cache、coverage、random、strong-link、dynamic-programming、compute-greedy、tail 和 boundary/hazard 诊断来源。manifest 的 `legacy_train` 字段记录了被保留并合并的旧 train 文件；其他项目的 Qwen 数据不在本项目数据链路中。

当前数据和 cache 位于：

```text
artifacts/data/general_assignment_train.npz
artifacts/data/general_assignment_validation.npz
artifacts/data/general_assignment_test.npz
artifacts/data/general_assignment_manifest.json

artifacts/cache/general_assignment_surrogate_plan.json
artifacts/cache/general_assignment_surrogate_labels.jsonl
artifacts/cache/general_assignment_surrogate_ppl.jsonl
```

生成计划但不调用真实模型：

```powershell
$env:PYTHONPATH = "src"
python scripts/surrogate/collect_general_assignment.py --plan-only
```

开始或恢复真实数据生成：

```powershell
python scripts/surrogate/collect_general_assignment.py --device cuda --progress-interval 25
```

不要在已有 cache 上悄悄更改模型、文本、seed range 或 resource config；配置指纹不匹配时应拒绝恢复。

## Surrogate 是如何训练的

训练入口为 `scripts/surrogate/train_general_assignment.py`，核心实现为 `src/uav_rl/surrogate_training.py` 和 `src/uav_rl/surrogate.py`。

### 输入特征

surrogate 不是直接读取 deployment 整数向量，而是读取 31 维 boundary drop probabilities，并额外计算 5 个工程特征：

1. 所有 boundary drop 的总和；
2. 最大 boundary drop；
3. drop probability 的平方和；
4. 非零 boundary 的比例；
5. cumulative hazard：`sum(-log(1 - drop_i))`。

因此网络输入维度是 `31 + 5 = 36`，目标是 action 在多 noise seed 下的平均 `log(PPL_noisy / PPL_clean)`。

### 网络和优化

当前 checkpoint 是五模型 ensemble：

- 5 个独立初始化的 MLP；
- 每个 MLP 为 `36 -> 512 -> 512 -> 1`；
- 两个 hidden layer 使用 ReLU；
- 损失为 MSE；
- optimizer 为 AdamW，learning rate `1e-3`，weight decay `5e-4`；
- 最多 1500 epochs；
- validation loss 连续 250 epochs 没有改善则 early stop；
- 五个成员使用不同但可复现的随机 seed。

normalization 的 mean/std 只从 train split 计算，然后固定地用于 validation、test 和 PPO。这样可以避免 validation/test 信息泄漏。

推理时五个模型分别预测：

```python
mean, uncertainty = ensemble.predict_with_uncertainty(drop_probabilities)
```

PPO reward 使用 ensemble mean；ensemble 成员间标准差只用于诊断，目前没有加入 uncertainty penalty。checkpoint 同时保存模型参数、normalization、训练配置、数据 manifest hash、split SHA256、成员指标和 validation 指标。`load_surrogate()` 也兼容旧的单模型 checkpoint。

### 训练命令和输出

```powershell
$env:PYTHONPATH = "src"
python scripts/surrogate/train_general_assignment.py `
  --device cuda `
  --epochs 1500 `
  --patience 250
```

默认输入和输出：

```text
输入
  artifacts/data/general_assignment_train.npz
  artifacts/data/general_assignment_validation.npz
  artifacts/data/general_assignment_test.npz
  artifacts/data/general_assignment_manifest.json

输出
  artifacts/models/ppl_surrogate_general_assignment_ensemble.pth
  artifacts/results/ppl_surrogate_general_assignment_metrics.json
  artifacts/results/ppl_surrogate_general_assignment_report.md
  artifacts/results/ppl_surrogate_general_assignment_plots/
```

评估包含总体 MAE、RMSE、R²、Spearman、绝对误差 p50/p90/p95/max、按 `sample_source` 分组的指标、uncertainty 与真实误差的相关性，以及 grouped reward regret。

严格验收门槛是：test MAE ≤ 0.08、Spearman ≥ 0.90、RMSE ≤ 0.12、每个主要来源 MAE ≤ 0.12、平均 reward regret ≤ 2%、p90 regret ≤ 5%。

当前保存的 ensemble 的 test 结果为：

- MAE：`0.2211`
- RMSE：`0.2935`
- R²：`0.9655`
- Spearman：`0.9827`
- high-boundary MAE：`0.3422`
- low-boundary MAE：`0.1258`
- medium-boundary MAE：`0.2333`
- resource-balanced MAE：`0.1830`

因此当前 checkpoint 的排序能力较好（Spearman 达标），但绝对误差和 high/medium-tail 区域仍未通过严格验收。不能只因为 Spearman 高就宣称 surrogate 已完全可靠；这些指标来自 `artifacts/results/ppl_surrogate_general_assignment_metrics.json`。

## PPO 是如何训练的

当前正式 PPO 入口为 `scripts/ppo/train_layerwise_topk.py`，核心 trainer 为 `src/uav_rl/rl/layerwise_ppo.py`。

### policy 和 rollout

policy 是 autoregressive layerwise actor-critic：从第 0 层到第 31 层依次选择 UAV。每一步 observation 包含 channel、当前 layer index、各 UAV 已使用的 memory/energy、上一层 UAV 等信息。action mask 会屏蔽违反资源约束的 UAV。

当前 trainer 的 `max_policy_boundaries=4`，即正式入口限制 policy rollout 最多出现 4 次跨 UAV boundary；资源模型本身支持任意 layer-to-UAV assignment，但这个 policy-level 限制是当前实验设计的一部分，修改它会改变训练分布，应单独做实验。

### reward 和 PPO update

训练阶段的主要步骤是：

1. 采样一批 channel；
2. policy 逐层生成 deployment；
3. 用 surrogate 计算 drop vector 对应的质量项，并与 latency 合成为 reward；
4. 使用 PPO clipped objective、value loss 和 entropy bonus 更新 actor-critic；
5. 每个 rollout 记录 reward、latency、log PPL ratio 和 invalid fraction。

默认入口参数为：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--episodes` | 1000 | 训练 episode 数 |
| `--rollout-size` | 128 | 每次 PPO update 的 channel 数 |
| `--checkpoint-interval-episodes` | 200 | candidate checkpoint 间隔 |
| `--validation-interval` | 4 | surrogate validation 触发间隔 |
| `--top-k` | 5 | 真实验证时保留的候选数 |
| `--candidate-samples` | 20 | 每个 channel 生成的 rollout 候选数 |
| `--teacher-channels` | 256 | behavior-cloning teacher channel 数 |
| `--behavior-cloning-epochs` | 20 | warm start 的 BC epoch 数 |
| `--true-channels` | 32 | 最终真实模型验证 channel 数 |
| `--true-noise-samples` | 4 | 每个验证 channel 的真实 noise seed 数 |

训练入口包含 teacher warm start：对 teacher channel 采样若干可行 assignment，用 surrogate reward 选出较好的 teacher action，再进行 behavior cloning，之后才进入 PPO 微调。PPO 本身仍然负责后续策略更新，不是直接把 baseline 当作最终策略。

### PPO checkpoint 和无损恢复

每个 PPO run 默认写入：

```text
artifacts/runs/surrogate_ppo/layerwise_topk/
  run_config.json
  training_state.pth
  best_policy.pth
  candidate_policies/episode_*.pth
  topk_true_ppl_cache.jsonl
  topk_true_validation.json
```

`training_state.pth` 保存的不只是模型权重，还包括 optimizer state、当前 episode、best model、history、Python/NumPy/PyTorch/CUDA RNG state、channel RNG 和 noise RNG。因此使用 `--resume` 时可以从上次保存点继续，而不是重新初始化随机状态：

```powershell
python scripts/ppo/train_layerwise_topk.py `
  --episodes 3000 `
  --checkpoint-interval-episodes 200 `
  --resume
```

只有在确认 run directory 对应的 config 和 surrogate checkpoint 没有变化时才应 resume。改变 surrogate、resource config、seed 或 policy architecture 后应创建新的 run directory。

## Top-K 和真实模型验证

Top-K 不是训练时调用真实 CodeLlama 的捷径，而是降低 PPO 单次随机 rollout 选择误差的一种候选机制：

1. policy 对 held-out channel 生成多个 deployment candidate；
2. 用 surrogate reward 排序；
3. 取 surrogate 排名第一的 candidate 作为策略输出；
4. 同时保留 Top-K 内真实 reward oracle，作为候选质量上界；
5. 用真实 CodeLlama 对所有候选执行固定 noise seeds，报告 surrogate Top-1、Top-K oracle、deterministic policy 和其他 baseline。

默认真实验证使用 32 个 channel、noise seeds `4_100_000` 起始的 4 个 seed，并把每次 `(drop_vector, noise_seed)` 的真实结果写入 run 目录的 JSONL cache。真实验证不使用训练期的 PPO noise seed。

当前 common-seed 对比结果位于：

```text
artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.json
artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.md
```

相同 32 个 channel 和 4 个 noise seeds 上的均值 reward：

| 方法 | mean reward |
| --- | ---: |
| Original surrogate-selected Top-1 PPO | `-0.400744` |
| Original Top-5 true oracle | `-0.391821` |
| Deterministic PPO | `-0.428221` |
| Proxy Beam 128 | `-0.419458` |

Diverse Top-K 已做过独立验证，结果没有优于当前原始 Top-K，因此不属于当前正式入口。其源码和产物保存在：

```text
legacy/experiments_2026_08/ppo/diverse_topk/
artifacts/archive/2026-08-20/diverse_topk/
```

## 如何判断一个实验是否可以用于论文结论

至少要同时满足：

1. surrogate 数据 manifest 的 isolation audit 通过；
2. PPO 训练使用固定且可恢复的 checkpoint；
3. 验证和测试使用训练期未使用的 channel/noise seed；
4. 最终指标来自真实 CodeLlama，而不是 surrogate；
5. 所有 baseline 使用完全相同的 channel、noise seed、模型和 token 配置；
6. 报告 surrogate 误差、真实 reward、Top-K selection gap 和 invalid fraction；
7. 如果 surrogate 未达到 MAE/RMSE/source-MAE 门槛，应把结果标记为探索性实验，不应把 PPO 的 surrogate reward 提升直接解释为真实模型提升。

## 测试和代码质量检查

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider
python -m ruff check src scripts tests
```

测试重点包括数据 cache 去重与恢复、split 隔离、label 聚合、ensemble mean/uncertainty、checkpoint 恢复、旧单模型 checkpoint 兼容和真实模型 evaluator 的 fake end-to-end 流程。

## 给新成员的阅读顺序

建议按下面顺序阅读，而不是从旧 prototype 开始：

1. `src/uav_rl/config.py`：系统、PPO 和真实模型配置；
2. `src/uav_rl/resource_assignment.py`：action、boundary 和资源约束；
3. `src/uav_rl/resource_environment.py`：latency、quality 和 reward；
4. `src/uav_rl/surrogate.py`：单模型、ensemble 和 uncertainty；
5. `src/uav_rl/surrogate_training.py`：训练、验收和报告；
6. `src/uav_rl/rl/layerwise_ppo.py`：rollout、PPO update、checkpoint；
7. `scripts/surrogate/collect_general_assignment.py`：真实数据生成；
8. `scripts/surrogate/train_general_assignment.py`：surrogate 训练；
9. `scripts/ppo/train_layerwise_topk.py`：正式 PPO 训练和真实 Top-K 验证。

`legacy/` 下的文件只用于历史追溯，不是当前运行入口。
