# UAV-RL

可复现的 UAV 协同 LLM 推理实验库。当前代码、脚本、实验产物和历史归档分层保存，避免把实验变体混入正式运行入口。

## 术语速查

- **PPL**：perplexity，语言模型困惑度；本文中通常越低越好。
- **surrogate**：根据真实 CodeLlama 标签训练的快速近似模型，用来替代 PPO 训练中的昂贵真实 forward。
- **channel**：UAV 之间的无线链路条件矩阵。
- **boundary**：相邻两层由不同 UAV 执行时产生的跨 UAV 切换点。
- **drop vector**：31 个可能 boundary 对应的丢包概率向量。
- **Top-K**：先生成 K 个候选，再按 surrogate 或真实模型排序。
- **common-seed**：所有方法使用完全相同的 channel 和 noise seed，因此结果可以直接比较。

## 目录

- [术语速查](#术语速查)
- [当前入口](#当前入口)
- [目录结构](#目录结构)
- [当前推荐实验和结论](#当前推荐实验和结论)
- [当前最好模型](#当前最好模型)
- [实验结果与分析](#实验结果与分析)
- [检查](#检查)
- [命名规则](#命名规则)
- [详细实验说明](#详细实验说明)
  - [研究对象和 action 表示](#研究对象和-action-表示)
  - [Surrogate 数据集](#surrogate-数据集)
  - [Surrogate 是如何训练的](#surrogate-是如何训练的)
  - [PPO 是如何训练的](#ppo-是如何训练的)
  - [Top-K 和真实模型验证](#top-k-和真实模型验证)
- [实验复现与可信度检查](#实验复现与可信度检查)
- [测试和代码质量检查](#测试和代码质量检查)
- [给新成员的阅读顺序](#给新成员的阅读顺序)
- [理论链路与代码对应](#理论链路与代码对应)
  - [从 deployment 到质量损失](#1-从-deployment-到质量损失)
  - [为什么 surrogate 输入是 drop vector](#2-为什么-surrogate-输入是-drop-vector而不是-uav-编号)
  - [为什么 latency 不由 surrogate 预测](#3-为什么-latency-不由-surrogate-预测)
  - [数据生成、训练和 PPO 的代码地图](#4-数据生成训练和-ppo-的代码地图)
  - [当前 surrogate 的适用范围](#5-当前-surrogate-的适用范围)
- [复现实验时最重要的边界](#复现实验时最重要的边界)
- [训练好的模型和公开 checkpoint](#训练好的模型和公开-checkpoint)

## 当前入口

| 用途 | 命令 |
| --- | --- |
| 真实 CodeLlama PPO | `scripts/ppo/train.py` |
| layerwise surrogate PPO + 普通 Top-K | `scripts/ppo/train_layerwise_topk.py` |
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

## 当前推荐实验和结论

当前 common-seed 真实验证使用 32 个无线 channel、4 个 activation-noise seed 和 CodeLlama-7b。common-seed 的意思是：所有方法使用完全相同的 channel 和 noise seed，因此 reward 可以直接比较：

- 当前 200000 episode PPO Top-1 reward：`-0.389706`
- 当前实验中观测到的最佳 Top-K Top-1 reward：`-0.387903`（10000 episode 快照）
- 当前 200000 episode Top-5 true oracle reward：`-0.381480`
- 当前 200000 episode deterministic PPO reward：`-0.384838`
- 当前 common-seed 中最强可部署 baseline：CoEdge-style adaptive partition，reward `-0.390294`
- 当前最强非 PPO surrogate 搜索 baseline：surrogate simulated annealing，reward `-0.406119`
- 新增 dynamic programming baseline reward：`-0.480867`（可变长度连续区块，不是固定 4×8）

结果位于 `artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.*`。

Diverse Top-K 已完成验证但没有带来提升，代码和产物归档在：

```text
legacy/experiments_2026_08/ppo/diverse_topk/
artifacts/archive/2026-08-20/diverse_topk/
```

当前正式入口使用普通 Top-K 候选机制，不使用已归档的 Diverse Top-K 变体。当前结果对应 high-augmented surrogate + layerwise Top-K PPO。

## 当前最好模型

当前推荐的可部署模型只有一套：

| 组件 | 当前最好版本 | Git 路径 | 说明 |
| --- | --- | --- | --- |
| Surrogate | High-augmented 5-model ensemble | `artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth` | PPO 训练阶段的质量 reward |
| PPO policy | High-augmented layerwise Top-K PPO | `artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth` | 当前 common-seed 真实模型评估最好的可部署 policy |

该 policy 的训练方式是：

- autoregressive layerwise actor-critic，逐层选择 UAV；
- 训练 reward 使用 frozen surrogate，不在每个 PPO action 上调用真实 CodeLlama；
- 先用 256 个 teacher channels 和 24 个 teacher candidates 做 behavior-cloning warm start；
- 已完成 200000 episode PPO 微调；其中 1000 到 200000 episode 均从同一个 `training_state.pth` 无损续训；
- “Top-K”表示先生成多个候选；当前每个 channel 生成 20 个候选，用 surrogate 排序，实际部署选择其中排名第 1 的候选（Top-1）；真实验证另外保留前 5 个候选，计算不可部署的 Top-5 true oracle 上界；
- 最终用真实 CodeLlama 在 held-out channel/noise seed 上验证。

这里的“当前主线模型”指最新完成的 200000 episode policy artifact。需要区分两个指标：在 Top-K surrogate-selected Top-1 上，当前实验观察到的最好值仍是 10000 episode 快照（`-0.387903`）；在 deterministic deployment 上，200000 episode 是目前更好的 PPO 结果（`-0.384838`）。Top-5 true oracle 需要用真实 CodeLlama 在候选中事后挑选，不能作为实际部署策略。

### 3000 episode 延长训练结果（历史对照）

这组历史结果复用了原 high-augmented run 的 `training_state.pth`，只增加 `training_episodes`，没有重新初始化 policy、optimizer 或随机数状态。

在相同的 32 channel、4 noise seed、真实 CodeLlama 验证协议下，3000 episode 的结果为：

| 指标 | 1000 episode 快照 | 3000 episode |
| --- | ---: | ---: |
| PPO Top-1 true reward | -0.400744 | **-0.392712** |
| PPO deterministic true reward | -0.428221 | **-0.408486** |
| PPO Top-5 true oracle | -0.391821 | **-0.384987** |
| invalid fraction | 0 | 0 |

PPO Top-1 相比 1000 episode 快照提升约 2.00%，但仍略低于当前 CoEdge-style adaptive partition 的 -0.390294。这次结果是历史延长训练对照；后续 200000 episode 结果见下节。

验证结果保存在 `artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/topk_true_validation.json`，最终 policy 保存在 `artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth`。

### 200000 episode 延长训练结果

本轮从 10000 episode 的 `training_state.pth` 无损续训到 200000 episode，每 20000 episode 保存一个 checkpoint。训练继续复用原有 policy、optimizer、surrogate、channel RNG 和 noise RNG，没有重新初始化；因此这是同一实验轨迹的延长，而不是新的独立随机实验。

新增 checkpoint 为：

```text
candidate_policies/episode_020000.pth
candidate_policies/episode_040000.pth
candidate_policies/episode_060000.pth
candidate_policies/episode_080000.pth
candidate_policies/episode_100000.pth
candidate_policies/episode_120000.pth
candidate_policies/episode_140000.pth
candidate_policies/episode_160000.pth
candidate_policies/episode_180000.pth
candidate_policies/episode_200000.pth
```

在相同的 32 channel、4 noise seed、真实 CodeLlama 验证协议下：

| 指标 | 10000 episode | 200000 episode |
| --- | ---: | ---: |
| PPO Top-1 true reward | **-0.387903** | -0.389706 |
| PPO deterministic true reward | -0.399736 | **-0.384838** |
| PPO Top-5 true oracle | **-0.378790** | -0.381480 |
| invalid fraction | 0 | 0 |

200000 episode 让 deterministic policy 相比 10000 episode 提升约 3.73%，但 Top-K surrogate-selected Top-1 下降约 0.46%，Top-5 true oracle 下降约 0.71%。这说明继续增加 PPO episode 已经不是当前主要瓶颈：policy 的确定性输出在变好，但 surrogate 对多候选的排序仍会引入选择误差。200000 episode 的 Top-1 reward `-0.389706` 仍略优于 CoEdge-style adaptive partition 的 `-0.390294`，差距约 0.15%，因此需要在更大的 held-out 集合上复验稳定性，不能只凭这一组 32 channel 宣称显著领先。

最终真实验证结果保存在 `artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/topk_true_validation.json`；最新 policy 保存在同目录的 `best_policy.pth`。训练状态 `training_state.pth` 仍保留，可继续 resume，但下一步更值得优先改进 Top-K candidate ranking 或评估 deterministic deployment，而不是盲目增加 episode。
其他已上传模型是对照或历史模型，不是并列的当前入口。这里的 `high-augmented` 表示训练数据中额外加入了高 boundary / 高丢包区域样本，目的是改善困难区域的排序能力：

- `ppl_surrogate_general_assignment_ensemble.pth`：通用 surrogate 对照版本；
- `layerwise_topk_2026-08-20b/best_policy.pth`：没有 high-boundary augmentation 的原始 Top-K 对照；
- `ppo_true_ppl_multiseed_best.pth`：直接用真实 PPL 训练的独立实验，不使用 surrogate，主要用于研究对照。

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

`PPL` 是 perplexity（语言模型困惑度）的缩写；在本项目中，PPL 越低通常表示 LLM 生成质量越好。`channel` 是一个 `5 x 5` 的 UAV 无线链路增益矩阵，表示任意两台 UAV 之间的通信条件。相邻 layer 分配给不同 UAV 时会产生一个 boundary，也就是一次跨 UAV 切换，因此共有 31 个可能的 boundary。每个 boundary 的 drop probability 由发送 UAV、接收 UAV 和 channel 共同决定。

`src/uav_rl/resource_assignment.py` 负责任意 layer-to-UAV assignment 的资源约束，包括 memory、计算能耗、通信能耗和 hover 能耗。当前 resource profile 是可复现实验场景参数，不是实际硬件规格；参数会写入数据 manifest 和 PPO run config。

`src/uav_rl/resource_environment.py` 统一计算 reward：先检查 assignment 是否可行，再得到 31 维 drop vector、计算/通信 latency 和质量项。训练时质量项来自 surrogate，真实验证时质量项来自冻结的 CodeLlama。

```text
reward = -(quality_weight * log(PPL_noisy / PPL_clean)
           + (1 - quality_weight) * normalized_latency)
```
`quality_weight` 控制质量损失和延迟的权衡；`normalized_latency` 是总延迟除以固定的 latency reference。因为 reward 是负的综合代价，所以越接近 0 越好；不合法 assignment 的 reward 为 `-100`。

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
| train | 2920 个 action | 每个 action 使用 4 或 24 个 noise seed（不同来源不同） | surrogate 参数训练 |
| validation | 64 个 action | 每个 action 使用 16 个 noise seed | early stopping 和模型选择 |
| test | 64 个 action | 每个 action 使用 16 个 noise seed | 最终评估，只运行一次 |

当前 train 集合并了 PPO cache、coverage、random、strong-link、dynamic-programming、compute-greedy、tail 和 boundary/hazard 诊断来源，因此历史 train action 的 seed 数可能是 4 或 24；validation/test 使用固定的 16 个 seed。manifest 的 `legacy_train` 字段记录了被保留并合并的旧 train 文件；其他项目的 Qwen 数据不在本项目数据链路中。

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

surrogate 是用真实 CodeLlama 标签训练的快速近似模型。它不是直接读取 deployment 整数向量，而是读取 31 维 boundary drop probabilities，并额外计算 5 个工程特征：

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

评估包含总体 MAE、RMSE、R²、Spearman、绝对误差 p50/p90/p95/max、按 `sample_source` 分组的指标、uncertainty 与真实误差的相关性，以及 grouped reward regret。这里的 MAE/RMSE 都是在 `log(PPL_noisy / PPL_clean)` 这个目标上计算的，不是直接的 PPL 误差；例如 MAE=0.22 表示预测的 log-PPL ratio 平均相差约 0.22。`high-boundary` 指跨 UAV 切换次数较多、通常更容易出现高丢包和高质量损失的区域。

项目曾定义一组内部 surrogate 质量 gate（MAE、RMSE、Spearman 和 regret），用于决定是否继续追加数据；这些 gate 是实验管理工具，不是 PPO 算法定义，也不替代最终真实模型验证。

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

当前 trainer 的 `max_policy_boundaries=4`，表示一次 rollout 最多允许 4 次“从一台 UAV 切换到另一台 UAV”。它不表示每台 UAV 最多只能放 8 层，也不表示只能使用 4 台 UAV：boundary 可以出现在任意两个相邻 layer 之间，所以 PPO 可以生成 1 到 5 个连续区块，区块长度也不要求相等。区块数也不等于 UAV 数量，因为 policy 可以在后面的 layer 切回之前用过的 UAV。资源模型本身支持任意 layer-to-UAV assignment；这个最多 4 次切换是当前 PPO policy 的实验限制，修改它会改变训练分布，应单独做实验。

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
| `--checkpoint-interval-episodes` | 500 | candidate checkpoint 间隔；每 500 episode 保存一个候选 checkpoint |
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
  --episodes 200000 `
  --checkpoint-interval-episodes 20000 `
  --resume
```

本轮延长训练会复用 `layerwise_topk_high_augmented_2026-08-20/training_state.pth`，只允许增加
`training_episodes`，不重新初始化 policy、optimizer 或随机数状态。由于已有状态正好完成了 1000
episode，新增候选 checkpoint 会落在 episode 1500、2000、2500 和 3000；原有的 200 间隔历史文件保留不覆盖。

只有在确认 run directory 对应的 config 和 surrogate checkpoint 没有变化时才应 resume。改变 surrogate、resource config、seed 或 policy architecture 后应创建新的 run directory。

## Top-K 和真实模型验证

Top-K 不是训练时调用真实 CodeLlama 的捷径，而是降低 PPO 单次随机 rollout 选择误差的一种候选机制。这里有两个不同的 K：

- **训练和实际部署**：每个 channel 生成 20 个候选 deployment，用 surrogate reward 快速排序，选择第 1 名作为实际输出；
- **真实模型验证**：从候选中保留前 5 个，分别用真实 CodeLlama 计算。第 1 名用于报告实际策略效果，5 个候选中的真实最优值只作为不可部署的 oracle（真实模型上界）。

完整流程是：

1. policy 在没有参与训练的 held-out channel 上生成多个候选部署方案；
2. 用 surrogate reward 排序；
3. 取 surrogate 排名第一的方案作为实际策略输出；
4. 同时保留前 5 个候选，测量真实模型上界；
5. 用固定 noise seeds 运行真实 CodeLlama，报告实际 Top-1、Top-5 oracle、deterministic policy 和其他 baseline。

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

Diverse Top-K 已做过独立验证，结果没有优于当前普通 Top-K，因此不属于当前正式入口。其源码和产物保存在：

```text
legacy/experiments_2026_08/ppo/diverse_topk/
artifacts/archive/2026-08-20/diverse_topk/
```

## 实验结果与分析

### 评估协议

下面的结果来自同一套严格 common-seed 真实模型评估。这里的 common-seed 意味着所有方法使用完全相同的 channel、noise seed、资源约束、模型和 token 配置，因此 reward 可以直接比较：

- 模型：`codellama/CodeLlama-7b-hf`；
- corpus：27 个有效序列、1689 个 evaluated tokens；
- channels：32 个，由 `generate_resource_channels(32, 20260824)` 生成；
- activation-noise seeds：`4100000`、`4100001`、`4100002`、`4100003`；
- 所有方法使用相同的 resource/memory/energy constraints；
- 所有 reward、PPL 和 latency 都由同一个真实 CodeLlama evaluator 重新计算；
- 这些 validation 结果不回写 surrogate 训练数据。

结果文件：

```text
artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.json
artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.md
```

### 结果表

reward 是负的综合代价，因此越接近 0 越好；PPL 和 latency 越小越好。`ppo_topk_true_oracle` 是不可部署的候选上界，不能和实际 policy 等价比较。

| 方法 | 算法类别 | 真实 reward ↑ | PPL ↓ | latency (s) ↓ | 相对当前 PPO |
| --- | --- | ---: | ---: | ---: | ---: |
| **High-augmented PPO Top-1** | **当前推荐：surrogate PPO + Top-K** | **-0.400744** | 16.735 | 0.713 | — |
| PPO Top-5 true oracle | 候选集真实模型上界（不可部署） | **-0.391821** | 16.407 | 0.714 | +2.23% |
| Proxy beam 128 | 非 PPO beam search | -0.419458 | 17.567 | **0.701** | -4.67% |
| **Surrogate local search** | **beam 512 初始化 + surrogate 局部搜索** | **-0.411060** | 17.245 | 0.704 | -2.57% |
| **CoEdge-style adaptive partition** | **自适应逐层边际代价分配** | **-0.390294** | 16.228 | 0.722 | **+2.61%** |
| **Surrogate simulated annealing** | beam 512 初始化 + surrogate 模拟退火 | -0.406119 | 16.986 | 0.708 | -1.34% |
| Constrained genetic surrogate | 约束遗传算法 + frozen surrogate | -0.417787 | 17.480 | 0.704 | -4.25% |
| MILP proxy oracle | 线性化 proxy 的 SciPy/HiGHS 求解器参考 | -0.421463 | 17.658 | 0.700 | -5.17% |
| Neurosurgeon-style best split | 两 UAV 单切分点枚举 | -0.419458 | 17.567 | 0.701 | -4.67% |
| Wide proxy beam 512 | beam search（宽度 512） | -0.421463 | 17.658 | 0.700 | -5.17% |
| PPO deterministic | 同一 PPO policy 的确定性 rollout | -0.428221 | 17.728 | 0.722 | -6.86% |
| Fixed-eight Strong-link | 历史固定分段：32 层先切成 4 段，每段 8 层，再按通信链路质量分配 UAV | -0.475804 | **15.310** | 1.027 | -18.73% |
| **Dynamic programming** | **可变长度连续区块；当前资源模型下的 DP 加性 proxy** | **-0.480867** | 15.619 | 1.014 | -19.99% |
| Surrogate random search 1024 | 1024 个可行 assignment 的 surrogate 搜索 | -0.488966 | 16.232 | 0.991 | -22.01% |
| Random feasible | 随机可行 assignment | -3.383793 | 860.959 | 5.289 | -744.38% |

### 各个方法到底在做什么

- **PPO Top-1**：当前推荐方法。PPO 根据 channel 和资源状态，从第 0 层到第 31 层逐层决定每一层交给哪台 UAV。每个 channel 生成 20 个候选部署方案，用 surrogate reward 快速排序，实际输出排名第 1 的方案；最终 reward 由真实 CodeLlama 测量。
- **PPO deterministic**：使用同一个训练好的 PPO policy，但每一步都只选概率最高的 UAV，不生成多个候选，用来衡量 Top-K 候选机制是否有帮助。
- **PPO Top-5 true oracle**：先让 PPO 生成 5 个候选，再用真实 CodeLlama 逐个计算，从中选真实 reward 最好的一个。它是候选集合的上界，但需要事后运行 5 次真实模型，因此不可部署。
- **Proxy beam 128**：不训练 PPO，而是从第 1 层开始逐层扩展任意 layer-to-UAV assignment，每一轮只保留近似分数最好的 128 条部分路径，最后再用真实 CodeLlama 评估完整方案。它允许在任意层切换 UAV，是当前最强的非 PPO 对照之一。
- **Wide proxy beam 512**：与 beam 128 使用相同的逐层任意 assignment 搜索，但保留 512 条部分路径。它不再受固定四段或每段八层限制；搜索阶段只用解析 proxy，不调用真实 CodeLlama。
- **Surrogate local search**：先用 beam 512 生成可行初始方案，再用 frozen surrogate reward 和解析 latency 做 3 轮局部改进，尝试单层迁移和连续区块迁移。它仍然只在最终统一 benchmark 阶段调用真实 CodeLlama。
- **Fixed-eight Strong-link（历史固定分段 + 链路优先）**：这是最早的一类固定结构 baseline。它先把 32 层强制切成 4 个连续区块，每个区块正好 8 层，再把这 4 个区块交给 4 台不同的 UAV。当前配置有 5 台 UAV，所以会有 1 台不参与部署。因为有 5 台 UAV、需要按顺序选择其中 4 台，所以最多有 `5 × 4 × 3 × 2 = 120` 种区块分配顺序；之后再删除违反 memory/energy 约束的方案，并只根据 3 次区块之间的跨 UAV 链路丢包近似值选择顺序。它不能改变区块边界，也不能自由决定使用几台 UAV；通常能得到较低 PPL，但可能带来较高的通信和分段延迟。

  示意如下：

  ```text
  layer  0–7    → UAV A
  layer  8–15   → UAV B
  layer 16–23   → UAV C
  layer 24–31   → UAV D
  ```

- **Dynamic programming（可变长度连续区块）**：这是当前资源模型下新增的 DP baseline。它不固定 4×8 层，而是在任意 layer 之间选择 boundary，每个区块长度可以不同（不超过 8 层），并从 4 或 5 台不同 UAV 中选择区块顺序；先用“边界丢包总量 + 归一化总延迟”的加性 proxy 做动态规划，再用真实 CodeLlama 评估最终 reward。它比 fixed-eight 更灵活，但仍不允许后续切回已经使用过的 UAV，也不直接优化真实非线性 PPL，所以不能与 PPO 的完整逐层策略等同。
- **Surrogate random search 1024**：每个 channel 随机生成 1024 个资源可行的任意 assignment，用 surrogate reward 选最优者，不做 PPO policy learning。
- **CoEdge-style adaptive partition**：逐层读取当前 UAV 的计算、memory 和 energy 余量，比较继续放置与切换 UAV 的边际 proxy cost，动态决定连续区块数量和边界；它不固定四段，也不使用真实 CodeLlama 搜索。
- **Surrogate simulated annealing**：以 beam 512 方案为初始点，用 surrogate reward 评价单层/连续区块迁移；允许以递减概率接受暂时变差的动作，减少局部最优。
- **Constrained genetic surrogate**：把每层的 UAV 编号作为 chromosome，使用单点交叉、可行性修复和 mutation，在 frozen surrogate reward 上演化；本轮配置为 population 32、16 generations。
- **MILP proxy oracle**：用 SciPy/HiGHS 求解线性化的逐层计算、边界丢包和传输 latency proxy，并最终重新检查真实资源约束；它是 proxy 参考，不是 CodeLlama 全局 oracle。
- **Neurosurgeon-style best split**：枚举一个 split point 和两台 UAV，选择 additive proxy 最好的前缀/后缀切分；这是文献结构基线，不代表当前任意多 UAV assignment 的完整搜索。
- **Random feasible**：只保证 memory/energy 可行，不使用质量或 latency 目标。

这些 baseline 的实现分别在 [`src/uav_rl/resource_baselines.py`](src/uav_rl/resource_baselines.py) 和 [`scripts/ppo/compare_general_assignment_baselines.py`](scripts/ppo/compare_general_assignment_baselines.py)。

### 实验分析

1. **CoEdge-style adaptive partition 在本轮 common-seed 结果中超过 PPO。** 它的真实 reward 为 `-0.390294`，比 PPO Top-1（`-0.400744`）高约 `2.61%`；但它是逐层贪心启发式，不是 PPO policy，因此需要额外测试其跨 seed 稳定性。
2. **PPO Top-5 true oracle 仍不是全局上界。** 它只在 PPO 自己生成的 5 个候选中用真实模型选择，reward 为 `-0.391821`；CoEdge 超过它说明 CoEdge 找到了 PPO 候选集之外的更好部署。
3. **模拟退火是当前搜索型 baseline 中最强的 surrogate 方法。** reward 为 `-0.406119`，优于 local search（`-0.411060`）和 GA（`-0.417787`），说明接受少量暂时变差的动作有助于跳出局部最优。
4. **GA 本轮没有超过 beam/local search。** 约束修复保证了所有 assignment 可行，但在有限 population/generation 预算下，surrogate 搜索仍有较大排序误差。
5. **MILP proxy oracle 与 beam 512 结果完全相同。** 这不是因为两者等价于真实全局最优，而是当前 MILP 只优化线性化 proxy，且非线性共享带宽 latency 和 CodeLlama PPL 没有进入 solver 目标。
6. **Neurosurgeon-style 单切分与 beam 128 结果相同。** 在当前资源约束和 proxy 下，最优单切分没有超过允许任意 assignment 的 beam 方法，说明多段/多 UAV 分配仍然是必要的搜索空间。
7. **local search 和 beam 宽度的结论仍成立。** local search 比 beam 512 提升约 `2.47%`，但低于 PPO；beam 512 仍略低于 beam 128，说明单纯扩大 additive proxy beam 不能解决目标错配。
8. **结果需要进一步做稳定性确认。** CoEdge 的优势目前来自同一套 32 channels/4 noise seeds 的一次正式 benchmark；在宣布它为新默认 baseline 前，应在更多 held-out channels 和 noise seeds 上复验。

当前结论应写成：**在统一 common-seed、真实 CodeLlama、32 channels/4 noise seeds 的评估中，high-augmented surrogate-trained Top-K PPO 是当前最好的可执行 PPO policy；它仍然落后于不可部署的 Top-5 true oracle，且并不意味着 surrogate 本身在所有 tail 区域都已达到低绝对误差。**
## 实验复现与可信度检查

为了让一次实验具备可复现性和可解释性，建议至少做到：

1. surrogate 数据 manifest 的 isolation audit 通过；
2. PPO 训练使用固定且可恢复的 checkpoint；
3. 验证和测试使用训练期未使用的 channel/noise seed；
4. 最终指标来自真实 CodeLlama，而不是 surrogate；
5. 所有 baseline 使用完全相同的 channel、noise seed、模型和 token 配置；
6. 报告 surrogate 误差、真实 reward、Top-K selection gap 和 invalid fraction；
7. 记录 surrogate 的误差和适用范围；无论 surrogate 指标如何，PPO 的最终提升都必须由真实 CodeLlama 验证。

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

## 理论链路与代码对应

这一节解释项目为什么这样建模，而不只是列出命令。核心思想是把“部署决策”“无线传输造成的 activation 损伤”“LLM 质量”和“系统延迟”拆成可以分别检查的模块。

### 1. 从 deployment 到质量损失

PPO 产生的是完整部署策略，而不是丢包率：

```text
deployment[layer] = uav_id
```

例如一个 32 层部署可以写成：

```text
[0, 0, 0, 1, 1, 2, 2, ...]
```

这个例子表示前 3 层交给 UAV 0，接着 2 层交给 UAV 1，再接着 2 层交给 UAV 2；相邻数字发生变化的位置就是 boundary，UAV 编号范围是 0 到 4。

环境沿着 layer 顺序检查相邻层。如果两层在同一 UAV，则不需要跨 UAV 传输；如果两层属于不同 UAV，则在这个 boundary 传输 activation，并由对应无线链路得到 drop probability：

```text
p_i = packet_drop_probability(
    channel[deployment[i], deployment[i + 1]]
)
```

因此一个 deployment 和一个 channel 会被转换为 31 维、有位置含义的向量：

```text
[p_0, p_1, ..., p_30]
```

转换实现位于 [`layerwise_drop_probabilities()`](src/uav_rl/resource_assignment.py)，调用链是：

```text
ResourceDeploymentEnvironment.evaluate()
  -> layerwise_drop_probabilities()
  -> SurrogateQualityEvaluator.evaluate()
```

### 2. 为什么 surrogate 输入是 drop vector，而不是 UAV 编号

在当前 activation-dropout 假设下，CodeLlama 质量损失取决于“哪一个 layer boundary 以多大概率损失 activation”，而不是 UAV 编号这个整数本身。UAV 0、UAV 1 等编号没有物理语义；物理语义来自它们之间的 channel gain，最终已经体现在 `p_i` 中。

使用 drop vector 有三个好处：

- 它保留了 dropout 发生的 layer 位置，`p_3` 和 `p_20` 不会被混为一谈；
- 它把 deployment 和 channel 的无线部分先转换成物理上有意义的表示，减少 surrogate 需要自己学习的组合关系；
- 不同 deployment 如果产生相同的 boundary drop pattern，可以复用同一个质量模型，而不必把每个 UAV ID 组合都当成完全不同的样本。

所以当前的职责划分是：

```text
deployment + channel
    ├── memory/energy feasibility
    ├── computation/communication latency
    └── 31-dim drop vector -> surrogate predicts log(PPL_noisy / PPL_clean)
```

surrogate 并没有丢掉 deployment。deployment 仍然由 PPO policy 生成，并由环境用于资源约束和 latency 计算；只是 surrogate 专门负责其中的 LLM quality 子问题。

### 3. 为什么 latency 不由 surrogate 预测

当前 latency 有明确的解析计算方式，不需要用神经网络近似：

- `layerwise_latency()` 计算每个 layer 在对应 UAV 上的计算时间；
- 对每个跨 UAV boundary，根据 channel gain 和 activation size 计算通信时间；
- `ResourceDeploymentEnvironment.evaluate()` 将 quality 和 normalized latency 合并成 reward。

因此 surrogate 的目标只有：

```text
f(drop_vector) = log(PPL_noisy / PPL_clean)
```

而不是：

```text
f(deployment, channel) = full reward
```

完整 reward 的组合在 [`resource_environment.py`](src/uav_rl/resource_environment.py) 中完成。这样可以把可解释的系统物理计算和需要真实 CodeLlama 才能得到的质量计算分开。

### 4. 数据生成、训练和 PPO 的代码地图

| 理论步骤 | 代码位置 | 作用 |
| --- | --- | --- |
| 读取 WikiText、tokenize、计算 clean PPL | [`data/ppl_dataset.py`](src/uav_rl/data/ppl_dataset.py)、[`metrics/perplexity.py`](src/uav_rl/metrics/perplexity.py) | 得到固定 `PPL_clean` |
| 真实 noisy PPL | [`true_quality.py`](src/uav_rl/true_quality.py) | 对 `(drop_vector, noise_seed)` 调用 CodeLlama 并写 JSONL cache |
| 生成 action/channel 计划 | [`general_assignment_dataset.py`](src/uav_rl/data/general_assignment_dataset.py) | 采样可行 deployment、分配 split 和 noise seeds |
| 聚合多-seed 标签 | [`general_assignment_dataset.py`](src/uav_rl/data/general_assignment_dataset.py) | 得到 action 平均 log-PPL ratio、标准差和 NPZ |
| surrogate 网络 | [`surrogate.py`](src/uav_rl/surrogate.py) | 31 个 drop 特征加 5 个工程特征，输出 log-PPL ratio |
| ensemble 训练 | [`surrogate_training.py`](src/uav_rl/surrogate_training.py) | 训练 5 个 MLP、validation early stopping 和 test 报告 |
| policy observation/action | [`layerwise_policy.py`](src/uav_rl/rl/layerwise_policy.py) | 逐层选择 UAV，并应用 action mask |
| PPO rollout/update | [`layerwise_ppo.py`](src/uav_rl/rl/layerwise_ppo.py) | surrogate reward、PPO clipped update、BC warm start 和 checkpoint |
| Top-K 候选 | [`train_layerwise_topk.py`](scripts/ppo/train_layerwise_topk.py) | 生成候选、surrogate 排序并交给真实模型验证 |
| baseline 对比 | [`compare_general_assignment_baselines.py`](scripts/ppo/compare_general_assignment_baselines.py) | 在相同 channel/noise seed 上比较方法 |

### 5. 当前 surrogate 的适用范围

当前设计隐含的建模假设是：在这个实验环境中，LLM 质量变化主要由 layer boundary 的 activation drop pattern 决定。如果未来引入不同 UAV 的量化精度、硬件数值误差、不同模型副本、重传协议或与 token 内容相关的丢包，那么 drop vector 可能不再是充分表示。

这种情况下应扩展 surrogate 输入，例如加入 deployment embedding、UAV compute speed、资源利用率、channel 特征或 latency 特征；但这会改变数据生成、模型训练和验收协议，不能只改网络输入维度而继续复用旧标签。

## 复现实验时最重要的边界

- surrogate 训练数据的标签来自真实 CodeLlama，但 PPO 训练阶段只调用 surrogate，避免每个 PPO action 都重复真实 forward；
- 最终 policy 验证和 baseline 对比必须调用真实 CodeLlama；
- validation/test 使用训练期没有出现的 channel 和 noise seeds；
- reward 的提升必须以真实模型结果为准，不能只看 surrogate reward 曲线；
- `artifacts/` 中的 checkpoint、cache 和报告是实验产物，算法逻辑在 `src/uav_rl/`，命令入口在 `scripts/`。

## 训练好的模型和公开 checkpoint

训练好的模型已经随本仓库 push 到 GitHub。下面只列当前有明确用途的 inference checkpoint；旧 tail/gated/residual 变体和大多数实验 cache 不作为当前公开入口。

| 状态 | 类型 | Git 路径 | 用途 | 大小 | SHA256 |
| --- | --- | --- | --- | ---: | --- |
| 对照 | General surrogate ensemble | [`artifacts/models/ppl_surrogate_general_assignment_ensemble.pth`](artifacts/models/ppl_surrogate_general_assignment_ensemble.pth) | 当前通用 assignment surrogate 默认 checkpoint | 5.41 MiB | `c2fd82f0df6e56e83a80bc492b9f9460ec3fd09210b52d53ea3dd02418f921ff` |
| **当前最好 surrogate** | **High-augmented surrogate ensemble** | [`artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth`](artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth) | common-seed baseline 对比使用的 surrogate | 5.41 MiB | `74a472278231db33560f6a57801a0af25a91d5d6bfa18a67bfe8fc203b0d84df` |
| 对照 | Original Top-K PPO | [`artifacts/runs/surrogate_ppo/layerwise_topk_2026-08-20b/best_policy.pth`](artifacts/runs/surrogate_ppo/layerwise_topk_2026-08-20b/best_policy.pth) | 1000-episode 原始 Top-K PPO policy，使用通用 surrogate | 1.10 MiB | `be93b847e2bfb8cc1a889e8b0a36b034e05cb09568270731426d84554d176f48` |
| **当前最好 policy** | **High-augmented Top-K PPO** | [`artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth`](artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth) | common-seed baseline 报告实际使用的 policy | 1.10 MiB | `f2ab36cde8eacf85bab27b758e14982e71321c934e6fccaf9f9c5cb94b949088` |
| 独立对照 | Direct true-PPL PPO | [`artifacts/models/ppo_true_ppl_multiseed_best.pth`](artifacts/models/ppo_true_ppl_multiseed_best.pth) | 不使用 surrogate、直接用真实 CodeLlama PPL 训练的 1000-episode PPO best policy | 4.84 MiB | `e424300aa7c113d72402cb4660873b43fe668d4a2d8e6d2fa1b9be9edf5b19f4` |

文件大小使用 MiB 展示；SHA256 以 checkpoint 原始字节计算。模型可以 clone 后直接加载：surrogate 使用 `load_surrogate()`，PPO policy 使用对应 run 的 `SystemConfig`、resource config 和 policy 类型。

### 对应入口和结果

- 通用 surrogate 训练：`scripts/surrogate/train_general_assignment.py`
- Top-K PPO 训练：`scripts/ppo/train_layerwise_topk.py`
- 真实 PPL PPO 训练：`scripts/ppo/train.py`
- high-augmented Top-K 配置：`artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/run_config.json`
- common-seed baseline：`artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.json` 和 `.md`
- 真实 PPL PPO 报告：`artifacts/results/ppo_true_ppl_multiseed_1000_report.md`

### 不上传的训练状态

`training_state.pth` 包含 optimizer、训练 history、Python/NumPy/PyTorch/CUDA 随机数状态，主要用于本地无损 resume，不是 inference 所必需的模型，所以没有纳入公开 checkpoint。根目录 `.gitignore` 仍然默认忽略其他 `artifacts/`；上表中的模型是经过筛选后显式加入 Git 的文件，不代表本地所有 `.pth`、cache 和结果都已上传。

clone 后可用下面的命令检查文件完整性：

```powershell
Get-FileHash -Algorithm SHA256 artifacts/models/ppl_surrogate_general_assignment_ensemble.pth
Get-FileHash -Algorithm SHA256 artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth
Get-FileHash -Algorithm SHA256 artifacts/models/ppo_true_ppl_multiseed_best.pth
Get-FileHash -Algorithm SHA256 artifacts/runs/surrogate_ppo/layerwise_topk_2026-08-20b/best_policy.pth
Get-FileHash -Algorithm SHA256 artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth
```
