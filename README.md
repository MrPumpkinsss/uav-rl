# UAV-RL：无人机协同 LLM 分层部署实验库

本仓库研究在动态无线信道、异构算力、内存和能量约束下，将 LLM Transformer layers 分配到多台 UAV，并联合优化推理质量与端到端时延。

当前论文主线不是证明 PPO 是最强强化学习算法，而是回答：

1. 如何根据 UAV 资源和瞬时链路条件选择 layer-to-UAV deployment；
2. 如何控制跨 UAV boundary 引起的 activation 丢包和模型质量退化；
3. 学习型在线策略能否达到有竞争力的质量—时延折中，同时减少逐 channel 搜索成本；
4. surrogate 筛选结果能否通过冻结 deployment 的真实 LLM 评估得到确认。

> **证据边界（必须保留）**：截至 2026-09-01，新增 EdgeShard-UAV、HexGen-inspired 和 LinguaLinked-UAV 已完成 64-channel **surrogate benchmark**，但尚未完成匹配模型的 true-LLM 复验。下面的新增方法排名不能写成最终真实 PPL 排名。

---

## 目录

- [1. 当前状态](#1-当前状态)
- [2. 问题定义](#2-问题定义)
- [3. PPL 和 reward 的计算](#3-ppl-和-reward-的计算)
- [4. 方法与 baseline](#4-方法与-baseline)
- [5. 2026-08-31 系统方法 benchmark](#5-2026-09-01-系统方法-benchmark)
- [6. Grouped exact oracle](#6-grouped-exact-oracle)
- [7. 已有真实 CodeLlama 证据](#7-已有真实-codellama-证据)
- [8. Surrogate 数据与训练链路](#8-surrogate-数据与训练链路)
- [9. PPO 与 RL 消融](#9-ppo-与-rl-消融)
- [10. 实验复现](#10-实验复现)
- [11. 项目结构](#11-项目结构)
- [12. Checkpoint 与产物](#12-checkpoint-与产物)
- [13. 测试与实验规范](#13-测试与实验规范)
- [14. 论文报告建议](#14-论文报告建议)

---

## 1. 当前状态

### 当前推荐部署策略

当前推荐的可部署策略是 200000-episode high-augmented PPO 的 deterministic rollout：

```text
artifacts/runs/surrogate_ppo/
└── layerwise_topk_high_augmented_2026-08-20/
    └── best_policy.pth
```

SHA256：

```text
7c70b8ca1dd01341cd49b931c2c4c075deb46ec91da9dc071f79b5f35acf46c6
```

“推荐”表示该 checkpoint 已完成项目现有的真实 CodeLlama 验证、接口稳定且在线决策开销低；**不表示已经证明它在所有新 baseline 上全局最优**。

### 当前推荐 surrogate

```text
artifacts/models/
└── ppl_surrogate_general_assignment_high_augmented_ensemble.pth
```

SHA256：

```text
74a472278231db33560f6a57801a0af25a91d5d6bfa18a67bfe8fc203b0d84df
```

该 surrogate 的正式数据链对应：

```text
codellama/CodeLlama-7b-hf
```

因此不能用它选择 deployment 后，直接把结果写成严格匹配的 Llama-2-7B 实验。若最终论文使用 Llama-2-7B，需要重新采集 Llama-2-7B 标签、训练匹配 surrogate、重新冻结所有方法的 deployment，再进行真实模型评估。

### 当前代码入口

新实验只从下面这些 active scripts 开始：

```text
scripts/ppo/train_layerwise_topk.py              # 当前主 PPO
scripts/ppo/train.py                             # 真实 CodeLlama PPO 对照
scripts/ppo/validate_true_policy.py              # 冻结 candidate 的真实验证
scripts/baselines/compare_system_baselines.py    # 系统 baseline screening
scripts/baselines/evaluate_frozen_system_baselines_true.py
scripts/rl/compare_algorithms.py                 # PPO/A2C/DQN 消融
scripts/surrogate/collect_general_assignment.py  # 采集 surrogate 标签
scripts/surrogate/train_general_assignment.py   # 训练 general-assignment surrogate
```

旧的 segment PPO、旧 surrogate PPO 和旧 baseline comparison 已移到 `legacy/experiments_2026_09/`。`legacy/` 只用于历史复现，不属于当前实验入口。
### 当前主实验方法

论文系统主表优先使用：

```text
PPO deterministic
EdgeShard-UAV
HexGen-inspired
Simulated annealing
Petals-balanced
Neurosurgeon-inspired
Random feasible
```

JointDNN-MUAV 和 PipeEdge-UAV 已保留，但更适合作为历史/附录对照。RL 算法消融单独比较 PPO、A2C 和 Masked Double-DQN。

---

## 2. 问题定义

当前场景包含：

- 32 个 Transformer layers；
- 5 台异构 UAV；
- 每台 UAV 不同的计算速度、内存容量和能量预算；
- 每个 channel realization 是一个 `5 × 5` UAV 链路增益矩阵；
- 相邻层部署到不同 UAV 时产生一次跨 UAV boundary。

一个 action/deployment 是长度为 32 的整数向量：

```text
deployment[layer] = uav_id
```

例如：

```text
[0, 0, 0, 1, 1, 3, 3, ...]
```

表示前 3 层在 UAV 0，接下来 2 层在 UAV 1，随后若干层在 UAV 3。只要相邻元素不同，就产生跨 UAV activation transmission。

### 资源约束

`src/uav_rl/resource_assignment.py` 统一检查：

- layer memory 占用；
- UAV memory capacity；
- layer computation energy；
- boundary communication energy；
- UAV hover energy；
- UAV total energy budget。

当前默认 profile 是可复现的研究场景参数，不是特定 UAV 硬件的实测规格。所有正式 dataset、checkpoint 和 run manifest 都应保存完整 resource config，后续可以替换为真实 profiling 数据。

### Boundary 和 drop vector

32 层之间有 31 个可能 boundary。对于 boundary `b`：

- 如果 `deployment[b] == deployment[b+1]`，drop probability 为 0；
- 如果两层属于不同 UAV，则根据发送 UAV、接收 UAV 和当前 channel 计算 packet drop probability。

最终得到 31 维：

```text
drop_vector = [p_0, p_1, ..., p_30]
```

surrogate 预测的是该 drop vector 对模型质量的影响，而不是直接预测 latency。

---

## 3. PPL 和 reward 的计算

### Token-level PPL

真实 PPL evaluator 使用标准 causal language-model next-token loss：

1. 对每个 token 位置用前序 token 预测下一个 token；
2. 对非 padding token 的 negative log-likelihood 求和；
3. 除以参与评估的 token 总数；
4. 最后取指数。

形式为：

```text
NLL = sum(token_negative_log_likelihood) / valid_token_count
PPL = exp(NLL)
```

这里必须先对所有有效 token 聚合 NLL，再取一次指数，不能直接平均每个 batch 或每个文本的 PPL，否则不同序列长度会产生偏差。

### Clean PPL 和质量损失目标

这里的 `PPL_clean` 是模型在**没有跨 UAV activation noise 或 packet-drop 干扰**时的基准困惑度。它使用同一个语言模型、同一批文本、同一套 tokenizer、相同的有效 token mask 和相同的 PPL 计算方式，但关闭 deployment 引入的 activation corruption。换句话说，`PPL_clean` 表示模型在理想无噪声条件下本来能达到的质量。

`PPL_noisy` 则是在给定 deployment 和 channel 后，根据 boundary drop vector 注入 activation noise，再对同一批文本计算得到的困惑度。两者的区别只应来自跨 UAV 传输造成的扰动，而不是数据、模型或 token 数量不同。

项目不直接回归绝对 PPL，而是使用相对质量退化：

```text
log_ppl_ratio = log(PPL_noisy / PPL_clean)
```

例如：

```text
PPL_clean = 10
PPL_noisy = 12
log_ppl_ratio = log(12 / 10) ≈ 0.182
```

因此，`log_ppl_ratio = 0` 表示没有额外质量损失；它为正且越大，表示 activation noise 造成的退化越严重。使用这个比值还有两个好处：不同文本或模型的 clean PPL 尺度被消除了，而且对 surrogate 来说通常比直接拟合绝对 PPL 更稳定。
### Latency

总时延由解析模型直接计算：

```text
total_latency = computation_latency + communication_latency
```

计算时延由逐层计算成本和对应 UAV speed 得到。通信时延由 boundary activation size、链路 spectral efficiency 和共享带宽分配得到。

Latency 不由 surrogate 预测，因为它可以从 deployment、channel 和系统参数直接计算；将解析量交给 surrogate 会增加不必要误差。

### Reward

统一环境位于：

```text
src/uav_rl/resource_environment.py
```

Reward 定义：

```text
reward = -(
    quality_weight * log(PPL_noisy / PPL_clean)
    + (1 - quality_weight) * total_latency / latency_reference
)
```

当前 `quality_weight = 0.5`。Reward 是负综合代价，因此：

```text
reward 越接近 0 越好
log-PPL ratio 越低越好
PPL 越低越好
latency 越低越好
```

非法 deployment 的 reward 为 `-100`。

---

## 4. 方法与 baseline

本项目把方法分成两组。**主实验**比较不同的模型部署策略；**RL 消融**只比较几种简单 RL 算法。所有方法最后都输出同一种 32 层 deployment，并使用同一个 resource checker、latency model 和 quality evaluator，因此可以直接比较。

### 4.1 PPO deterministic（本文在线策略）

PPO 是本文的在线部署方法。它按照 layer 顺序逐层决定下一层放在哪台 UAV，决策时会同时看到当前信道、各 UAV 的剩余资源和上一层所在的 UAV。训练阶段使用 surrogate reward，部署阶段使用 deterministic argmax，因此在线只需要一次 policy forward，不需要针对每个新信道重新运行搜索。

代码位于：

```text
src/uav_rl/rl/
scripts/ppo/train_layerwise_topk.py
```

PPO 的重点不是证明它是最强的 RL 算法，而是学习一个可以快速响应信道变化的 deployment policy。`max_policy_boundaries = 4` 是 **boundary freeze threshold**：达到阈值后，只要当前 UAV 仍然可行就继续使用它；如果当前 UAV 已经无法继续，策略仍然可以切换到其他 UAV。因此它不是绝对的 boundary 数量上限。

### 4.2 PPO Top-K candidate reranking（候选集消融）

PPO 目前也支持 Top-K candidate 模式，但它不是另一种训练算法，而是同一个 PPO policy 的候选生成与重排实验。对每个新 channel，policy 采样多个可行 deployment，用 surrogate reward 排序，并保留前 `K` 个候选；实际可部署的版本使用 surrogate 选择的 Top-1。只有在离线分析时，才可以用真实 LLM PPL 在这 `K` 个候选中选择最优者，这个 **true Top-K oracle** 不能作为在线方法或公平主表结果。

入口仍然是：

```text
scripts/ppo/train_layerwise_topk.py
```

关键参数是 `--top-k` 和 `--candidate-samples`。训练完成后，结果写入 run directory 下的：

```text
topk_true_validation.json
candidate_policies/
```

因此 README 中主表显示 `PPO deterministic` 是有意的：它代表一次 deterministic policy forward 的实际在线策略。Top-K 应放在 PPO 的候选重排/推理消融中，用来回答“增加候选数能否改善质量，以及 surrogate 排序损失了多少”，不能与 EdgeShard-UAV 等一次性部署 baseline 直接混写成同一种运行模式。

### 4.3 EdgeShard-UAV：连续分片方法

EdgeShard-UAV 是 EdgeShard 设备选择和连续分片思想在 UAV 场景下的适配版本。它把 32 层切成若干连续 block，然后选择 UAV 的使用顺序。它主要优化解析的计算和通信时延，不使用 PPL surrogate，因此代表“只根据系统状态做快速分片”的方法。

实现文件：

```text
src/uav_rl/system_baselines/edge_shard_uav.py
```

代码用动态规划枚举可行的 UAV 顺序和 block 边界，并检查内存、计算能耗和通信能耗。由于我们没有复现原 EdgeShard 的完整运行时，所以论文中应称为 **EdgeShard-UAV adaptation**，而不是原方法的完整复现。

### 4.4 LinguaLinked-UAV：能力感知的移动设备流水线

LinguaLinked-UAV 借鉴移动设备协同 LLM 推理中的“按设备能力分配模型、再根据通信条件组织流水线”的思路。代码先根据 UAV 的计算速度、内存和剩余能量估计每台 UAV 应承担的 layer 数，再枚举可行的 UAV 顺序和连续 block，最后选择通信时延最低的方案。它不使用 PPL surrogate，因此可以作为不依赖学习质量模型的移动设备协同 baseline。

实现文件：

```text
src/uav_rl/system_baselines/lingualinked_uav.py
```

这里的实现是适配当前 layer-assignment action space 的 **LinguaLinked-UAV adaptation**，并没有复现原方法的完整移动设备 runtime、并行请求调度或系统吞吐量实验。

### 4.5 HexGen-inspired：拓扑感知搜索

HexGen-inspired 使用一个简单的进化搜索来寻找 UAV 顺序和连续 layer boundaries。它先用 EdgeShard-UAV 和随机方案生成初始解，然后不断调整边界、交换 UAV 顺序或替换 UAV，并保留 reward 较高的方案。

实现文件：

```text
src/uav_rl/system_baselines/hexgen_search.py
```

它会读取冻结的 surrogate reward，所以比 EdgeShard-UAV 使用了更多质量信息，也会花费更长的搜索时间。这个实现借鉴的是 HexGen 的异构拓扑搜索思想，没有复现其 tensor parallelism、serving runtime 和请求调度，因此统一称为 **HexGen-inspired**。

### 4.6 Simulated annealing：通用搜索方法

模拟退火从一个可行 deployment 开始，随机修改某一层或一段连续层。如果新方案更好就接受；即使暂时变差，也可能以一定概率接受，从而跳出局部最优。它直接使用 surrogate reward，是质量导向较强的搜索 baseline，但每个 channel 都要单独搜索，决策时间明显高于 PPO。

### 4.7 Petals-balanced：流水线均衡方法

Petals-balanced 把模型切成连续 block，并尽量让各个 pipeline stage 的计算负载均衡。它适合检验一种常见的 LLM 分布式推理思路：如果只追求 pipeline 平衡，而不针对 UAV 信道和质量损失做优化，最终的综合 reward 是否会下降。

### 4.8 Neurosurgeon-inspired：经典双设备切分

Neurosurgeon-inspired 只考虑两台 UAV 和一个切分点。它枚举不同切分位置和 UAV 组合，再用解析的质量—时延 proxy 选出最优方案。这个 baseline 简单、容易解释，但表达能力明显低于允许任意 layer assignment 的 PPO。

### 4.9 JointDNN-MUAV：数学优化对照

JointDNN-MUAV 用 MILP 同时表示 layer placement 和跨 UAV boundary，并加入内存、能量和通信约束。它代表较早的显式数学优化思路。由于 JointDNN 较早，本项目保留它作为历史和附录对照，不再把它作为唯一的核心 baseline。

### 4.10 PipeEdge-UAV：纯时延对照

PipeEdge-UAV 使用动态规划选择 UAV 顺序和连续 block，目标只有计算时延加通信时延。它可以说明一个重要 trade-off：时延最低的 deployment 不一定具有最低的 PPL，也不一定具有最好的综合 reward。

### 4.11 Random feasible：下界

Random feasible 只随机生成满足资源约束的 deployment，不使用信道质量、surrogate 或 latency 目标。它不是竞争方法，而是 sanity check，用来确认经过优化的 deployment 确实比随机可行方案更好。

### 4.12 方法对比关系

| 方法 | 主要思想 | 是否使用 surrogate | 主要优点 | 主要局限 |
| --- | --- | --- | --- | --- |
| **PPO** | 学习在线 layer placement policy | 训练时使用 | 在线决策快，适应动态信道 | 需要训练，性能依赖泛化 |
| **LinguaLinked-UAV** | 按设备能力分配 block，再按信道排序 | 否 | 贴近移动设备协同推理 | 质量目标没有直接进入选择 |
| **EdgeShard-UAV** | DP 选择连续分片 | 否 | 解释简单，时延开销较低 | 只能使用连续且不重复的 UAV block |
| **HexGen-inspired** | 拓扑感知进化搜索 | 是 | 搜索能力强，能直接优化 reward | 每个 channel 都需要搜索，较慢 |
| **Simulated annealing** | 随机邻域搜索 | 是 | 容易跳出局部最优 | 在线成本很高 |
| **Petals-balanced** | 平衡 pipeline stage | 否 | 代表 LLM pipeline 思路 | 不直接优化质量损失 |
| **Neurosurgeon-inspired** | 双 UAV 单切分 | 否 | 简洁、可解释 | 搜索空间过于受限 |
| **JointDNN-MUAV** | MILP 显式优化 | 否 | 约束表达清楚 | 优化较早且逐 channel 求解较慢 |
| **PipeEdge-UAV** | DP 最小化 latency | 否 | 能体现时延下界 | 忽略质量目标 |
| **Random feasible** | 随机可行部署 | 否 | 提供性能下界 | 不进行优化 |

### 4.13 RL 算法消融

RL 消融只比较：

```text
PPO、A2C、Masked Double-DQN
```

三者使用相同的状态、动作空间、资源 mask、训练预算、channel seeds、surrogate reward 和 held-out channels。生产环境中使用的 PPO checkpoint 不放进从零训练的公平 RL 对照表中。

RL baseline 的详细协议见：

```text
docs/RL_BASELINE_PROTOCOL.md
```

---

## 5. 2026-09-02 系统方法 benchmark（含 PPO Top-K）

### 5.1 实验协议

结果目录：

```text
artifacts/runs/system_baseline_comparison/
└── with_ppo_topk_64ch_2026-09-02/
    ├── channels.npy
    ├── frozen_deployments.npz
    ├── comparison_summary.json
    ├── comparison_table.csv
    ├── comparison.png
    └── EXPERIMENT_REPORT.md
```

协议：

```text
channels                     = 64
channel seed                 = 20260910
PPO deterministic             = one policy forward
PPO surrogate Top-1           = Top-K=5, candidate-samples=20
PPO true Top-5 oracle         = evaluated later from frozen candidates
EdgeShard plans/state       = 8
LinguaLinked-UAV            = capability-balanced contiguous pipeline
HexGen-inspired population  = 48
HexGen-inspired generations = 48
JointDNN time limit         = 1.0 second/channel
simulated annealing steps   = 1024/channel
random seed                 = 20260911
quality evaluator            = frozen high-augmented surrogate
GPU                          = NVIDIA GeForce RTX 5070 Ti
```

所有方法使用相同 channel、resource config、latency model 和最终 reward evaluator。表中 CI 是每个方法相对 PPO 的 paired reward difference 的 95% normal interval。

决策时间是在同一台本地机器上的单次 wall-clock benchmark，包含 Python selector 开销；它适合比较当前实现的相对在线成本，不应当被解释为跨硬件的绝对系统吞吐量。

### 5.2 结果

| 方法 | Reward ↑ | 相对 PPO 差值 `[95% CI]` | log-PPL ratio ↓ | Latency (s) ↓ | Boundaries | 决策 ms/channel ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PPO deterministic** | -0.383815 | 0 | 0.207401 | 0.735128 | 2.062 | **8.26** |
| **PPO surrogate-selected Top-1** | **-0.373300** | +0.010515 `[+0.001859, +0.019172]` | **0.197015** | 0.721161 | 2.016 | 183.76 |
| **HexGen-inspired** | -0.372926 | +0.010890 `[-0.004879, 0.026658]` | 0.204905 | 0.709825 | 2.000 | 794.66 |
| LinguaLinked-UAV | -0.544335 | -0.160519 `[-0.175725, -0.145314]` | 0.313883 | 1.016668 | 3.000 | 712.56 |
| Simulated annealing | -0.373881 | +0.009935 `[-0.005966, 0.025835]` | **0.203512** | 0.714160 | 2.000 | 2973.18 |
| Neurosurgeon-inspired | -0.388179 | -0.004363 `[-0.020974, 0.012247]` | 0.238366 | 0.705948 | 2.000 | 209.05 |
| JointDNN-MUAV | -0.398647 | -0.014832 `[-0.032614, 0.002951]` | 0.261025 | 0.703687 | 2.000 | 1827.43 |
| EdgeShard-UAV | -0.424044 | -0.040229 `[-0.057753, -0.022705]` | 0.315048 | **0.699451** | 2.000 | 265.96 |
| PipeEdge-UAV | -0.431887 | -0.048072 `[-0.065950, -0.030194]` | 0.330299 | 0.700022 | 2.000 | 112.74 |
| Petals-balanced | -0.802524 | -0.418709 `[-0.444753, -0.392664]` | 0.531086 | 1.409244 | 4.000 | 112.68 |
| Random feasible | -3.477950 | -3.094135 `[-3.634770, -2.553500]` | 2.838805 | 5.402419 | 8.469 | 8.90 |

所有方法：

```text
invalid_fraction = 0
```

这一节的表格是同一批 64 个 channel 上的 **surrogate screening**。`PPO surrogate-selected Top-1` 已经纳入冻结 deployment；`PPO Top-5 true oracle` 不放进这个 surrogate reward 表，而是由下一步 true-LLM evaluator 在 `ppo_topk_candidates.npz` 中逐候选评估后加入真实结果。

需要特别注意模型链：当前冻结 surrogate 是基于 CodeLlama-7B 数据训练的，而本次真实复验如果使用本地 Llama-2-7B，只能作为跨模型的探索性复验，不能作为严格的同模型最终证据。要形成论文主结果，应重新用 Llama-2-7B 采集 `PPL_clean/PPL_noisy` 标签、训练 Llama-2-7B surrogate，并重新冻结所有方法。

### 5.3 结果解释

1. **PPO surrogate-selected Top-1 在这组 64-channel surrogate screening 中优于 deterministic PPO。** paired difference 为 `+0.010515`，95% CI 为 `[+0.001859, +0.019172]`。但它的在线 selector 需要生成并评估 20 个候选，决策时间约为 deterministic PPO 的 `22.2×`，因此不能直接替代低延迟主策略。
2. **HexGen-inspired 的 surrogate mean 与 PPO Top-1 接近。** 它相对 deterministic PPO 的差值为 `+0.010890`，95% CI 跨过 0，不能宣称显著优于 PPO。
3. **Simulated annealing 同样略高于 deterministic PPO，但 CI 跨 0。** 它是另一个有竞争力的 surrogate-assisted offline search。
4. **PPO deterministic 的在线决策仍然最快的学习型策略。** Top-1、HexGen-inspired、EdgeShard-UAV、JointDNN 和 simulated annealing 分别约为 deterministic PPO 决策开销的 `22.2×`、`96.2×`、`28.1×`、`235.1×` 和 `392.5×`。
5. **EdgeShard-UAV 取得最低 latency，但质量 proxy 较差。** 它比 PPO 平均低约 `0.0357 s`，同时 log-PPL ratio 从 `0.2074` 增加到 `0.3150`，所以综合 reward 更低。这说明纯 latency-oriented contiguous sharding 不能替代质量感知优化。
6. **Petals-balanced 不是强 reward optimizer。** 在当前动态 UAV 信道中，它固定形成更多 pipeline boundaries，通信和质量成本较高。
7. **Random feasible 验证优化的必要性。** 虽然全部可行，但 reward、quality 和 latency 均明显更差。

最准确的当前结论是：

> PPO 在冻结 surrogate 上提供了与强搜索方法有竞争力的质量—时延折中，并将在线决策开销降低一个到两个以上数量级；最终真实质量排名仍需对冻结 deployment 运行匹配 true LLM。

不能写成：

> PPO 显著超过所有 baseline。

---

## 6. Grouped exact oracle

完整 `5^32` assignment 无法直接穷举。为估计小规模搜索上界，项目实现：

```text
src/uav_rl/system_baselines/exact_grouped_oracle.py
scripts/baselines/run_exact_grouped_oracle.py
```

它将 32 层分为 `G` 个确定性的近等长 super-layers，枚举全部 `5^G` group assignments，再展开为标准 32-layer deployment，并使用完整资源约束和统一 reward evaluator 评分。

### 2026-08-31 结果

目录：

```text
artifacts/runs/exact_grouped_oracle/
└── grouped_8_2026-08-31/
    ├── channels.npy
    ├── oracle_deployments.npy
    └── summary.json
```

协议与结果：

```text
groups                  = 8
channels                = 8
channel seed            = 20260920
assignments/channel     = 5^8 = 390625
feasible/channel        = 194880
mean best reward        = -0.356989
quality evaluator       = frozen surrogate
```

各 channel 的 grouped optimum reward：

```text
-0.355101, -0.354153, -0.345522, -0.360383,
-0.359067, -0.352931, -0.355202, -0.373552
```

该结果只能称为：

```text
exact optimum inside the 8-group surrogate action set
```

它不是完整 32-layer 全局 oracle，也不是真实 LLM oracle。由于这里使用了另一组 channel seed，不能直接用 mean 与第 5 节 64-channel 表计算 paired optimality gap。

---

## 7. 已有真实 CodeLlama 证据

已有 common-seed 验证使用：

```text
model                  = codellama/CodeLlama-7b-hf
channels               = 32
activation-noise seeds = 4
```

关键结果：

| 方法 | 真实 reward ↑ | 说明 |
| --- | ---: | --- |
| PPO deterministic，200000 ep. | **-0.384838** | 当前推荐可部署策略 |
| PPO surrogate-selected Top-1，200000 ep. | -0.389706 | surrogate 从候选中选择 |
| PPO best observed Top-1，10000 ep. | -0.387903 | 历史快照 |
| PPO Top-5 true oracle，200000 ep. | -0.381480 | 候选集真实上界，不可部署 |
| CoEdge-inspired layer greedy | -0.390294 | 历史启发式对照 |

解释：

- 200000-episode PPO 的 deterministic 模式优于同 checkpoint 的 surrogate-selected Top-1；
- Top-5 true oracle 又优于实际 Top-1，说明候选生成有潜力，但 surrogate ranking 仍有误差；
- 这些真实结果没有覆盖 2026-08-31 新加入的 EdgeShard-UAV 和 HexGen-inspired；
- 新系统 baseline 必须读取已经冻结的 `frozen_deployments.npz` 做真实模型评估，不能根据真实 PPL 重新搜索。

---

## 8. Surrogate 数据与训练链路

### 8.1 真实标签生成

入口：

```text
scripts/surrogate/collect_general_assignment.py
```

底层：

```text
src/uav_rl/data/general_assignment_dataset.py
```

每个 deployment：

1. 根据 channel 转换为 31 维 boundary drop vector；
2. 使用多个 activation-noise seeds 运行真实 CodeLlama；
3. 计算每个 seed 的 `log(PPL_noisy / PPL_clean)`；
4. 聚合为训练标签；
5. 逐条写入 JSONL cache，支持中断恢复。

当前数据配置包括：

```text
model       = codellama/CodeLlama-7b-hf
dataset     = WikiText-2 raw test split
max texts   = 50
max length  = 512
evaluated   = 27 sequences / 1689 tokens（当前 manifest）
```

### 8.2 为什么输入是 drop vector

LLM 质量退化发生在跨 UAV activation transmission。两个不同 deployment 只要产生相同 boundary drop vector，对 activation-noise evaluator 来说就具有相同质量扰动条件。

因此 surrogate 输入 drop vector，而不是 UAV ID 序列，可以减少对设备编号的无意义记忆，并使质量模型与 latency/resource 模型解耦。

### 8.3 Ensemble 训练

入口：

```text
scripts/surrogate/train_general_assignment.py
```

当前推荐模型为 high-boundary augmented ensemble。训练和验证关注：

- MAE、RMSE、R²；
- Spearman ranking correlation；
- p50/p90/p95/max error；
- high-boundary / tail 区域误差；
- uncertainty 与真实误差相关性；
- grouped reward regret。

Surrogate 只用于：

- PPO 训练；
- 候选排序；
- inexpensive screening；
- surrogate-assisted search。

最终论文质量证据必须来自真实模型。

---

## 9. PPO 与 RL 消融

### PPO 训练入口

```text
scripts/ppo/train_layerwise_topk.py
```

训练过程：

1. 生成 channel；
2. 逐层 rollout；
3. 使用 resource action mask 排除当前不可行动作；
4. 完成 deployment 后由 surrogate + latency 计算 reward；
5. 使用 GAE 和 clipped PPO 更新 policy/value network；
6. 周期性保存 candidate checkpoints 和完整 training state；
7. 最终冻结 policy，进行 deterministic 或 Top-K 验证。

训练支持 teacher warm start，但 teacher 只提供 behavior-cloning 初始化，之后仍由 PPO 优化；受控 RL 算法表必须全部从零训练，不能把 warm-start production PPO 混入公平对照。

### 简单 RL 对照

实现：

```text
src/uav_rl/rl/algorithms/
├── a2c.py
└── dqn.py
```

入口：

```text
scripts/rl/compare_algorithms.py
scripts/rl/evaluate_algorithms_true.py
scripts/rl/plot_algorithm_comparison.py
```

已有 1024-episode、三 seed pilot 的 surrogate mean：

| 算法 | Reward mean | seed std |
| --- | ---: | ---: |
| PPO | -0.5278 | 0.0593 |
| DQN | -0.8640 | 0.3301 |
| A2C | -1.1795 | 0.5484 |

该结果只是 pilot。DQN 在 1024 episode 时 epsilon 仍约为 0.93，不能作为最终论文 RL 排名。

---

## 10. 实验复现

所有命令从项目根目录执行：

```powershell
cd E:\Projects\2026.8\uav-rl
$env:PYTHONPATH = "src"
```

### 10.1 复现 64-channel 系统 benchmark

```powershell
python scripts/baselines/compare_system_baselines.py `
  --channels 64 `
  --channel-seed 20260910 `
  --edge-shard-plans-per-state 8 `
  --hexgen-population 48 `
  --hexgen-generations 48 `
  --jointdnn-time-limit 1.0 `
  --annealing-steps 1024 `
  --random-seed 20260911 `
  --surrogate-device cuda `
  --top-k 5 `
  --candidate-samples 20 `
  --output-dir artifacts/runs/system_baseline_comparison/with_ppo_topk_64ch_2026-09-02
```

输出会冻结 channel 和每个方法的 deployment。

### 10.2 对冻结 deployment 运行真实 LLM

```powershell
python scripts/baselines/evaluate_frozen_system_baselines_true.py `
  --comparison-dir artifacts/runs/system_baseline_comparison/with_ppo_topk_64ch_2026-09-02 `
  --model-id "<matching local CodeLlama model directory>" `
  --noise-samples 4 `
  --device cuda
```

`model-id` 必须与 surrogate 数据链匹配。脚本只读取冻结 deployment，不会重新调用 selector。

### 10.3 运行 8-group exact oracle

```powershell
python scripts/baselines/run_exact_grouped_oracle.py `
  --channels 8 `
  --channel-seed 20260920 `
  --groups 8 `
  --batch-size 8192 `
  --max-assignments 1000000 `
  --device cuda `
  --output-dir artifacts/runs/exact_grouped_oracle/grouped_8_2026-08-31
```

### 10.4 收集 surrogate 标签

```powershell
python scripts/surrogate/collect_general_assignment.py --help
```

真实 PPL 任务会按 `(drop_vector, noise_seed)` 写入 JSONL cache，中断后可以恢复。

### 10.5 训练 surrogate

```powershell
python scripts/surrogate/train_general_assignment.py --help
```

### 10.6 训练 PPO

```powershell
python scripts/ppo/train_layerwise_topk.py --help
```

### 10.7 RL 算法对比

```powershell
python scripts/rl/compare_algorithms.py --help
```

---

## 11. 项目结构

```text
uav-rl/
├── README.md
├── Robust_Inference2025-main.pdf
├── pyproject.toml
├── src/uav_rl/
│   ├── config.py
│   ├── resource_assignment.py
│   ├── resource_environment.py
│   ├── resource_baselines.py
│   ├── quality.py
│   ├── true_quality.py
│   ├── surrogate.py
│   ├── surrogate_training.py
│   ├── data/
│   ├── metrics/
│   ├── models/
│   ├── rl/
│   │   └── algorithms/
│   └── system_baselines/
│       ├── edge_shard_uav.py
│       ├── hexgen_search.py
│       └── exact_grouped_oracle.py
├── scripts/
│   ├── baselines/
│   │   ├── compare_system_baselines.py
│   │   ├── evaluate_frozen_system_baselines_true.py
│   │   └── run_exact_grouped_oracle.py
│   ├── ppo/
│   ├── rl/
│   ├── surrogate/
│   ├── benchmarks/
│   └── maintenance/
├── tests/
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── RECENT_LLM_BASELINES.md
│   └── RL_BASELINE_PROTOCOL.md
├── artifacts/
│   ├── data/
│   ├── models/
│   └── runs/
└── legacy/                 # read-only superseded code and manifests
```

设计原则：

- 当前实现放在 `src/uav_rl/`；
- 命令入口放在 `scripts/`；
- 生成数据、checkpoint 和结果放在 `artifacts/`；
- 已结束历史代码放在 `legacy/`；
- 新方法使用清晰功能名，不使用 `v2/v3/final_new` 等文件名；
- 每次 run 用独立目录和 manifest 保存超参数、seed、hash 和输出。

更完整结构说明：

```text
docs/PROJECT_STRUCTURE.md
```

---

## 12. Checkpoint 与产物

### 推荐模型

| 类型 | 路径 | SHA256 |
| --- | --- | --- |
| High-augmented surrogate ensemble | `artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth` | `74a472278231db33560f6a57801a0af25a91d5d6bfa18a67bfe8fc203b0d84df` |
| 200000-episode PPO | `artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth` | `7c70b8ca1dd01341cd49b931c2c4c075deb46ec91da9dc071f79b5f35acf46c6` |

### 关键结果目录

```text
artifacts/runs/surrogate_ppo/
├── layerwise_topk_high_augmented_2026-08-20/
└── common_seed_baseline_comparison.json

artifacts/runs/system_baseline_comparison/
├── authoritative_heuristics_64ch_2026-08-30/
└── with_ppo_topk_64ch_2026-09-02/

artifacts/runs/exact_grouped_oracle/
└── grouped_8_2026-08-31/
```

`artifacts/` 默认不进入普通 Git 跟踪。需要发布正式实验结果时，应明确选择必要的小型 summary、CSV、图和 report，避免提交大模型权重或大规模 cache。

---

## 13. 测试与实验规范

### 代码检查

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q -p no:cacheprovider --basetemp .pytest_tmp_full
python -m ruff check src scripts tests
git diff --check
```

2026-08-31 当前结果：

```text
88 passed
Ruff: All checks passed
```

### 公平性要求

1. 所有方法使用相同 held-out channels；
2. 真实评估使用相同 noise seeds；
3. 使用相同 resource config 和 latency reference；
4. 分开报告 surrogate screening 和 true-LLM evidence；
5. 保存并冻结每个 selector 产生的 deployment；
6. 禁止观察真实 PPL 后重新选择候选；
7. 报告 reward、质量、latency、决策时间、invalid fraction 和 boundary count；
8. 报告 paired confidence interval，而不只报告均值；
9. 对 adaptation/inspired 方法保留准确限定词；
10. surrogate-assisted 方法必须标明其使用了额外的质量模型信息。

### 模型匹配要求

严格实验链必须满足：

```text
true-label model
= surrogate training model
= frozen-deployment true evaluator model
```

当前正式 surrogate 对应 CodeLlama-7B。若切换到 Llama-2-7B，需要重建整个匹配链，不能只替换最后的 evaluator。

---

## 14. 论文报告建议

### 主表

建议保留：

```text
PPO deterministic
EdgeShard-UAV
HexGen-inspired
Simulated annealing
Petals-balanced
Neurosurgeon-inspired
Random feasible
```

主指标：

```text
reward
true PPL / log-PPL ratio
end-to-end latency
online decision time
invalid fraction
boundary count
paired 95% CI
```

### 附录表

```text
JointDNN-MUAV
PipeEdge-UAV
CoEdge-inspired
其他历史 proxy/beam/search baseline
```

### RL 消融表

```text
PPO
A2C
Masked Double-DQN
```

### 当前可以写的结论

> 在冻结 surrogate 的 64-channel benchmark 中，PPO 与 HexGen-inspired 和 simulated annealing 的综合 reward 没有形成显著统计分离，但 PPO 的在线决策开销分别低约 74 倍和 278 倍。纯 latency-oriented EdgeShard-UAV 和 PipeEdge-UAV 获得更低时延，却因预测质量损失较高而降低综合 reward，说明 UAV 协同 LLM 分割需要联合建模质量与时延。

### 当前不能写的结论

```text
PPO 显著优于所有 baseline。
PPO 是最强 RL 算法。
新增方法的 surrogate 排名等价于真实 LLM 排名。
CodeLlama surrogate 结果可以直接当作 Llama-2-7B 结果。
8-group exact oracle 是完整 32-layer 全局最优。
```

最终论文主表应在匹配模型权重可用后，对 2026-08-31 已冻结的 deployment 完成 common-seed true-LLM evaluation。
