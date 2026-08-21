# Common-seed baseline comparison

本轮加入五个更强的 baseline，并在完全相同的 32 个 channel、4 个 noise seed 和真实 CodeLlama-7b evaluator 上重新运行全部有效方法。

## 评估协议

- channel seed：`20260824`
- noise seeds：`4100000`–`4100003`
- 模型：`codellama/CodeLlama-7b-hf`
- 27 个序列，1689 个 evaluated tokens
- clean PPL：`12.875911`
- PPO Top-K：5 个候选，每个 channel 20 次 surrogate sampling
- beam：128 和 512
- GA：population 32，16 generations
- simulated annealing：1024 steps
- MILP：SciPy/HiGHS，1 秒/channel
- 最终 reward、PPL 和 latency 全部由真实 CodeLlama 评估；surrogate 只用于搜索阶段

## 结果

reward 是负的综合代价，因此越接近 0 越好；PPL 和 latency 越小越好。

| 方法 | 真实 reward ↑ | PPL ↓ | 平均 latency (s) ↓ | 相对 PPO Top-1 |
| --- | ---: | ---: | ---: | ---: |
| **CoEdge-style adaptive partition** | **-0.390294** | 16.228 | 0.722 | **+2.61%** |
| PPO Top-5 true oracle（不可部署上界） | -0.391821 | 16.407 | 0.714 | +2.23% |
| **PPO Top-1 surrogate-selected** | **-0.400744** | 16.735 | 0.714 | — |
| Surrogate simulated annealing | -0.406119 | 16.986 | 0.708 | -1.34% |
| Surrogate local search | -0.411060 | 17.245 | 0.704 | -2.57% |
| Proxy beam 128 | -0.419458 | 17.567 | 0.701 | -4.67% |
| Neurosurgeon-style best split | -0.419458 | 17.567 | 0.701 | -4.67% |
| Constrained genetic surrogate | -0.417787 | 17.480 | 0.704 | -4.25% |
| Wide proxy beam 512 | -0.421463 | 17.658 | 0.700 | -5.17% |
| MILP proxy oracle | -0.421463 | 17.658 | 0.700 | -5.17% |
| PPO deterministic | -0.428221 | 17.728 | 0.722 | -6.86% |
| Fixed-eight Strong-link | -0.475804 | 15.310 | 1.027 | -18.73% |
| Dynamic programming | -0.480867 | 15.619 | 1.014 | -19.99% |
| Surrogate random search 1024 | -0.488966 | 16.232 | 0.991 | -22.01% |
| Random feasible | -3.383793 | 860.959 | 5.289 | -744.38% |

所有方法的 `invalid_fraction` 都是 `0.0`。

## 新 baseline 分析

### CoEdge-style adaptive partition

这是本轮最强的可部署方法，真实 reward `-0.390294`，比 PPO Top-1 高约 `2.61%`。它逐层比较当前 UAV 的计算代价、链路切换代价、memory/energy 余量和负载惩罚，动态决定是否切换 UAV。它不是 PPO，也没有使用真实模型搜索，因此这个结果说明当前 PPO policy 仍有改进空间。

需要注意：它超过了 `ppo_topk_true_oracle`，但这并不矛盾。Top-K true oracle 只在 PPO 生成的 5 个候选中选择，CoEdge 可以生成 PPO 候选集之外的 assignment；因此 Top-K oracle 不是全局上界。

### Surrogate simulated annealing

模拟退火 reward 为 `-0.406119`，优于 beam 512 加 local search（`-0.411060`）。它允许以递减概率接受暂时变差的邻居，因此能跳出只接受改进动作的局部搜索陷阱。

### Constrained genetic algorithm

GA reward 为 `-0.417787`。约束修复保证了 assignment 的 memory/energy 可行，但本轮有限预算（32 个个体、16 代）下没有超过 beam/local-search。它仍然是有价值的全局搜索对照，但不应把本轮结果解释为 GA 的充分性能上限。

### MILP proxy oracle

MILP reward 与 beam 512 完全相同（`-0.421463`）。当前 solver 只优化线性化的逐层计算、边界丢包和传输 latency proxy；真实共享带宽的非线性 latency 和 CodeLlama PPL 没有进入 MILP 目标。因此它是 proxy 参考，不是真实模型全局 oracle。

### Neurosurgeon-style best split

单切分 baseline reward 与 beam 128 相同（`-0.419458`）。它只枚举一处 boundary 和两台 UAV，不能表达多段、多 UAV 或任意层切换，主要用于文献结构对照。

## 当前结论

1. 本轮最强可部署 baseline 是 CoEdge-style adaptive partition，暂时超过 PPO Top-1 约 `2.61%`。
2. PPO Top-1 仍明显优于 beam、GA、DP、fixed-eight 和随机搜索，但不再是当前 common-seed 结果中 reward 最好的方法。
3. 模拟退火是当前最强的 surrogate 搜索型 baseline。
4. 单纯增加 beam width 或使用线性化 MILP proxy 没有改善真实 reward，进一步证明 additive proxy 与真实 PPL 目标存在错配。
5. CoEdge 的优势需要在更多 held-out channels 和 noise seeds 上复验后，才能作为下一轮 PPO 的主要行为克隆 teacher 或默认部署 baseline。

## 运行信息

- 总耗时：约 `520.1 s`
- CodeLlama forward：`300`
- cache hits：`2260`
- cached entries：`1940`

详细 JSON：[`common_seed_baseline_comparison.json`](common_seed_baseline_comparison.json)