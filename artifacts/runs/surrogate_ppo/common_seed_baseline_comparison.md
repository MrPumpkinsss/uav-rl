# Common-Seed General-Assignment Baseline Comparison

## 结论

在完全相同的 32 个 validation channels、4 个 activation-noise seeds 和真实 CodeLlama evaluator 上，当前 surrogate-selected Top-1 PPO 的 reward 为 **-0.400744**，是本轮所有可执行方法中最好的结果；只低于同一组 Top-5 候选的真实 oracle（-0.391821）。

新增的 **Dynamic programming** baseline reward 为 **-0.480867**。它已经不再是固定 4×8 层结构，而是在当前资源约束下搜索可变长度的连续区块；不过它仍使用一个加性“丢包 + 延迟” proxy 选动作，最终由真实 CodeLlama 评估。因此它比随机搜索更有结构，但没有超过 PPO，也没有超过最强的 proxy beam baseline。

最强的非-PPO baseline 仍然是 `proxy_beam_128`，真实 reward 为 **-0.419458**，比 Top-1 PPO 低约 4.67%。

## 严格统一的评估条件

- 模型：`codellama/CodeLlama-7b-hf`
- corpus：27 sequences，1689 evaluated tokens
- channels：32 个，`generate_resource_channels(32, 20260824)`
- noise seeds：`4100000, 4100001, 4100002, 4100003`
- 资源模型：32 layers、5 UAVs、每个 UAV 最多 8 layers，并启用 memory/energy 约束
- 所有方法使用同一真实 CodeLlama evaluator、同一 clean PPL 和同一 noise seed 集合
- 评估不向 surrogate 或 PPO 回写数据

## 结果

| 方法 | 动作空间/算法类别 | 真实 reward ↑ | PPL ↓ | 平均 latency (s) ↓ | 相对 Top-1 PPO |
|---|---|---:|---:|---:|---:|
| PPO Top-1（surrogate 选中） | 任意逐层 PPO + Top-K | **-0.400744** | 16.735 | 0.713 | — |
| PPO Top-5 真实 oracle | PPO 候选集真实模型上界（不可部署） | **-0.391821** | 16.407 | 0.714 | +2.23% |
| Proxy beam 128 | 任意逐层 assignment 的 proxy beam search | -0.419458 | 17.567 | **0.701** | -4.67% |
| PPO deterministic | 同一 PPO policy 的确定性 rollout | -0.428221 | 17.728 | 0.722 | -6.86% |
| Fixed-eight Strong-link | 固定 4 个区块、每块 8 层；链路丢包优先 | -0.475804 | **15.310** | 1.027 | -18.73% |
| Fixed-eight additive proxy | 固定 4×8 层；丢包 + 总延迟 proxy | -0.475507 | 15.431 | 1.015 | -18.66% |
| **Dynamic programming** | **可变长度连续区块；DP 加性 proxy** | **-0.480867** | 15.619 | 1.014 | -19.99% |
| Surrogate random search (1024) | 任意可行 assignment 的 surrogate 搜索 | -0.488966 | 16.232 | 0.991 | -22.01% |
| Fixed-eight compute-greedy | 固定 4×8 层；计算速度优先 | -0.641694 | 21.970 | 1.067 | -60.13% |
| Random feasible | 随机可行 assignment | -3.383793 | 860.959 | 5.289 | -744.38% |

这里“相对 Top-1 PPO”按 reward 的绝对值归一化计算；reward 越大越好，因此负值代表该方法不如 Top-1 PPO。

## Baseline 说明

### Dynamic programming

这是当前资源模型下新增的动态规划 baseline，不是旧的固定四段方法：

- 可以在任意 layer 之间放置 boundary；
- 每个连续区块长度可以不同，但不超过每台 UAV 的 8 层容量；
- 选择 4 或 5 台不同 UAV 组成区块顺序，并过滤 memory/energy 不可行方案；
- 用“边界丢包总量 + 归一化总延迟”的加性 proxy 做 Bellman 搜索；
- 最终 reward、PPL 和共享带宽 latency 仍由真实 CodeLlama/resource evaluator 计算。

它的真实 reward 为 -0.480867，略低于 fixed-eight Strong-link。这个结果说明：放宽固定 4×8 层边界本身并不能保证更好的真实 reward；当前 DP 优化的是可解释的加性 proxy，而真实 CodeLlama 的 PPL 影响是非线性的，proxy 与真实目标之间仍有排序误差。DP 代码位于 [`src/uav_rl/resource_baselines.py`](../../../../src/uav_rl/resource_baselines.py)，入口注册在 [`compare_general_assignment_baselines.py`](../../../../scripts/ppo/compare_general_assignment_baselines.py)。

### Fixed-eight Strong-link

枚举在当前资源约束下可行的四段、每段八层部署，只按相邻 UAV link 的 drop proxy 选择最优路径。它的边界位置和区块数量都是固定的，因此属于历史弱基线，不代表当前 PPO 的动作空间。

### Proxy beam 128

这是本轮最强的非-PPO baseline。它在任意 layer-to-UAV assignment 空间上做宽度为 128 的 beam search，逐层考虑资源可行性，并用 additive drop + latency proxy 排序。它允许一般 assignment 的资源状态，但最终真实质量仍由 CodeLlama 测量。

### Surrogate random search 1024

每个 channel 采样 1024 个任意可行 assignment，用 surrogate reward 选一个部署，作为“直接用 surrogate 做搜索”的参考。它没有超过 PPO，说明当前 PPO 学到的 channel-to-assignment 映射比简单独立采样更有效。

### PPO Top-5 true oracle

只在 PPO 生成的 5 个候选中使用真实模型选择最优项，不是可部署算法，而是衡量候选集上限。Surrogate-selected Top-1 与该上限的差距为 0.008923 reward。

## 解读

1. `proxy_beam_128` 仍是最强非 PPO baseline，但落后 Top-1 PPO 约 4.67%。
2. 新增 DP 不受固定 4×8 层边界限制，但真实 reward 仍低于 PPO，说明动作空间更灵活不等于 proxy 选择更准确。
3. Strong-link 的 PPL 比 PPO 低，但 latency 高约 44%；当前 reward 同时考虑质量和 latency，所以总体 reward 反而更差。
4. Top-K 的真实 oracle 只比 surrogate-selected Top-1 好 0.008923，说明 surrogate 排序已经保留了大部分 Top-K 候选增益。
5. 所有方法 invalid fraction 均为 0；本轮比较没有因为非法 assignment 获得或损失优势。

## 可复现产物

- 结果 JSON：[common_seed_baseline_comparison.json](common_seed_baseline_comparison.json)
- 真实 PPL cache：[common_seed_baseline_cache.jsonl](common_seed_baseline_cache.jsonl)
- 评估脚本：[compare_general_assignment_baselines.py](../../../../scripts/ppo/compare_general_assignment_baselines.py)
- baseline 实现：[resource_baselines.py](../../../../src/uav_rl/resource_baselines.py)
- DP 测试：[test_resource_baselines.py](../../../../tests/test_resource_baselines.py)

本轮恢复任务最终记录：697 次真实 CodeLlama forward、1223 次 cache hit，当前进程耗时约 1042 秒；cache 中保留了此前已完成的真实结果。