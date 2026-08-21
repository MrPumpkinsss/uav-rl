# Common-seed baseline comparison

本轮 benchmark 已完成。所有方法使用完全相同的 32 个 channel、4 个 activation-noise seed 和真实 CodeLlama-7b evaluator；旧的 `Fixed-eight additive proxy` 与 `Fixed-eight Compute-greedy` 不再参与正式比较。

配置：

- channel seed：`20260824`
- noise seeds：`4100000`–`4100003`
- CodeLlama：`codellama/CodeLlama-7b-hf`
- 27 个序列，1689 个 evaluated tokens
- PPO Top-K：5 个候选，每个 channel 20 次 surrogate candidate sampling
- proxy beam：128；wide beam：512
- surrogate local search：从 beam 512 初始化，3 轮局部搜索
- 所有最终 reward、PPL 和 latency 均由真实模型评估；surrogate 只用于 PPO/搜索阶段

## 结果

reward 是负的综合代价，因此越接近 0 越好；PPL 和 latency 越小越好。

| 方法 | 真实 reward ↑ | PPL ↓ | 平均 latency (s) ↓ | 相对 PPO Top-1 |
| --- | ---: | ---: | ---: | ---: |
| **PPO Top-1 surrogate-selected** | **-0.400744** | 16.735 | 0.714 | — |
| **PPO Top-5 true oracle**（不可部署上界） | **-0.391821** | 16.407 | 0.714 | +2.23% |
| **Surrogate local search** | **-0.411060** | 17.245 | 0.704 | -2.57% |
| Proxy beam 128 | -0.419458 | 17.567 | 0.701 | -4.67% |
| Wide proxy beam 512 | -0.421463 | 17.658 | 0.700 | -5.17% |
| PPO deterministic | -0.428221 | 17.728 | 0.722 | -6.86% |
| Fixed-eight Strong-link | -0.475804 | 15.310 | 1.027 | -18.73% |
| Dynamic programming | -0.480867 | 15.619 | 1.014 | -19.99% |
| Surrogate random search 1024 | -0.488966 | 16.232 | 0.991 | -22.01% |
| Random feasible | -3.383793 | 860.959 | 5.289 | -744.38% |

所有方法的 `invalid_fraction` 都是 `0.0`。

## 结论

1. 当前最好的可部署方法仍然是 surrogate-trained PPO Top-1，真实 reward 为 `-0.400744`。
2. 新增的 `proxy_beam_surrogate_local_search` 是最强的非 PPO 方法，真实 reward 为 `-0.411060`，比 PPO Top-1 低约 `2.57%`，但比 beam 512 提高约 `2.47%`。
3. `wide_proxy_beam_512` 没有超过 beam 128：`-0.421463` 对比 `-0.419458`。这说明加宽 proxy beam 不能消除 proxy 排序误差。
4. Fixed-eight Strong-link 和 dynamic programming 的 PPL 较低，但通信/分段 latency 较高；当前 reward 是质量和 latency 的综合结果，所以总 reward 不如 PPO。
5. PPO Top-5 true oracle 只用于不可部署的上界参考，因为它需要对候选逐个调用真实 CodeLlama。

## 运行信息

- 总耗时：约 `165.5 s`
- CodeLlama forward：`116`
- 真实模型 cache hits：`1804`
- cache entries：`1512`

详细 JSON：[`common_seed_baseline_comparison.json`](common_seed_baseline_comparison.json)