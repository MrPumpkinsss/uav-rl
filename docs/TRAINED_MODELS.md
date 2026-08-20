# Trained checkpoints

本页列出当前已经发布到 Git 仓库的训练 checkpoint。它们是可以直接用于 inference、surrogate reward 或真实模型验证的模型文件；历史实验模型仍保留在本地 `artifacts/` 或归档目录中，不全部上传。

## 已发布模型

| 类型 | Git 路径 | 用途 | 大小 | SHA256 |
| --- | --- | --- | ---: | --- |
| General surrogate ensemble | [`artifacts/models/ppl_surrogate_general_assignment_ensemble.pth`](../artifacts/models/ppl_surrogate_general_assignment_ensemble.pth) | 当前通用 assignment surrogate 默认 checkpoint | 5.41 MiB | `c2fd82f0df6e56e83a80bc492b9f9460ec3fd09210b52d53ea3dd02418f921ff` |
| High-augmented surrogate ensemble | [`artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth`](../artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth) | common-seed baseline 对比使用的 surrogate | 5.41 MiB | `74a472278231db33560f6a57801a0af25a91d5d6bfa18a67bfe8fc203b0d84df` |
| Original Top-K PPO | [`artifacts/runs/surrogate_ppo/layerwise_topk_2026-08-20b/best_policy.pth`](../artifacts/runs/surrogate_ppo/layerwise_topk_2026-08-20b/best_policy.pth) | 1000-episode 原始 Top-K PPO policy，使用通用 surrogate | 1.10 MiB | `be93b847e2bfb8cc1a889e8b0a36b034e05cb09568270731426d84554d176f48` |
| High-augmented Top-K PPO | [`artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth`](../artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth) | common-seed baseline 报告实际使用的 policy | 1.10 MiB | `f2ab36cde8eacf85bab27b758e14982e71321c934e6fccaf9f9c5cb94b949088` |
| Direct true-PPL PPO | [`artifacts/models/ppo_true_ppl_multiseed_best.pth`](../artifacts/models/ppo_true_ppl_multiseed_best.pth) | 不使用 surrogate、直接用真实 CodeLlama PPL 训练的 1000-episode PPO best policy | 4.84 MiB | `e424300aa7c113d72402cb4660873b43fe668d4a2d8e6d2fa1b9be9edf5b19f4` |

文件大小使用十进制字节换算为 MiB 展示；SHA256 以本地 checkpoint 原始字节计算。

## 对应配置和结果

- 通用 surrogate 训练入口：`scripts/surrogate/train_general_assignment.py`
- Top-K PPO 训练入口：`scripts/ppo/train_layerwise_topk.py`
- 真实 PPL PPO 训练入口：`scripts/ppo/train.py`
- high-augmented Top-K 的运行配置：`artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/run_config.json`
- common-seed baseline 结果：`artifacts/runs/surrogate_ppo/common_seed_baseline_comparison.json` 和 `.md`
- 真实 PPL PPO 报告：`artifacts/results/ppo_true_ppl_multiseed_1000_report.md`

运行配置和结果文件同样属于实验产物；如果它们没有被 Git 跟踪，仍可在运行机器的本地 `artifacts/` 中找到。模型 checkpoint 自身已经纳入 Git，clone 仓库后可以直接加载。

## 不上传的文件

`training_state.pth` 包含 optimizer state、训练历史和 Python/NumPy/PyTorch/CUDA 随机数状态，主要用于本地无损 resume，不是部署 inference 所必需的模型。它们体积更大且依赖对应的 run directory，因此当前不纳入公开 checkpoint 列表。

surrogate 的旧 tail/gated/residual 变体也没有全部发布。它们属于历史实验或诊断模型，位于本地 `artifacts/models/`、`artifacts/results/` 或 `artifacts/archive/`，不能从模型文件名推断为当前默认入口。

## 完整性检查

在 clone 后检查 SHA256：

```powershell
Get-FileHash -Algorithm SHA256 artifacts/models/ppl_surrogate_general_assignment_ensemble.pth
Get-FileHash -Algorithm SHA256 artifacts/models/ppl_surrogate_general_assignment_high_augmented_ensemble.pth
Get-FileHash -Algorithm SHA256 artifacts/models/ppo_true_ppl_multiseed_best.pth
Get-FileHash -Algorithm SHA256 artifacts/runs/surrogate_ppo/layerwise_topk_2026-08-20b/best_policy.pth
Get-FileHash -Algorithm SHA256 artifacts/runs/surrogate_ppo/layerwise_topk_high_augmented_2026-08-20/best_policy.pth
```

加载 surrogate 时使用 `load_surrogate()`；加载 PPO policy 时使用与对应 `run_config.json` 相同的 `SystemConfig`、resource config 和 policy 类型。不要把不同 run 的 policy 和 surrogate 随意混配。