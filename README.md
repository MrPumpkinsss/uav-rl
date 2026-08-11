# UAV-RL

本项目用于复现论文 *Robust Collaborative LLM Inference in UAV-enabled Wireless
Networks* 中的分层 LLM 部署、无线传输扰动、PPL 和 DROS 优化实验。

论文目前仍是未完成稿，评估章节没有给出 CodeLlama 型号、数据切片和运行硬件。
为使第一阶段结果可复现，本项目将“一次 PPL 计算”定义为：

- 模型：`codellama/CodeLlama-7b-hf`
- 数据：WikiText-2 test 前 50 条原始记录，过滤空文本
- 输入：最大 512 tokens，batch size 4
- 精度：BF16；使用 eager attention，避免 activation dropout 后的 FP16 数值溢出
- PPL：按所有有效 next-token 的总 NLL 加权，符合论文中的 corpus PPL 公式
- 计时范围：所有批次的 forward、交叉熵和 PPL 汇总，不包含模型/数据加载和分词
- 流程：首个真实 batch 预热，完整计算 3 次并同步 GPU

## 目录结构

```text
uav-rl/
├── src/uav_rl/
│   ├── benchmarks/       # 完整实验耗时基准
│   ├── data/             # 离线 CodeLlama PPL 数据生成
│   ├── metrics/          # PPL 等指标实现
│   ├── models/           # activation dropout
│   └── rl/               # 连续分层部署策略、环境和 PPO
├── scripts/              # 命令行实验入口
├── tests/                # 单元测试
├── artifacts/            # 生成的基准结果，不提交 Git
├── Robust_Inference2025-main.pdf
└── *.py                  # 原始研究原型，后续按 DROS 流程逐步迁移
```

原始 `main.py`、`main_2.py`、`noise_ppl.py` 和 PPO 文件保留用于追溯，但存在硬编码
路径、导入即加载模型以及算法与论文 DROS 不一致等问题，不应作为新的实验入口。

## 环境安装

建议使用 Python 3.11：

```bash
conda create -n uav-rl python=3.11 -y
conda run -n uav-rl python -m pip install \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n uav-rl python -m pip install -e '.[dev]'
```

安装完成后可运行测试和静态检查：

```bash
conda run -n uav-rl pytest -q
conda run -n uav-rl ruff check .
```

## PPL 基准

```bash
conda run -n uav-rl python scripts/benchmark_ppl.py
```

结果会打印到终端，并保存到
`artifacts/benchmarks/codellama_ppl.json`。改变口径时必须显式传参，例如：

```bash
conda run -n uav-rl python scripts/benchmark_ppl.py \
  --sample-limit 100 --max-length 512 --batch-size 4 --runs 3
```

首次运行需要从 Hugging Face 下载约 13 GB 的 CodeLlama-7B FP16 权重。

## 离线数据和 PPO

每个动作直接为 32 个 decoder layer 选择 UAV。每个 UAV 最多承载 8 层，且其
承载层必须构成一个连续区间；策略离开某个 UAV 后不能再次返回。PPL 数据只在
预生成阶段调用 CodeLlama，surrogate 和 PPO 训练均不加载大模型。

动态规划基线使用状态 `(已分配层数, 已使用 UAV 集合, 最后一个 UAV)`，每次转移
为一个尚未使用的 UAV 分配 1 至 8 个连续层，因此严格满足容量和连续性约束。由于
surrogate PPL 和多链路最优带宽分配都依赖完整部署，无法直接写成该 Bellman 状态
下的可加成本，DP 在选动作时精确最小化以下代理：边界丢包率之和，加上归一化的
计算时延和各链路独占全带宽时的通信时延；两部分仍使用实验的 0.5/0.5 权重。DP
选出的部署最终仍由相同 surrogate 或真实 CodeLlama PPL、相同多链路带宽模型和
相同 reward 统一评估，代理值不作为论文结果上报。

两个 surrogate oracle 直接优化完整 surrogate reward，不使用 DP 的可加代理。
`four_segment_surrogate_oracle` 穷举 120 个四段满容量部署；
`full_surrogate_oracle` 穷举当前约束下全部 58,920 个合法部署，其中包括 120 个
四段部署和 58,800 个五段部署。候选集只生成一次，丢包率、联合带宽时延和 reward
按候选向量化计算，surrogate 分批推理以限制显存占用。Oracle 只表示 surrogate
reward 下精确最优；真实 CodeLlama PPL 不参与动作搜索，因此它们不是 true-LLM
oracle，也不应被解释为真实 reward 上界。

```bash
python scripts/generate_ppl_dataset.py --samples 256 --seed 20260810
python scripts/append_tail_ppl_dataset.py \
  --dataset artifacts/data/codellama_ppl_dataset_1024.npz \
  --samples 256 --candidate-pool 8192
python scripts/append_coverage_ppl_dataset.py
python scripts/train_surrogate.py
python scripts/train_and_evaluate_ppo.py --episodes 8192 --test-channels 256
```

生成的数据、模型和结果分别位于 `artifacts/data`、`artifacts/models` 和
`artifacts/results`。这些运行产物默认不提交 Git。

当前 surrogate 数据共 3328 条，由 1024 个原始随机样本、256 个高丢包尾部样本
和 2048 个边界覆盖样本增量叠加而成，旧样本没有重新计算。模型以 31 个逐边界
丢包率和 5 个确定性累计统计特征为输入，所有特征仅使用训练集统计量标准化。
固定 80/20 划分上的验证 MAE 为 0.05993（log PPL ratio），验证 R2 为 0.90157。

PPO 使用 8192 个独立教师信道进行行为克隆预热。教师在 120 个合法四段动作中
搜索，并用现有数据中的 523 个同形真实 PPL 样本构建 8-NN 局部质量模型；随后
运行 8192 个 PPO episode。teacher、validation 和 test seed 分别为 20260814、
20260813 和 20260811，配置会拒绝三者重复。最终 checkpoint 来自第 1024 个 PPO
episode，因而不是仅预热的 episode-0 参数。

在相同的 256 个固定测试信道上，七种方法均使用真实 CodeLlama token 加权 corpus
PPL。PPO 使用最终 checkpoint 独立重算；六个基线来自同一测试 seed、Random seed、
noise seed 和 latency reference 的真实 benchmark：

| 方法 | 真实 Reward | 延迟（秒） | 真实 PPL 均值 | 真实 PPL 中位数 |
| --- | ---: | ---: | ---: | ---: |
| PPO | -0.4658 | 1.0216 | 14.9103 | 14.8736 |
| Random | -0.6563 | 1.2854 | 22.1830 | 16.3406 |
| Compute-greedy | -0.5789 | 1.0717 | 27.5501 | 15.8300 |
| Strong-link | -0.4702 | 1.0385 | 14.8476 | 14.8038 |
| Dynamic programming | -0.4654 | 1.0206 | 14.9108 | 14.8736 |
| Four-segment surrogate oracle | -0.4670 | 1.0195 | 14.9697 | 14.9437 |
| Full surrogate oracle | -0.4670 | 1.0195 | 14.9697 | 14.9437 |

真实 reward 下，PPO 比 Random 高 29.03%，比 Compute-greedy 高 19.53%，比
Strong-link 高 0.93%，但比 Dynamic programming 低 0.079%。DP 的平均 PPL 比
PPO 高 0.00056，平均延迟低约 1.00 ms，因此在 0.5/0.5 加权 reward 下略优于
当前 PPO。这个结果保留真实差距，不用 DP 的代理成本替代最终 reward。

两个 surrogate oracle 在 256 个信道上逐信道选择了完全相同的四段动作。在 RTX
3090 上，四段搜索用时 0.24 秒，全量搜索用时 10.40 秒；surrogate reward 都为
-0.47225，优于 DP 的 -0.47365 和 PPO 的 -0.47417。但真实 reward 均为 -0.46702，
反而低于 DP 和 PPO，主要因为真实 PPL 均值上升到 14.9697。这说明最大化 surrogate
会放大其局部预测误差；即使验证 R2 为 0.90157，也不能把 surrogate oracle 当作
真实上界。两个方法的 512 个方法-信道评分共享相同动作缓存，只执行了 256 次唯一
CodeLlama PPL。

合并后的同信道结果位于
`artifacts/results/ppo_true_llm_evaluation_3328_local_teacher_ppo_combined.json`；文件中
分别记录了 PPO 与基线原始结果的来源路径。

真实大模型评估可独立复现：

```bash
python scripts/benchmark_true_policy_ppl.py --test-channels 256

# 已有同信道基线结果时，只重算新 PPO：
python scripts/benchmark_true_policy_ppl.py --methods ppo --test-channels 256

# 只计算新增 DP，复用已有同信道方法的结果：
python scripts/benchmark_true_policy_ppl.py \
  --methods dynamic_programming --test-channels 256 \
  --output artifacts/results/dynamic_programming_true_llm_evaluation_3328.json

# 只评估两个 surrogate oracle：
python scripts/benchmark_true_policy_ppl.py \
  --methods four_segment_surrogate_oracle full_surrogate_oracle \
  --test-channels 256 \
  --output artifacts/results/surrogate_oracles_true_llm_evaluation_3328.json
```
