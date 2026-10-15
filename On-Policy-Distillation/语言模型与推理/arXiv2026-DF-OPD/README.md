# DF-OPD：没有外部题库，OPD 还能学到多少？

## 论文与归档

- **完整题名**：Data-Free On-Policy Distillation: How Far Can We Go Without External Data?
- **论文**：[arXiv:2609.14193](https://arxiv.org/abs/2609.14193)，本目录保存 v4，2026-10-07，31 页。
- **用户提供入口**：[Hugging Face Papers](https://huggingface.co/papers/2609.14193)。
- **官方仓库**：[Ryuki661/DF-OPD](https://github.com/Ryuki661/DF-OPD)。本地 `/usr/data/DF-OPD`，本目录 `main` 为软链接。
- 当前按 arXiv 预印本归档，没有据此认定会议录用状态。来源、版本、哈希见 `SOURCE.txt` 和 `manifest.jsonl`。
- 截至此次归档，官方仓库只有 README 和图片，没有公开训练实现；克隆仓库不等于已经取得可复现代码。

## 1. 核心问题

**OPD 到底依赖多少外部训练题目？如果 teacher 已经拥有能力，能不能由它自己出题，再把能力蒸馏给 student？**

这篇主要是数据依赖的实证研究。它没有提出新的 token 蒸馏损失，而是逐步减少外部题目：大量真实题目 → 少量真实题目 → teacher 自生成题目 → 空 prompt。

## 2. 具体怎么做

1. 训练之前，用包含领域、主题要求的简单指令，让冻结的 teacher 生成一个固定题库；不提供外部 seed 示例，也不要求它同时生成参考解答。
2. 对题目做提取、格式检查和去重。论文没有按答案正确性、难度或解答质量分数筛题。
3. student 从这个题库抽题，并自行生成回答。
4. 同一个 teacher 在 **student 已经生成的前缀** 上预测下一 token，提供分布反馈。
5. 反复采样和更新 student；teacher 保持冻结，题库也不随每轮训练重新生成。

记自生成题库为 $D_T$，论文沿用 reverse-KL OPD：

$$
\mathcal L(\theta)=\mathbb E_{x\sim D_T,\,y\sim\pi_\theta(\cdot|x)}
\sum_t D_{\mathrm{KL}}\left(
\pi_\theta(\cdot|x,y_{<t})\;\|\;
\pi_T(\cdot|x,y_{<t})\right).
$$

实际计算使用 top-k surrogate。这里改变的是题目来源，不是 student rollout、teacher 逐 token 监督这一机制。

多 teacher 场景下，各领域 teacher 分别出题，训练时根据题目的领域标签路由到对应 teacher。

## 3. 为什么很少的题目也可能有效？

**题目数量不等于监督数量。** 同一道题多次采样会产生不同回答和前缀，因此会暴露不同的 teacher 纠正信号。随着 student 更新，同一道题后续访问的状态也会变化。

论文用监督信号的线性重构覆盖度支持这一解释：冻结初始 student，对一道题的采样从 1 条增加到 4096 条，覆盖度从 0.114 增至 0.507。这个代理指标提供证据，但不是“少题必然充分”的证明。

另一条证据是跨领域迁移：用编程题训练、仍评估数学时，两个单 teacher 设置保留了数学题训练增益的 96% 和 92%。作者据此认为，OPD 可以传递超越具体题目知识的推理能力。

## 4. 主要实验结果与边界

单 teacher 的两个设置是：

- DeepSeek-R1-Distill-Qwen-1.5B student 与同规模、经过 RL 的 JustRL-DeepSeek-1.5B teacher。
- Qwen3-1.7B student 与 Qwen3-30B-A3B-Instruct-2507 teacher。

| 比较 | 论文观察 |
|---|---|
| 少量真实题目 | 8 道题可达到与约 17k 真实题目相近的表现 |
| 64 道自生成题目，RL teacher | 数学宏平均 56.61，完整真实题库 56.41 |
| 64 道自生成题目，大模型 teacher | 49.16，完整真实题库 47.36 |
| 多 teacher：数学、代码、指令遵循 | 1k 自生成题目的 Overall 为 31.31，7458 道真实题目为 31.18 |

这些是论文所测模型和任务上的结果，主指标使用评估过的 checkpoint 中的峰值。不能把“64 题足够”直接推广到所有模型、领域或 teacher。

## 5. Data-free 的准确含义

这里指 **蒸馏阶段不需要外部训练题库**。teacher 和 student 已经预训练，teacher 出题包含已有知识；流程仍然有题目、rollout、teacher 推理和参数更新。

题库小也不代表训练计算量按题目数量同比减少。论文在单 teacher 设置中每批仍使用 64 个题目槽位，小题库通过重复题目填满 batch，再生成新轨迹。

## 6. 空 prompt 可以吗？

只有有限设置有效。一个设置中 student 会在回答内部主动出题再解答，即 Self-QA；这相当于把题目生成藏进 rollout，而不是完全消除了问题。

RL teacher 设置中，Self-QA 比例为 62.0%，空 prompt 保留约 95.6% 的数学增益；较大 teacher 设置仅保留约 7.4% 的增益。因而论文不支持普遍用空 prompt 替代训练题目。

## 7. 和我们前面讨论的联系

- 这是标准 OPD：回答由 student 生成，teacher 依据同一前缀给分布反馈。
- 它没有通过“teacher 看到参考解法”构造 OPSD 的 privileged teacher，也不是 ReDraft 那种修订文本再 SFT。
- 它没有专门修复错误前缀、定位关键动作或筛除错误 teacher 信号；这些仍可与数据生成机制结合。
- 一个可探索方向是结合 Dr.OPD/RLCSD 的信号选择或奖励调制，再考察自生成题库是否需要自适应扩充。此项是研究设想，不是本文已验证的结论。

## 阅读定位

- §2：OPD 目标、模型及实验设置。
- §3：题目数量、难度、初始 distillation gap 与监督覆盖分析。
- §4：DF-OPD、单 teacher 与多 teacher 实验。
- §5：空 prompt 与 Self-QA。
- Appendix E：出题模板、处理过程与完整结果。

本目录仅归档与阅读整理，未安装依赖或运行训练。
