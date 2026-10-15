# When Are Teacher Tokens Reliable? 解读

完整题名：**When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning**。

依据：[本地 paper.pdf](paper.pdf)，arXiv:2605.21606v1，2026-05-20，20 页。本文依据该版本正文与附录，目录名不作为会议录用证据。

## 1. 这篇文章到底在解决什么问题

核心问题是：**在 student 自己生成的前缀上，拥有参考信息的 teacher 所给出的下一步分布，到底有多可靠？应该给所有位置相同的监督权重吗？**

标准 OPSD 的均匀 token 权重没有显式区分这种可靠性。本文用分支续写实验研究可靠性，然后提出按相对位置加权的 PW-OPSD：早期监督较弱，后期监督较强。

摘要中出现 entropy，是因为作者在质疑它作为可靠性代理的充分性。最终训练权重来自相对位置，并不由 entropy 计算，也没有以降低 entropy 为目标。

## 2. Motivation：为什么需要这个方法

Teacher 的高熵表示多个下一步都有一定概率，但可能对应两种不同现象：

- **不可靠的不确定性**：候选下一步看似合理，student 接着做却很难答对。
- **良性多样性**：存在多个有效解法或表达方式，student 沿不同分支仍能答对。

例如，“设未知数”“画辅助线”“使用代数变换”可能是多个可行方向；另一个前缀上则可能是几条都难以完成的路径。即使局部分布熵相似，它们作为 student 训练目标的价值也可能不同。这个例子用于解释概念，不是论文中的具体实验样本。

Teacher 能看到答案或参考解法，也不保证其建议适合普通输入下的 student。作者因此评价的是：强制采用某个 teacher 候选 token 后，**没有特权信息的 student 能否继续得到正确答案**。

## 3. 背景和已有方法的问题

| 方法 | 调整依据 | 本文关注的不足 |
|---|---|---|
| OPSD | token 均匀权重 | 没有显式估计不同位置的监督可靠性 |
| EOPD | teacher entropy 触发 FKL 增强 | entropy 不能直接区分失败分支与有效多样性 |
| REOPOLD | teacher–student log-ratio、裁剪与熵相关采样 | 与 student 后续是否成功并不等价 |
| PW-OPSD | 相对位置的递增权重 | 用低成本结构先验近似可靠性，仍不是逐状态准确判断 |

论文还讨论了 TODI、GATES。应注意：表 1 直接比较的是几种局部不确定性指标与位置，不能将它扩大解释为对所有分歧或共识方法的全面否定。

## 4. 方法总览

### 4.1 先做诊断：branch viability

依据第 3.2 节及附录 A：

1. 使用 Qwen3-4B 的普通 student 提示生成轨迹，称为 spine。
2. 诊断主要保留最终回答正确的 spine。
3. 同一模型换成含正确最终答案的 teacher 提示，对 spine 的各前缀打分。
4. 根据 top-16 有效 token 的截断熵选择高歧义位置，并加上概率、间距与特殊 token 过滤。每题最多 5 个位置。
5. 每个位置取 teacher 的 top-3 有效候选 token。
6. 将候选 token 强制接在原 student 前缀后；每个候选用普通 student 提示继续生成 6 次，不提供答案。
7. 用最终答案正确率定义候选分支的 viability。

分类标准为：至少两个候选分支的成功率达到 0.75，标为 diversity；全部候选成功率低于 0.40 且均值低于 0.40，标为 real-uncertain；中间情况为 gray，不参与二分类 AUROC。

Teacher 若带着正确答案继续生成，可以借用该答案恢复，这会掩盖建议对 student 的实际帮助。因此续写必须用 student 的输入条件。

**提示信息的区别：**诊断实验附录 A 明确给 teacher 正确的最终答案；它不是逐字复现 OPSD 训练模板。训练设定则沿用 privileged OPSD。不能将诊断实验说成统一向 teacher 提供完整验证推理轨迹。

### 4.2 从诊断到训练

作者发现：在该诊断子集中，早期高歧义位置更容易出现失败分支，后期更常见有效多样性。因此采用递增的位置权重，降低早期的蒸馏强度。

**分支续写是研究诊断，不是 PW-OPSD 每次训练都运行的步骤。**实际训练仍使用普通 student rollout、单次 privileged teacher 打分和原有 clipped FKL。

## 5. 数学推导和关键公式

### 5.1 Entropy 测量什么

$$
H(q_t)=-\sum_{j\in\mathcal V}q_t(j)\log q_t(j).
$$

$q_t$ 为 teacher 在当前前缀上的 token 分布。熵刻画该分布的分散程度，不包含各 token 之后最终能否答对的标签。因此相同熵可以对应完全不同的后续成功率。

### 5.2 原始 OPSD 的实现目标

令 $q_t=p_t^{\mathrm{opsd}}$ 为 privileged teacher 分布，$p_t$ 为普通 student 分布：

$$
\ell_t=\sum_{j\in\mathcal V}\min\left(q_t(j)\log\frac{q_t(j)}{p_t(j)},\tau_{\mathrm{clip}}\right),
\qquad
L_{\mathrm{OPSD}}=\frac1{|\mathcal M|}\sum_{t\in\mathcal M}\ell_t.
$$

$\mathcal M$ 是有效 token 集合。裁剪针对每个词表项，在词表求和前执行；不是把整项 KL 裁剪，也不是 PPO 的概率比裁剪。裁剪后是 FKL surrogate，不宜把它当作未经改动的 KL 散度。本文保持这个内部损失形式，改变外部聚合。

### 5.3 可靠性加权解释

设 $I_t\in\{0,1\}$ 表示 teacher 目标对保持或恢复正确解是否可靠，$c_t=(x_{\mathrm{stu}},x_{\mathrm{tch}},y_{<t})$，$D_t=\mathrm{KL}(q_t\|p_t)$。作者定义：

$$
R(\theta)=\mathbb E[I_tD_t(\theta)]
=\mathbb E[\rho^*(c_t)D_t(\theta)],
\qquad \rho^*(c_t)=P(I_t=1\mid c_t).
$$

固定模型与 teacher 时，$D_t$ 由上下文确定，等式由条件期望得到。它说明理想可靠性过滤风险可以用可靠性后验加权表示。**它没有证明位置 sigmoid 就是真实后验，也没有证明降低该损失必然提升正确率。**采样后的轨迹视为固定训练数据，更新不对 rollout 采样过程求导。

### 5.4 分支混合恒等式

若 teacher 分布为分支混合 $q_t=\sum_z\alpha(z\mid h_t)q_t^z$，则：

$$
\mathbb E_z[\mathrm{KL}(q_t^z\|p_t)]
=\mathrm{KL}(q_t\|p_t)+I_q(Y_t;Z\mid h_t).
$$

$Z$ 是潜在分支，$h_t$ 是 student 前缀，互信息项度量混合中隐藏的分支差别。证明可将对数比拆成 $\log(q_t^z/q_t)+\log(q_t/p_t)$ 后求和。附录 D 的序列版本把互信息按时间累加。

互信息项不依赖 student 分布，因此此恒等式本身不会给出不同的 student 梯度；论文也明确承认这一点。它帮助解释多样性，不能独立推出“早期应该降低权重”。位置方向来自实验。

### 5.5 PW-OPSD 的最终目标

对第 $i$ 条长度为 $L_i$ 的有效轨迹，第 $t$ 个 token 的相对位置和权重为：

$$
r_{i,t}=\frac{t-0.5}{L_i},
\qquad
w_{i,t}=w_{\min}+(1-w_{\min})\sigma\left(\frac{r_{i,t}-\tau}{s}\right).
$$

默认 $(w_{\min},\tau,s)=(0.25,0.30,0.10)$。下限保证早期仍有监督；$\tau$ 控制转折位置；$s$ 控制平滑程度。位置按每条序列的实际长度计算，不按 batch padding 长度计算。

$$
L_{\mathrm{PW\text{-}OPSD}}
=\frac1B\sum_{i=1}^B\frac1{L_i}\sum_{t=1}^{L_i}w_{i,t}\ell_{i,t}.
$$

该目标先在每条序列内平均，再跨序列平均。因此相对 OPSD 有两个变化：位置权重，以及 per-sequence reduction。分母是 $L_i$，不是权重和；加权也改变总体损失尺度。

## 6. 为什么这个方法可能有效

在可靠性确实随位置增长的条件下，权重让更可能不可靠的早期监督对梯度的影响减小，并保留更可靠的后期监督。Per-sequence reduction 同时避免长轨迹仅因 token 更多而获得更多总权重。

这是经验支持的结构先验，不是对每个前缀的独立判定。较早的步骤可能决定后面大量推理，较晚的步骤可能受到正确 spine 已经形成的约束，这是一种可能解释；论文的相关性结果本身不能证明这些因果机制。

## 7. 为什么已有方法不足：它们忽略了什么

均匀权重忽略了监督可靠性的变化。局部 entropy 则忽略“候选的后续可执行性”：模型对多个 token 都给较高概率，不代表这些 token 会让普通 student 失败。

但本文并未证明 entropy 在所有 OPD 场景都无用。其候选本来就是高截断熵位置，限制了比较范围；筛选条件也可能影响 entropy 的区分能力。更合理的结论是：**在所测高歧义分支中，entropy 不足以区分可靠多样性和难以恢复的分支。**

## 8. 真正的贡献及与我们 idea 的关系

1. 把 teacher 的局部分布歧义，与普通 student 的分支可行性区分开。
2. 用“teacher 提议候选、student 强制分支后续写”进行可靠性诊断。
3. 找到诊断样本中的位置趋势，并把它转成低成本的位置加权方法。

这与之前 [RL 迁移 idea](../../idea/传统RL方法迁移到OPD-对比与研究思路.md) 中的“用 student 后续成功率检验 teacher 指导”有直接重合。因此，该诊断思路已是明确相关工作，不能作为未被研究过的新点。

仍需区分：本文主要在**正确 spine 的高歧义位置**进行诊断，训练只使用固定位置先验；我们讨论的**错误前缀上的纠错、逐状态或候选动作的动态价值修正**并不等于本文训练方法。不过这些方向的新颖性仍需进一步检索，不能据此直接断言空白。

## 9. 实验、消融和证据链

### 9.1 诊断样本与统计结果

84 道初始题目中得到 62 条正确 spine，去除仅有 gray 候选的题目后剩 61 题。二分类样本含 8 个 real-uncertain 与 271 个 diversity 候选。

| 特征 | 题内去均值后的 AUROC |
|---|---:|
| Teacher predictive entropy | 0.51 |
| MC-dropout mutual information | 0.51 |
| Dirichlet precision | 0.54 |
| Top-16 截断熵 | 0.57 |
| 早期位置分数 $1-r$ | 0.83 |

位置 AUROC 的题目级 bootstrap 95% 区间为 [0.66, 0.95]。MC-dropout 仅供诊断，使用最后四层的推理时扰动、5 次样本，不是训练模块。

### 9.2 下游结果

| Qwen3-4B，Avg@12（%） | OPSD | PW-OPSD Moderate | 差值（百分点） |
|---|---:|---:|---:|
| MATH-500 | 95.33 | 95.34 | +0.01 |
| AIME 2024 | 75.19 | 76.20 | +1.01 |
| AIME 2025 | 66.67 | 67.78 | +1.11 |
| HMMT 2025 | 43.89 | 43.33 | -0.56 |

Aggressive 版本在 HMMT 2025 达到 45.37，但不是默认 Moderate。不能混用不同 schedule 的最佳结果作为同一配置表现。

固定 Moderate 后，四基准均值在 Qwen3-4B、DeepSeek-R1-Distill-Llama-8B、Olmo-3-7B-Think 分别提升 0.39、0.35、0.50 个百分点；不是每个单项都提高。

### 9.3 两个修改需要联合看

附录 L 的 AIME 2024 Avg@12：

| 聚合方式 | 无位置权重 | 有位置权重 |
|---|---:|---:|
| 全局 token 平均 | 75.19 | 74.54 |
| 每序列平均 | 74.72 | 76.20 |

在该实验中，两项一起改变才有收益，单独加位置权重反而下降。因此不能把全部提升简单归因于位置趋势。

### 9.4 评估条件

训练采用 LoRA rank 64、固定 teacher（teacher forward 关闭 LoRA）、学习率 $5\times10^{-6}$、100 步 checkpoint、训练生成上限 1024 token。评估生成上限为 38,912 token，每题采样 12 次。

Avg@12 是 12 次回答的平均正确率；Pass@12 是至少一次答对；Maj@12 是最多票答案的正确率。报告的三个随机种子是**评估种子**，不能理解为三个独立训练种子所证明的训练稳定性。

## 10. 局限和适用边界

- 诊断的失败类别仅 8 个候选，类别不平衡且置信区间较宽。
- 正确 spine、高歧义位置和过滤条件限制了结论的适用范围，不能外推成“所有早期 token 都不可靠”。
- 每候选仅 6 次续写，阈值标签带有采样不确定性，gray 区间被排除。
- 分支成功率是行为代理，不是“沿 teacher KL 更新后是否提升任务表现”的直接因果测量。
- 相对位置是粗粒度先验，不能识别后期突然出错或早期必须强监督的情况。
- 数学恒等式支持可靠性加权的解释，不证明位置调度的最优性。
- 实验收益较小且依赖指标和 schedule；没有统一改善所有 benchmark 的所有指标。
- 附录 H 披露了共享的 prompt padding、训练/评估模板差异等实现约定，复现时需要纳入解释。
- 诊断需要额外分支生成和多次打分；“无额外 teacher 计算”指最终训练方法相对 OPSD，而非整个研究诊断过程。

## 11. 总结：完整故事线

**均匀蒸馏不区分可靠性 → 高 entropy 有多种含义 → 让 student 沿 teacher 候选分支继续生成以测量可行性 → 在选定样本中发现早期更不可靠 → 用递增位置权重与每序列平均修改 OPSD → 获得有限但可复用的收益。**

读这篇最应记住：entropy 是局部分布歧义，reliability 关心指导对 student 后续完成任务是否有用；PW-OPSD 最终用位置近似后者。
