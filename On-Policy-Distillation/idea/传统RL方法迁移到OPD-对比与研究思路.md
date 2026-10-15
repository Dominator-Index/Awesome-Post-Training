# 传统 RL 方法迁移到 OPD：方法对比与研究思路

整理日期：2026-10-07。

本文记录围绕 TOP-D、OPSD 和错误前缀监督的讨论。下面的实验方案属于待验证的研究假设；所列参考文献并非完整相关工作检索，不能据此认定方法具有新颖性。

## 1. 出发点：TOP-D 借鉴了什么？

[Trust Region Policy Distillation（TOP-D）](https://arxiv.org/abs/2607.04751) 明确借鉴传统强化学习的 trust region 思想，包含两个层面：

1. **构造更接近 student 的 teacher**：在概率空间混合原始 teacher 和当前采样策略，缓和极端监督信号。
2. **限制策略更新幅度**：使用类似 PPO 的裁剪目标，在控制更新幅度的同时复用已生成的轨迹。

用采样时的 student 策略 $\pi_{\mathrm{old}}$ 表示混合中的 student，可写为：

$$
\tilde\pi_T(a\mid s)
=\alpha\pi_T(a\mid s)+(1-\alpha)\pi_{\mathrm{old}}(a\mid s),
\qquad 0<\alpha<1.
$$

对应的采样阶段信号为：

$$
\tilde r(s,a)
=\log\frac{\tilde\pi_T(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}
=\log\left(\alpha\frac{\pi_T(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}+1-\alpha\right).
$$

该信号下界为 $\log(1-\alpha)$，可以缓和 teacher 对 student 所采样 token 赋予极低概率时的负向信号。这个下界本身不保证 teacher 的指导符合最终任务正确性。

本地论文：[TOP-D PDF](../训练机制与稳定性/NeurIPS2026-Trust-Region-Policy-Distillation/paper.pdf)。TOP-D 与 [Trust Region On-Policy Distillation（TrOPD）](https://arxiv.org/abs/2606.01249) 是不同论文。

## 2. 核心问题：局部偏好与最终成功之间的差距

设状态 $s_t=(x,y_{<t})$ 为题目与 student 生成的前缀，动作 $a_t=y_t$ 为下一个 token。对于常见的 reverse-KL 型 OPD，局部采样信号为：

$$
r_t^{\mathrm{OPD}}
=\log\pi_T(a_t\mid s_t)-\log\pi_S(a_t\mid s_t).
$$

这个信号反映 teacher 与 student 的概率偏好差异，未必等价于该动作对最终正确率的贡献。不同 OPD 实现也可能采用全词表 KL 或其他散度，不能将上述估计器视为所有 OPD 的统一实现。

例如，student 已经生成了错误前缀：

```text
2+3=6，因此……
```

两种可能的后续是：

- A：继续使用 6 计算。
- B：指出前面算错，改为 2+3=5 后重新计算。

Teacher 可能在局部预测上偏好连贯地继续 A，但 B 可能更有利于最终答对。这是需要测量的现象，不能预先假定所有 teacher 都会如此。

在 [Self-Distilled Reasoner / OPSD](https://arxiv.org/abs/2601.18734) 中，teacher 可以看到经过验证的参考推理过程，而 student 没有这部分输入。参考解法带来信息优势，但 teacher 仍需将它转化为适合 student 当前状态的指导。

由此提出研究问题：

> 如何判断 teacher 在 student 当前前缀上给出的指导，能否提高 student 后续独立完成任务的成功率？

## 3. 可借鉴的传统 RL 方法对比

| 方向 | 传统 RL 思路 | 在 OPD 中的迁移方式 | 针对的问题 | 代价与限制 | 已知相关工作及定位 |
|---|---|---|---|---|---|
| 更新稳定性 | Trust region、PPO 裁剪 | 构造邻近 teacher，限制每次更新 | 极端信号、更新过大 | 稳定不代表监督正确 | TOP-D 已直接采用 |
| 方差降低 | Baseline、控制变量 | 对采样蒸馏信号减去状态 baseline | 梯度噪声大 | 无法单独纠正系统性错误指导 | KL for a KL 已直接研究 |
| 长期价值 | Actor–Critic、优势估计 | 估计某个动作对 student 最终成功率的影响 | 局部偏好与长期任务结果不一致 | 额外采样、critic 偏差及策略更新后的失配 | OPPO、PACT 等已涉及，需进一步比较 |
| 信用分配 | 多步回报、GAE | 将后续结果传递给前面的关键选择 | 早期决策的影响延迟出现 | 多步估计的偏差与方差；critic 错误可能传播 | 不能仅以“加入 GAE”主张新颖性 |
| 分支评价 | Rollout evaluation、基于后续代价的模仿 | 从同一前缀比较不同候选步骤的最终效果 | teacher 推荐是否适合当前 student | 分支生成成本高 | 与 AggreVaTe 等思想相关 |
| 数据利用率 | 经验回放、重要性采样、V-trace | 复用旧轨迹并修正策略分布差异 | 采样和 teacher 监督昂贵 | 长序列权重方差、截断偏差、缓存过期 | 需对比 TOP-D、AsyncOPD、f-OPD 等 |

### 3.1 长期价值：评价 student 能否执行指导

对可验证任务，定义：

$$
Q^{\pi_S}(s,a)
=P(\text{最终答对}\mid s,\text{先执行 }a,\text{随后由 }\pi_S\text{ 继续}),
$$

$$
A^{\pi_S}(s,a)=Q^{\pi_S}(s,a)-V^{\pi_S}(s).
$$

这里关键是使用 **student 后续作答的价值**。某个复杂推理步骤可能适合 teacher，但 student 接下去无法完成；用 teacher 的续写成功率代替 student 的成功率，会产生能力错配。

可研究的机制包括：利用 advantage 修正动作偏好；对不可靠监督降低权重；在支持充分时构造候选步骤之间的偏好监督。这些选择具有不同优化目标，需要分别推导和比较。尤其是，对整个状态的 KL 损失统一加权只能改变该状态的训练强度，不能保证改变状态内部对 A、B 两个动作的相对偏好。

### 3.2 多步回报与 GAE：先明确奖励目标

传统 GAE 通过 TD 误差构造优势估计：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\qquad
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

需要区分两种设计：

- **累计蒸馏信号**：优化 teacher–student 分布匹配相关目标。
- **引入最终正确性奖励**：优化含任务奖励的混合目标。

把 teacher 的局部概率偏好累加，并不会自动获得正确性监督。TOP-D 第 2.1 节讨论了完整序列 reverse KL 梯度中的累计信号，以及实践中采用即时信号的做法。因此，采用多步估计时应明确目标、估计器和近似，而非将两者混用。

### 3.3 经验回放：区分两种过期

复用旧样本需要区分：

1. **轨迹过期**：生成样本的 student 已经不同于当前 student。
2. **监督过期**：如果 OPSD 的 teacher 随模型更新，缓存的 teacher 分布也可能过期。

可以联合考虑轨迹年龄、策略差异、teacher 变化和经验价值。高 teacher–student 分歧不一定意味着高学习价值，也可能表示 teacher 在这个错误前缀上不可靠。

重要性采样和截断不能被当作长序列分布失配的自动解法；需要说明动作分布和前缀分布分别如何处理，以及引入了何种偏差。

## 4. 优先研究假设：以 student 后续成功率检验 teacher 指导

### 假设

参考推理过程让 teacher 获得正确解法的信息，但它给出的下一步未必能由当前 student 执行。若在关键前缀上评估 student 的后续成功率，并据此修正指导，可能改善错误恢复和最终正确率。

### 最小机制原型

1. Student 在不看参考解法的情况下生成轨迹。
2. 在少数推理步骤边界选择前缀，例如存在明显计算错误、teacher–student 分歧或纠错机会的位置。
3. 为同一前缀构造少量候选步骤，覆盖 student 原始续写、teacher 建议和可行的纠错续写。
4. 每个候选步骤之后，都让 student 独立继续完成回答；续写时不提供参考解法。
5. 使用答案验证器估计候选步骤的成功率，并报告采样不确定性。
6. 根据估计结果构造动作或步骤级监督，与原始 OPD 的指导比较。

使用完整推理步骤可提升语义可解释性，但步骤包含多个 token，其概率、长度效应和训练损失需要单独定义。上述方案仍以 student 访问的前缀为起点；如果候选步骤由 teacher 或人工规则提供，这部分分支数据并非完全由 student 当前策略采样，应明确其混合采样性质。

### 应先做的诊断

在训练新方法前，先测量：**teacher 对候选步骤的偏好，与 student 执行该步骤后的成功率是否一致？**

若不一致主要集中于错误前缀或能力跨度较大的步骤，这支持针对性修正的动机。若二者已高度一致，额外分支计算可能没有足够收益。

## 5. 对比实验设计

下表是实验计划，尚未实施。

| 组别 | 方法 | 要回答的问题 |
|---|---|---|
| A | 原始 OPSD | 特权信息自蒸馏的基础表现如何？ |
| B | OPSD + TOP-D 式概率混合与裁剪更新 | 单独改善训练稳定性能带来多少收益？这是迁移版本，需说明实现差异 |
| C | OPSD + 最终正确性奖励 | 收益是否仅来自额外任务奖励？ |
| D | OPSD + 分支 rollout 评价与监督修正 | 在同等预算下，评价指导的实际效果是否有额外价值？ |
| E | 与 D 相同的额外生成预算，但用于普通 student rollout | 收益是否只是来自更多采样？ |

### 主要指标

- 最终答案准确率，以及相同计算预算下的学习曲线。
- 错误前缀后的纠正率：分别统计自然生成错误和人工构造错误，避免混在一起。
- Teacher 候选偏好与 student 实际续写成功率的一致性。
- 总生成 token、teacher 打分成本、训练计算量和实际耗时。
- 生成长度、无效反复纠错率，以及正确前缀被误改的比例。

### 关键消融

- Student 续写评价与 teacher 续写评价的差别。
- Teacher 只看最终答案与看到完整参考推理过程的差别。
- 只降低不可靠状态的训练权重，与改变具体动作偏好的差别。
- 随机选前缀与选择关键前缀的差别。
- 不同候选数、每个候选续写次数和分支预算。

### 公平性与风险

- 不只对齐训练步数；需报告总计算预算，分支采样不能忽略。
- 验证器衡量的是任务结果，未必证明所有中间推理步骤正确。
- 成功率估计随 student 更新而变化；陈旧评价需要更新或限制复用。
- 如果额外奖励、候选搜索或参考解法贡献了主要收益，应通过消融明确来源。

## 6. 潜在贡献与新颖性边界

值得聚焦的贡献是：

> 识别并改善“teacher 理解参考解法，但其局部指导不适合当前 student”的情形，尤其是错误前缀上的恢复决策。

以下表述本身不足以作为新颖性主张：给 OPD 加 critic、使用 advantage、采用 GAE、加入经验回放、让 teacher 看答案。需要进一步核对 OPPO、PACT、AggreVaTe、已有 teacher 可靠性和错误恢复工作，明确具体机制、成立条件及实证差异。

## 7. 参考文献

- [TOP-D: Trust Region Policy Distillation](https://arxiv.org/abs/2607.04751)
- [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
- [KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865)
- [OPPO: Bayesian Value Recursion for Token-Level Credit Assignment in LLM Reasoning](https://arxiv.org/abs/2605.21851)
- [PACT: From Credit Assignment to Critic Alignment](https://arxiv.org/abs/2609.26355)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Reinforcement and Imitation Learning via Interactive No-Regret Learning / AggreVaTe](https://arxiv.org/abs/1406.5979)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [IMPALA / V-trace](https://arxiv.org/abs/1802.01561)

关联笔记：[OPSD 理解：老师可以看答案](../OPSD理解-老师可以看答案.md)。
