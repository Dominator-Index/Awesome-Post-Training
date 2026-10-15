# AOPD 解读：When to Follow the Teacher: Advantage-Based On-Policy Distillation

> 来源：本目录 `paper.pdf`，19 页，匿名投稿，首页标注 ICLR 2027 Under Review。以下依据该版本正文及 Appendix A–B；没有把在审状态写成录用。公开论文链接和官方代码尚未识别。

## 1. 这篇文章到底在解决什么问题

**teacher 喜欢的动作，是否能提高当前 student 最终答对的概率？**

常规 OPD 在 student 生成的前缀上查询 teacher，解决了训练前缀与推理前缀不一致的问题。但它仍倾向于把 teacher 分布当成应当模仿的目标。teacher 可能推荐一条自己能完成、student 却无法完成的推理路线；student 原本可行的另一条路线，也可能因不符合 teacher 偏好而被压低。

AOPD 用 student 的 task advantage 定义监督价值：采取某个动作后，**后续仍由 student 完成**，预期最终奖励是否比 student 原策略更高？这和 teacher 自己能否解题是两个问题。

它同时适用于外部 teacher 和 OPSD 的 privileged teacher。即使 teacher 看到了参考解法，其推荐动作也可能超出 student 当前的后续能力。

## 2. Motivation：为什么需要这个方法

### 2.1 一个能说明问题的例子

这是对 Appendix B.2 两步反例的数值化解释。第一步选择简单路线或困难路线，第二步决定成功与否：

- student 选困难路线的概率为 0.1；在困难路线成功率 0.1，在简单路线成功率 0.9。
- teacher 选困难路线的概率为 0.9；在两条路线成功率都是 0.9。

student 总成功率是 $0.1\times0.1+0.9\times0.9=0.82$，teacher 为 $0.9$。teacher 的确更强。

但只把第一步改得更像 teacher，后续仍交给当前 student，就会让 student 更频繁进入自己不擅长的路线。困难路线对 student 的 advantage 为 $0.1-0.82=-0.72$；简单路线为 $0.9-0.82=0.08$。

因此，“teacher 整体更强”不足以保证“每次局部模仿都提高 student 奖励”。后续困难路线能力的改善，需要与第一步路线选择的改变相协调。

### 2.2 与错误前缀的关系

这篇并不要求先找出“哪个 token 在数学上错误”。它判断的是 token 对当前 student 后续任务成功的作用。一个形式上正确的 token 可能引出 student 无法完成的路线；一个不同于参考解法的 token 也可能对 student 有利。

所以它没有逐 token 正误标签，也不会自动输出错误定位报告。

## 3. 背景和已有方法的问题

| 方法 | 提供什么信号 | 本文针对的缺口 |
|---|---|---|
| SFT / 离线蒸馏 | 固定参考轨迹 | 推理时 student 会访问训练中没有的前缀 |
| 常规 OPD / OPSD | 学生前缀上的 teacher 分布 | teacher 偏好不必然提高当前 student 的任务奖励 |
| GRPO 等 RL | student rollout 的最终奖励 | 序列奖励稀疏；没有 teacher 分布来调制动作更新 |
| OPD + RL | 模仿项加奖励项 | 奖励项没有改变模仿项的监督方向，两者可能冲突 |
| 熵 / 兼容性筛选 | teacher 不确定性或分布接近程度 | 这些代理指标不是 student 后续成功的直接定义 |
| 反事实干预续写 | 替换动作后实际生成续写 | 能测局部作用，但增加 rollout 和奖励评估成本 |

这里是在说明这些方法相对本文目标的不足，不代表它们在所有任务上无效。实验中 AOPD 也没有赢过所有指标。

## 4. 方法总览

每个训练 iteration：

1. 固定当前 student 为 rollout policy $S$，待更新模型记为 $S_\theta$。
2. 每个 prompt 生成 $G$ 条独立 student 回答，计算最终奖励 $R_i$。
3. 用其他回答的平均奖励作为第 $i$ 条的 leave-one-out baseline。
4. teacher 与 student 在每条学生前缀上计算 token 概率。
5. 用采样 token 的 teacher/student 概率比和全词表 teacher 监督项构造损失。
6. 对两项施加一致的截断，只通过 $S_\theta$ 反向传播。

训练无需额外 teacher rollout、反事实续写、学习到的 value model 或过程奖励模型。但仍需要终局奖励函数、teacher 前向以及分布计算；相对完全不使用奖励的 OPD，奖励评估本身不是免费的。

## 5. 数学推导和关键公式

### 5.1 定义 student 后续能力

设历史 $h_t=(x,y_{<t})$，动作 $a$ 是下一 token，终局奖励在 $[0,1]$：

$$
Q^S(h,a)=\mathbb E_{\text{后续}\sim S(\cdot|ha)}[R],\qquad
V^S(h)=\mathbb E_{a\sim S(\cdot|h)}[Q^S(h,a)],
$$

$$
A^S(h,a)=Q^S(h,a)-V^S(h).
$$

$Q^S$ 是先采取 $a$、再让 student 完成后的成功价值；$V^S$ 是 student 原策略的平均价值。差值才是此动作相对 student 原策略的收益。它们都依据冻结的 $S$ 定义，不能替换为 teacher 完成回答的价值。

### 5.2 从 forward-KL 到 advantage 加权监督

论文从 FKL 出发。去掉与待更新参数无关的 teacher 熵，普通 FKL 最大化：

$$
J_{\mathrm{FKL}}(\theta)=\mathbb E_{x,y\sim S}
\left[\frac1H\sum_{t=0}^{H-1}\mathbb E_{a\sim T(\cdot|h_t)}\log S_\theta(a|h_t)\right].
$$

$T$ 是 teacher，$H$ 是论文的轨迹 horizon。这一目标给 teacher 动作正的模仿权重。AOPD 改为：

$$
J_{\mathrm{AOPD}}(\theta)=\mathbb E_{x,y\sim S}
\left[\frac1H\sum_t\mathbb E_{a\sim T(\cdot|h_t)}
A^S(h_t,a)\log S_\theta(a|h_t)\right].
$$

这是 Eq. 3.5。正 advantage 强化，负 advantage 抑制；它是带符号的加权 log-likelihood，不能把它简单称为始终非负的 KL 损失。历史分布、teacher 和 advantage 在这次更新中都固定。

### 5.3 如何不生成反事实续写？

对实际采样动作 $a_t\sim S(\cdot|h_t)$，定义：

$$
w_t=\frac{T(a_t|h_t)}{S(a_t|h_t)},\qquad \widetilde R=R-b.
$$

$b$ 是与当前回答独立的 baseline，实践使用其他回答的均值。概率比把 student 动作采样转换为 teacher 加权期望。Eq. 3.6 的估计器是：

$$
\widehat J_{\mathrm{AOPD}}(\theta)=\frac{\widetilde R}{H}\sum_t
\left[w_t\log S_\theta(a_t|h_t)
-\sum_aT(a|h_t)\log S_\theta(a|h_t)\right].
$$

关键是第二项。固定历史 $h$ 和 baseline $b$：

$$
\mathbb E[(R-b)w\log S_\theta(a|h)\mid h,b]
=\sum_aT(a|h)(Q^S(h,a)-b)\log S_\theta(a|h),
$$

$$
\mathbb E\left[(R-b)\sum_aT(a|h)\log S_\theta(a|h)\mid h,b\right]
=(V^S(h)-b)\sum_aT(a|h)\log S_\theta(a|h).
$$

第一项估计 $Q-b$，第二项估计 $V-b$，相减得到 $Q-V=A$，baseline 也消失。这就是 Lemma 3.1 和 Appendix B 的证明主线。

因此，它**没有先为每个位置准确计算一个 advantage 数值**。它直接给出整个 advantage 加权目标及梯度的无偏估计。单条回答的最终奖励仍然很粗；token 层面的不同由概率比和不同历史上的 teacher 分布体现，并通过多次采样在期望上恢复目标。

### 5.4 一次采样怎样影响更新？

将 $w_t$ 和 $T$ 视为固定，局部梯度为：

$$
\nabla\widehat J_t=\widetilde R
\left[w_t\nabla\log S_\theta(a_t|h_t)
-\sum_aT(a|h_t)\nabla\log S_\theta(a|h_t)\right].
$$

- $\widetilde R>0$：强化本次成功回答中的采样动作，teacher 越支持，该项越强；同时对 teacher 加权的其他选择施加相反方向。
- $\widetilde R<0$：压低本次失败动作，teacher 越支持这个失败动作，其抑制项越强；同时促进替代的 teacher 选择。
- $\widetilde R=0$：这一回答没有 AOPD 更新。

这是单次估计器的方向说明，不表示成功轨迹的每个 token 都有正的真实 advantage，或失败轨迹的每个 token 都有负的真实 advantage。

### 5.5 截断必须两边一致

为避免 $S(a|h)$ 很小时概率比爆炸，实验取 $c=4$：

$$
\widetilde w_t=\min(w_t,c),\qquad
\widetilde T(a|h)=\min\{T(a|h),cS(a|h)\}.
$$

实际目标为：

$$
\widetilde J(\theta)=\frac{\widetilde R}{H}\sum_t
\left[\widetilde w_t\log S_\theta(a_t|h_t)
-\sum_a\widetilde T(a|h_t)\log S_\theta(a|h_t)\right].
$$

这是 Eq. 3.8。$\widetilde T$ 不重新归一化，第一项和第二项使用相同的截断测度。于是它的期望仍为 $\widetilde T$ 加权的 advantage 目标。

**截断改变了原始目标。** 它不是未截断 Eq. 3.5 的无偏估计，而是截断后目标的无偏估计。只截断采样比、不改第二项，会破坏这套对应关系。

## 6. 为什么这个方法应该有效：理论机制

### 6.1 理想 policy-space proximal 更新

论文通过反向于命名的 proximal KL $D_{\mathrm{KL}}(S\|\pi)$ 控制一步变化：

$$
S^+\in\arg\max_\pi\left\{J(\pi)-\frac1\eta D_{\mathrm{KL}}(S\|\pi)\right\}.
$$

这里 KL 在冻结 student 访问的历史上平均；$\eta$ 越小，理论更新越保守。在能逐历史自由优化分布的条件下，FKL 的精确解为：

$$
S^+_{\mathrm{FKL}}(a|h)=\frac{S(a|h)+\eta T(a|h)}{1+\eta}.
$$

它是 student 与 teacher 的插值。其 student advantage 为 $\frac\eta{1+\eta}\mathbb E_T[A^S]$，可正可负。

AOPD 的精确解则为：

$$
S^+_{\mathrm{AOPD}}(a|h)=
\frac{S(a|h)+\eta T(a|h)A^S(h,a)}{1+\eta\mathbb E_T[A^S(h,a)]}.
$$

分子直接按动作 advantage 加减质量；步长足够小，所有系数才为正。这不是普通 teacher/student 凸插值，因为 $A^S$ 可以为负。

### 6.2 非负性来自平方 advantage

$$
\mathbb E_{a\sim S^+_{\mathrm{AOPD}}}[A^S(h,a)]
=\frac{\eta\mathbb E_{a\sim T}[(A^S(h,a))^2]}
{1+\eta\mathbb E_{a\sim T}[A^S(h,a)]}\ge0.
$$

分子是平方 advantage，分母在小步长条件下为正。这就是 Theorem 5.1 的局部改进机制。

再使用有限 horizon 的 performance-difference identity：

$$
\mathcal R(\pi)-\mathcal R(S)
=\mathbb E_{x,y\sim\pi}\sum_tA^S(h_t,a_t),
$$

每个可达历史上的非负局部 advantage 导出整体预期奖励不下降。严格改善还需要更新后实际访问到 teacher 支持且 advantage 非零的动作。

对截断权重，$|A^S|\le1$ 且 $\widetilde T\le cS$，取 $0<\eta<1/c$ 就能保证 proximal 系数为正；同样的平方项机制成立。

### 6.3 理论不能怎样解读

这是固定 rollout policy、teacher 和 advantage，具有足够支持、精确求解、足够小步长的 **policy-space** 结论。实际 LoRA/SGD、有限样本和学习率不直接满足该精确解。理论中的 $\eta$ 也不等于 Adam 的学习率。

因此不能把它写成“实际训练每一步奖励都必然上升”。其价值是说明理想更新的方向为何可以与奖励一致。

当 $S=T$ 时，常规 FKL 不再移动，而 AOPD 的期望梯度仍可依据 reward 产生策略改进。这是 teacher 不再构成固定模仿终点的原因；严格超过 teacher 仍需非零 advantage 等条件。

## 7. 为什么简单加 RL 不能保证解决问题？

Appendix B.2 对 $J_{\mathrm{mix}}=\mathcal R(\pi)/H+\lambda J_{\mathrm{FKL}}$ 的局部方向推导为：

$$
v(a|h)=S(a|h)A^S(h,a)+\lambda\big(T(a|h)-S(a|h)\big).
$$

第一项按 task advantage 更新，第二项继续无条件向 teacher 移动。对应一阶奖励变化包含：

$$
\eta\sum_t\mathbb E_{h_t\sim S}
\left[\mathbb E_S[(A^S)^2]+\lambda\mathbb E_T[A^S]\right]+O(\eta^2).
$$

RL 的平方项非负，但模仿项可能负。论文构造两步任务，证明对任意固定 $\lambda>0$，存在更强的 teacher，使混合目标的小步更新仍降低 student 奖励。

这是存在性反例，不是说 OPD+RL 一定失败。AOPD 改变的是 teacher 监督本身的权重，而非只给原监督加一个奖励目标。

## 8. 真正的贡献，以及与已有归档的比较

主要贡献是：以当前 student 的 counterfactual advantage 定义 teacher 监督价值，再推导一个从标准 rollout 和终局奖励计算的估计器。配套的对称截断和 proximal 分析解释了稳定性与理想更新方向。

| 论文 | 如何判断监督价值 | 与 AOPD 的差别 |
|---|---|---|
| Dr.OPD | 蒸馏梯度与奖励提升方向的对齐，近似双层优化权重 | AOPD 在动作层定义 $Q^S-V^S$，不依赖参数空间梯度对齐评分 |
| RLCSD | 正确/错误参考上下文的对比信号，调制奖励更新 | AOPD 用 teacher 动作概率及学生任务 advantage，未构造正负参考对比 |
| Teacher-Token-Reliability | 与分支可行性相关的监督可靠性及位置设计 | AOPD 不把 teacher 熵直接当作任务 advantage |
| TOP-D / trust-region 思路 | 控制分布变化或 teacher/student 插值 | AOPD 也借助 proximal 理论，但实际训练目标是带符号的 advantage 加权监督 |
| ReDraft | 修订学生回答，验证后做 SFT | AOPD 保留 student rollout，使用分布损失和最终奖励更新 |

以上按本地已归档论文方法比较，没有据此断言作者间的优先权。“first”属于作者主张，需要更全面的相关工作检索才能判断。

## 9. 实验、消融和证据链

### 9.1 设置

- student：Qwen3-1.7B、Qwen3-8B、Qwen2.5-7B-Instruct；Qwen3 均为 non-thinking。
- 数学：Open-Thoughts 训练，AIME24、AMC、HMMT 评估；奖励为答案正确与否。
- 代码：verified TACO 训练，HumanEval、MultiPL-E 评估；训练奖励由五个测试样例提供部分分数。
- 医学 QA：MedMCQA 训练，MedQA 测试。
- 外部 teacher：数学用 Qwen3-4B-Instruct，代码用 Qwen3-8B；自蒸馏使用带参考解法的 privileged student。
- 相同 iteration 数、batch size 32 和 LoRA 配置，但有方法专用超参数。

### 9.2 外部 teacher 结果：Table 1

| 指标 | OPD FKL | OPD RKL | AOPD |
|---|---:|---:|---:|
| Open-Thoughts pass@1 | 73.2 | 72.0 | 78.4 |
| AIME24 pass@1 | 32.1 | 32.6 | 35.0 |
| AMC pass@1 | 61.8 | 61.0 | 65.1 |
| HMMT pass@1 | 16.0 | 14.2 | 15.8 |
| HumanEval HE | 71.3 | 68.3 | 74.4 |
| HumanEval HE+ | 66.5 | 62.2 | 68.9 |
| MultiPL-E | 33.3 | 33.2 | 32.5 |

AOPD 在多数列有优势，但 HMMT 和 MultiPL-E 不是最佳；HumanEval pass@10 也由 GRPO 更高。不要把结论写成所有指标都胜出。

### 9.3 OPSD 场景：Table 2

| student / AIME24 | GRPO | OPSD | AOPD |
|---|---:|---:|---:|
| Qwen3-1.7B | 22.3 | 15.6 | 35.0 |
| Qwen3-8B | 30.6 | 41.7 | 52.1 |

1.7B 的 AOPD 比 OPSD 高 19.4 个百分点；8B 高 **10.4** 个百分点。正文把 $52.1-41.7$ 写成 13.4，有算术不一致；此处以表格数字的差值为准。

### 9.4 teacher 干预与消融

§4.3 对 512 条学生回答进行单 token 替换，再让 student 贪心续写：

| 替换位置 | 原失败变成功 | 不变 | 原成功变失败 |
|---|---:|---:|---:|
| 随机位置 | 6.9% | 85.1% | 8.0% |
| 最大 teacher/student KL 分歧 | 9.6% | 80.6% | 9.8% |

这支持 teacher 动作并非始终有益。注意这是额外的诊断实验；训练估计器本身不执行这些干预。它使用 teacher 最大概率 token 与贪心续写，不能直接当作完整随机策略下每个动作 advantage 的测量。

Appendix A 显示，group size 4 增至 8 收益有限；截断明显减少训练波动。前者在这些配置上支持较小组大小，不能普遍断言 G=4 总是足够。

### 9.5 比较公平性与复现细节

AOPD 禁用 top-p/top-k pruning，temperature=0.8；OPD/OPSD 使用 temperature=1.1、top-p=0.95、top-k=20。GRPO/KDRL 的外部 teacher 比较使用 G=8，而 OPD/AOPD 用 G=4。采样数和分布不完全一致，因此表格不构成完全等算力、同分布的单因素比较。

论文未在这份材料中清楚说明 temperature=0.8 后，重要性分母究竟使用实际采样分布还是未变温的模型分布。无偏推导要求 $S$ 与真实 rollout 分布一致；复现时必须检查，不能只禁用 top-k 就认为所有条件满足。

## 10. 局限和适用边界

1. **需要任务奖励。** 没有可靠答案校验或测试时，奖励噪声会直接污染 advantage 加权监督。
2. **同奖励组没有更新。** 使用 leave-one-out baseline，二值奖励下全错组与全对组都令 $R_i-b_i=0$。它没有自动解决全失败冷启动。
3. **依赖支持重叠。** $T(a|h)>0$ 必须蕴含 $S(a|h)>0$。极小学生概率还会造成高方差，截断缓解但改变目标。
4. **仍非显式 token 因果标签。** 单轨迹的序列奖励无法保证每个 token 当次更新都正确；无偏是多次采样的期望属性。
5. **全分布第二项有成本。** 论文公式对词表求和，仅取得 sampled-token teacher logprob 不足以直接实现原估计器。top-k 近似需另行分析。
6. **带符号目标需要保守更新。** 如果脱离 proximal 约束无限优化负权重 log-likelihood，概率可能被推向边界；理论保证不能直接套到任意 SGD。
7. **评估范围有限。** 主要是可奖励的短期文本任务，Qwen3 non-thinking，尚不足以覆盖长 thinking、复杂多轮 agent 或不可验证生成。
8. **实现资料不完整。** 当前匿名 PDF 没有识别出的官方代码链接，也不能从文件名 53047 推断 OpenReview forum id。

## 11. 总结：完整故事线

SFT 的问题是前缀来自参考轨迹；OPD 把前缀换成 student 自己生成的，但仍可能过度跟随 teacher。AOPD 进一步问：这个 teacher 动作能不能帮助当前 student 完成后续任务？

它把答案写成 student 的 $Q^S-V^S$，再用现有 rollout 的最终奖励、teacher/student 概率比和一项分布监督差，估计整个 advantage 加权目标。对称截断控制高方差；精确 proximal 理论用平方 advantage 解释理想更新的奖励非负性。

对我们关心的“teacher 是否会沿着错误前缀继续教”问题，它提供的是**结果导向的监督选择**：不预设 teacher 偏好总是好，也不把每个 token 当作已知正误。它仍依赖学生探索到有区分度的结果以及可靠的终局奖励。
