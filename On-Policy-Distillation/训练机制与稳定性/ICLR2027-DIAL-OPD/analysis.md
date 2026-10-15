# DIAL-OPD：Learning More from Fewer Tokens 解读

来源：[论文](https://arxiv.org/pdf/2610.11659)。按用户提供信息标记 **ICLR 2027**，不表述为已录用。用户提供的[代码地址](https://github.com/EIT-NLP/DIAL-OPD)在 2026-10-10 归档时网页与 GitHub API 均返回 404，因此目前没有可克隆的公开 checkout，也没有 main 链接；404 不能区分尚未公开、私有或地址改变。

**认知：分歧大不等于值得监督；两个模型都几乎不支持的 token，也能产生很大的 log-ratio。**

**方法：在相对分歧与绝对概率差之间调节选词分数，每条回答只保留 top 比例 token，然后对保留位置使用原始 OPD 更新。**

## 1. 这篇文章到底在解决什么问题

OPD 通常对回答的所有 token 给教学反馈。但 student 容量有限，全位置监督可能让无效分歧占用训练力度。直接按绝对 log-ratio 筛选也会选中 low-low token：student 与 teacher 都赋低概率，二者比值却很极端。

文章需要回答：**在固定监督数量下，如何挑选更有效的 token？** 它研究监督密度和选择规则，不是减少完整回答生成的 STOPD 方法。

## 2. Motivation：为什么需要这个方法

作者先随机减少监督位置，发现某些保留比例优于全 token 监督；再比较 Reward-Top 与随机选词，指出大分歧有用但不是充分条件。low-low 在这项诊断中定义为双方概率均不超过 1%；10% 保留时它们占选中 token 的 20.59%，却承担 49.80% 的绝对 reward。以等数量替换 low-low token 的实验进一步支持“监督分配比仅增密度重要”。

这还没有证明低概率 token 必然错误或没有学习价值；它是一个概率尺度诊断。

## 3. 背景和已有方法的问题

随机选择不识别有效分歧；按 teacher 概率选择可能错过需要压低的 student 高概率错误；按 log-ratio 选择看不到绝对概率。其他选择方法需要额外前向、参考模型或前缀规则，文章将其适配到相同 per-response 保留率后比较。适配版本不能自动代表各方法原始最优配置。

统一记号：$h_t=(x,y_{<t})$，$p_t=\pi_S(y_t\mid h_t)$ 是 student 对自己采样 token 的概率，$q_t=\pi_T(y_t\mid h_t)$ 是 teacher 在同一前缀上的概率。以下公式采用这个约定；PowerOPD 原文部分段落使用相反的 $p,q$ 命名。

即时 sampled-token OPD 的更新方向为

$$
g=\frac1Z\sum_t\operatorname{sg}[\log(q_t/p_t)]\nabla_\theta\log\pi_\theta(y_t\mid h_t).
$$

$\operatorname{sg}$ 表示系数不反传，$Z$ 是聚合分母。严格地说，这是冻结采样前缀分布时的条件 KL 更新。完整轨迹 KL 的精确梯度还包含早期动作改变未来前缀的项；不要将两个梯度无条件等同。

## 4. 方法总览

1. student 生成完整 rollout，teacher 为完整前缀序列评分。
2. 计算 sampled token 的 $p_t,q_t,r_t=\log(q_t/p_t)$。
3. 将 log-ratio 的幅度与绝对概率尺度组合成选择分数。
4. 在每条回答内保留 top-$K_i$ 个位置。
5. 原始有符号 log-ratio 仍用于更新，未保留 token 仍留在上下文中。

prompt、padding 和 terminal EOS 不计入训练候选位置。每步 fresh rollout 后重新计算 mask。

## 5. 数学推导和关键公式

### 5.1 logarithmic mean 包含插值

$$
L(p,q)=\int_0^1 p^{1-u}q^u\,du
=\frac{q-p}{\log q-\log p},\qquad L(p,p)=p.
$$

积分沿 log 概率的线性路径取平均；积分中的 $p^{1-u}q^u$ 在两概率之间。这里是某个 sampled token 的标量，不是归一化后的完整词表 teacher 分布。

### 5.2 两种分歧之间的几何插值

$$
\boxed{s_{\beta,t}=L(p_t,q_t)^\beta|r_t|
=|r_t|^{1-\beta}|q_t-p_t|^\beta,\qquad\beta\ge0.}
$$

$\beta=0$ 按相对 log 分歧选择；$\beta=1$ 按绝对概率差选择。$0<\beta<1$ 是这两种分数的几何插值，$\beta>1$ 是进一步加强概率权重的外推，不能继续叫 convex interpolation。$p=q$ 时分数取零，应避免负指数形式引起数值 $0/0$。

同一比例的分歧可以重要性不同：$(p,q)=(0.08,0.8)$ 与 $(10^{-4},10^{-3})$ 的 log-ratio 都为 $\log10$，而概率差是 0.72 与 0.0009，相差 800 倍。

### 5.3 top-K mask 与最终 loss

$$
K_i=\max\{1,\operatorname{round}(\rho|V_i|)\},\qquad
S_i=\operatorname{TopK}_{t\in V_i}(s_{\beta,i,t}),\qquad m_{i,t}=\mathbf1\{t\in S_i\}.
$$

$\rho$ 为监督保留比例。排序按幅度，正负 token 都可以选中。

$$
\boxed{\mathcal L_{\rm DIAL}=-\frac{\sum_{i,t}m_{i,t}\operatorname{sg}[r_{i,t}]\log\pi_\theta(y_{i,t}\mid h_{i,t})}{\sum_{i,t}m_{i,t}}.}
$$

mask 和 reward 在求导时固定；分母为整个 batch 的选中 token 数。**插值分数用于选择，未替换最终 reward。** 因此选中一个极端 log-ratio token 后，它的更新系数仍可能很大。

### 5.4 两条理论性质

$$
s_\beta(\epsilon p,\epsilon q)=\epsilon^\beta s_\beta(p,q),\qquad 0<\epsilon\le1.
$$

原始比值不变，但概率尺度下降后，分数按幂下降。这正是相对于 log-ratio 的新增信息。

$$
\lim_{b\to0^+}s_\beta(a,b)=
\begin{cases}
+\infty,&0\le\beta<1,\\
a,&\beta=1,\\
0,&\beta>1,
\end{cases}\qquad a>0.
$$

Appendix E 的证明来自 $(a-b)^\beta[\log(a/b)]^{1-\beta}$ 的渐近阶。$\beta>1$ 虽更抑制 low-low，也可能漏掉一个模型支持、另一个极低概率的有效纠错分歧。性质证明没有给出最终正确率保证。

## 6. 为什么这个方法 work

它把固定监督预算移出低绝对概率但高相对比值的位置，同时保留正向鼓励和负向纠错。适度概率加权能避免无条件相信最大的 log-ratio。过大 $\beta$ 又会压低单边低概率的重要分歧，因此最佳 $\beta$ 并非越大越好。

## 7. 为什么其他方法不行：它们忽略了什么

Reward-Top 忽略尺度；teacher-only ranking 忽略 student 已经确信的错误；简单 prefix 选择不能定位回答中间有价值的分歧。但没有任何实验表明所有高概率分歧都语义重要，也没有表明所有 low-low token 应该删除。

## 8. 这篇文章真正的 contribution

不是发现“插值”本身，而是将**相对和绝对分歧的调节用于固定预算下的监督位置选择**，并用概率平面诊断与替换实验解释何时会失效。RoadMap 主要改 W，伴随 N/L 的监督分母与 token 预算；H 的生成长度机制没有改变。

## 9. 实验、消融和证据链

- Qwen3-0.6B-Base/1.7B-Base student，Qwen3-4B/8B instruction teacher，四组合；DeepScaleR 训练 800 steps，batch 32，学习率 $5\times10^{-7}$，rollout cap 1024。
- 七个数学集：GSM8K、MATH500、AMC23、AIME24、AIME25、Minerva、Olympiad；保留 20% 或 40%，主表比较 $\beta=1,1.2$。
- 40% 监督时，对 Vanilla OPD 的七任务平均正确率增益最高 +5.25 个百分点；对最强 full-token variant 最高 +1.24 点（§6.2）。这是特定模型组合的最高增益。
- AIME25 Pass@16 的 13.33%→26.67% 来自 8B→1.7B、$\beta=1.2$ 的设置；不能等同平均正确率翻倍。
- 同 128 条回答、20% 保留的选词诊断中，$\beta:0\to1$ 将选中 token 的 low-low 占比从 8.26% 降至 0.29%；$\beta=2$ 更低但会遗漏单边支持的位置（§6.3）。
- Appendix C 明确区分固定前缀条件 KL 的 Monte Carlo 梯度与完整轨迹梯度，这是理解 OPD 的重要边界。

主实验只验证数学；本次没有复现。正文的容量解释需要谨慎：intro 声称同大小时随机稀疏监督优势消失，但 §3.1 报告 4B 的 30% 为 72.6%、full 为 71.2%，仍有小幅收益。现有结果不足以把现象全部归因于小模型容量。

## 10. 局限和适用边界

probability support 是代理，不是 token 对最终任务收益的因果贡献。错误前缀上的低概率正确纠错 token 可能被排除；高概率风格 token 仍可能入选。该方法确定性筛选且没有 IPW 补偿，改变了监督分配，不能声称仍无偏估计原始全 token OPD。

少监督 60% 不等于省下 60% 生成或 transformer 算力：完整 rollout 与评分仍需要完成。通用代码生成上的有效性没有在主结果中建立。

## 11. 总结：完整故事线

监督密度并非越高越好；log-ratio 又会误把低支持分歧当重点。DIAL 用概率尺度校准选择，把学习预算集中到部分分歧 token。它与 PowerOPD 共享“别只看比值”的认知，具体操作一个是筛选、一个是直接重写系数。

精确联系：$s_{\beta=1}=|q-p|=|r^{\rm Power}_{\alpha=1}|$，但 DIAL 的最终系数为 $m_t\log(q/p)$，不是 $q-p$。对照阅读：[OPD 信号变换对比](../../RoadMap/OPD信号变换对比.md)。
