# 09｜TMLE：针对目标校准预测模型，而不是直接换 IPW 权重

**认知：整体预测准确，不代表我们关心的目标均值准确；可以沿目标相关方向校准预测。**

**方法：拟合 initial outcome model，用 efficient influence function 构造 fluctuation，校准后再 plug in 目标。**

依据：[van der Laan–Rubin 2006 原始 TMLE](https://www.stat.cmu.edu/~ryantibs/journalclub/vanderlaan_2006.pdf)。本文先给 missing-outcome mean 的标准特化，再讨论 OPD；不是在 LLM 上已验证的算法。

## 1. 一个可写清楚的 estimand

$Y\in[0,1]$，$I$ 表示可见，所有样本的 $X$ 已知，$e(X)=\Pr(I=1\mid X)>0$。目标是

$$
\mu=\mathbb E[Y]=\mathbb E[m_0(X)].
$$

对应 influence function 为

$$
D(O)=\frac I{e(X)}(Y-m_0(X))+m_0(X)-\mu.
$$

它与 AIPW score 有相同结构，因此 TMLE 与 DR 有联系。一般说的是正则条件下渐近性质，不是任意有限批次的精确设计无偏。

## 2. binary outcome 的 targeting 步骤

先得到 $\widehat m(X)\in(0,1)$，令 clever covariate $H(X)=1/\widehat e(X)$。构造

$$
\operatorname{logit}\widehat m_\epsilon(X)
=\operatorname{logit}\widehat m(X)+\epsilon H(X).
$$

用已观察的 $Y$ 拟合这个单参数 logistic fluctuation，initial logit 为 offset。score 方程为

$$
\sum_i\frac{I_i}{\widehat e(X_i)}[Y_i-\widehat m_{\widehat\epsilon}(X_i)]=0.
$$

然后

$$
\boxed{\widehat\mu_{\rm TMLE}=\frac1n\sum_i\widehat m_{\widehat\epsilon}(X_i).}
$$

targeting 让 AIPW residual 的经验均值为零；估计值来自校准后的预测平均。输出仍在 $[0,1]$。TMLE 并非所有场景必须用这个 logistic 版本，这里选它便于解释有界结果。

## 3. OPD 最自然的第一处：截断后的存活长度

STOPD 完整 token 数是 $N=\sum_iT_i$。写成最大位置范围上的 alive indicators：

$$
A_{i,t}=\mathbf1\{T_i\ge t\},\qquad
N=\sum_{i=1}^{B}\sum_{t=1}^{L}A_{i,t}.
$$

这里 $A$ 不是 causal treatment，只是自然长度存活标签。$X=(x_i,t)$ 可在未生成后段时也计算；predictor 预测位置存活概率。

取原始 stop 指示 $I_{i,t}=\mathbf1\{S_i\ge t\}$、观察概率 $p_t$。当 $t\le S_i$，alive label 可知：生成到该 token 则为 1；若先遇到 EOS，之后对应位置为 0。因此可以对 observed alive labels 做 targeting，再将预测存活概率加总。

它有机会提供受边界约束的长度估计，用于诊断/下一批 normalizer。但是：

- 同一 rollout 的位置相关，需要按 rollout 建模与统计分析，不能当 $BL$ 个 IID 样本。
- 当批 targeting 用了当前 stop，若立即拿该估计作当前 loss 分母，就回到随机分母 ratio bias。
- 只能改善 denominator 的候选质量，不能直接消除 STOPD 的尺度误差；可考虑预测下一步 $C_k$ 并单独比较。
- 不必非用 TMLE：AIPW、survival regression、lagged count 都是应比较的更简单 baseline。

## 4. teacher feedback 与成功率也能讨论，但目标要明确

已生成完整 rollout 时，可对部分可见的 bounded verifier outcome 或评分的均值做 TMLE。teacher log-ratio 和梯度是无界/向量，不能未经定义就塞进上述 Bernoulli likelihood。

连续 outcome 有对应 fluctuation，但需要合适的 outcome model、目标函数与矩条件。将多个梯度坐标分别 targeted 往往不便宜；targeted loss value 也不自动意味着其 actor 梯度被正确校准。

## 5. implementation question

对于低预算 OPD，**比训练普通 predictor 更针对 normalizer/gradient 的校准是否能提高稳定性？**

实现先从 scalar length 或 bounded quality mean 入手，而不是重写整个 transformer 的 maximum likelihood。需要记录 targeting 的数据消耗、拟合误差与方差；当前 target fitting 与当前更新的依赖不能忽略。

## 6. 可行性判断

- 数学连接明确：DR efficient score 与 targeted plug-in estimator。
- 对 STOPD denominator 的存活预测是可具体讨论的入口。
- 对完整 actor gradient 的直接迁移还不够，不能声称 TMLE 天然获得和原文 IPW 一样的 finite-batch guarantee。
- 它优化目标估计，不替代 task reward，也不会自行判断 teacher 的知识是不是正确。
