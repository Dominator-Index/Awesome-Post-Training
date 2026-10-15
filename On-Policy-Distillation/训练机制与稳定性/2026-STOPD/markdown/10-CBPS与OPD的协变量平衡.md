# 10｜CBPS：观察子集既要可建模，也要与目标总体在特征上平衡

**认知：一个 propensity classifier 预测 selection 很准，也不代表加权后的样本像目标总体。**

**方法：拟合选择概率时，同时约束加权协变量矩。**

依据：[Imai–Ratkovic，Covariate Balancing Propensity Score，2014](https://imai.fas.harvard.edu/research/files/CBPS.pdf) §2。以下 OPD calibration 是迁移讨论，必须和原始 CBPS 分开。

## 1. 原始 treatment 形式

设 $A\in\{0,1\}$，$e_\beta(X)=\Pr(A=1\mid X)$，$\phi(X)$ 是要平衡的特征。原文 Eq.8 的核心为

$$
\boxed{\mathbb E\left[\frac{A\phi(X)}{e_\beta(X)}-
\frac{(1-A)\phi(X)}{1-e_\beta(X)}\right]=0.}
$$

可以同时使用 likelihood score 和 balance moments，在 GMM / empirical likelihood 框架估计 $\beta$。CBPS 核心不是 arbitrary clipping，也不是后处理 loss 一定就叫 CBPS。

## 2. missing teacher feedback 的对应矩

对完整 student rollout，teacher 查询指示 $Q$，目标是完整集合的特征均值。missing-data 类比的 moment 为

$$
\mathbb E\left[\left(\frac Q{e(X)}-1\right)\phi(X)\right]=0.
$$

有限样本若选择再平衡权重，可要求

$$
\frac1n\sum_i Q_i w_i\phi(X_i)\approx\frac1n\sum_i\phi(X_i).
$$

例如数学/代码领域、相对位置、student uncertainty、低维 hidden-state projection，都可以作可见特征。这只说明选中集合在这些特征上类似总体，不证明结果变量相同。

## 3. 为什么不该直接替换 STOPD 的 $p_t$

STOPD 的 $p_t$ 来自已知随机设计，数学上正确。用一个 fitted CBPS probability 代替它可能降低某些样本波动，也可能引入原本不存在的概率误差。不能把“因果观察数据需要估计倾向得分”的动机原封不动搬来。

同时，未生成后段的 hidden states 不可见，不能计算完整总体的这些特征矩。位置、prompt-domain 这类预先可见特征有机会使用，hidden-state balancing 更适合完整 rollout 的 teacher query。

## 4. 一个可实施的 calibration 变体

完整 rollout 上，先按真正已知的 $e_i$ 随机查询。再以 $w_i^{\rm IPW}=1/e_i$ 为初值，求接近原权重的平衡权重：

$$
\min_w\sum_{i:Q_i=1}(w_i-1/e_i)^2
\quad\text{s.t.}\quad
\sum_{i:Q_i=1}w_i\phi(X_i)=\sum_{i=1}^n\phi(X_i),\quad w_i\ge0.
$$

这是借鉴 balance 的校准示意，不是原文完整 CBPS 方程；高维或低覆盖时 exact balance 可能不可行，需要近似约束。

因为 $w$ 是同批样本的函数，不能再用 $\mathbb E[Q_i/e_i]=1$ 的证明断言改权后仍对每个任意 $Y_i$ 精确无偏。

## 5. 何时平衡能控制教学偏差

若 outcome 大致在特征线性空间中，$Y_i=\beta^\top\phi(X_i)+\varepsilon_i$，则特征 imbalance 控制可解释部分的估计偏差，剩余取决于 residual。梯度是向量，同理需相关方向的特征；只平衡 entropy 或长度不能保证参数更新方向平衡。

与 DR/ARB 一起讨论时，可以让 prediction 抓主要模式，balance 控制未解释残差的选择偏移，见 [11](11-ARB近似残差平衡与OPD.md)。这是两类经典估计机制的联系，不应写成互不相关的模块拼接。

## 6. 什么 question 值得研究

**相同 teacher 查询预算下，位置/领域的样本平衡能否减小有限批次更新偏移，而不依赖精确预测每个 teacher token？**

它更适用于复杂历史 selector、未知选择概率、数据 replay/筛选后的样本迁移，或作为已知随机设计的模型辅助校准。必须分别报告平衡误差、权重极值、梯度偏差、wall time。

## 7. 结论

CBPS 有结合点，但在原始 STOPD 中并非最直接的替换件。它改的是 probability/weight estimation 和特征平衡；因果识别依然需要假设，梯度精确保持依然需要证明。平衡观察到的特征不能排除未观察结果导致的 selection bias。
