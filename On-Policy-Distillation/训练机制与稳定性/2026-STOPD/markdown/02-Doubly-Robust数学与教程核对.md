# 02｜Doubly Robust：先预测，再逆概率校正残差

**认知：如果有便宜的预测，不必把整个真实信号都乘大权重；可以只给预测误差乘大权重。**

**方法：$m(X)+I/e(X)\,[Y-m(X)]$。它包含 IPW：把 $m$ 设成零即可。**

依据：[用户给的 DR 教程](https://arxiv.org/html/2406.00853v1) §3 与 Appendix A，以及 [Bang–Robins 2005](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1541-0420.2005.00377.x)。知乎链接 `https://zhuanlan.zhihu.com/p/448876300` 多次读取失败，本地请求只得到 JS challenge，未获得正文；没有将其作为事实依据。下面重推数学，避免照搬教程中的简化表述。

## 1. STOPD 对应“缺失数据”版本

$Y$ 是想估计的完整信号，$X$ 是无论 $Y$ 是否被观察都可获取的特征，$I\in\{0,1\}$ 是观察指示。需要

$$
e(X)=\Pr(I=1\mid X)>0,\qquad
\mathbb E[Y\mid X,I=1]=\mathbb E[Y\mid X]=\mu(X).
$$

后一个条件是条件均值可忽略缺失。随机观察设计很容易满足；如果按未知真实 $Y$ 是否好来删样本，则未必满足。

若 $Y$ 为参数梯度向量，公式逐坐标成立，但预测与实现代价会不同。

## 2. 三种估计器

$$
\widehat\mu_{\rm IPW}=\frac1n\sum_i\frac{I_i}{\widehat e(X_i)}Y_i,
$$

$$
\widehat\mu_{\rm reg}=\frac1n\sum_i\widehat m(X_i),
$$

$$
\boxed{\widehat\mu_{\rm DR}=\frac1n\sum_i\left[\widehat m(X_i)+\frac{I_i}{\widehat e(X_i)}(Y_i-\widehat m(X_i))\right].}
$$

IPW 只依赖观察概率，regression 只依赖预测，DR 将预测与真实残差结合。未观察时也必须能计算 $\widehat m(X)$；这正是直接套进截断 rollout 的难点。

## 3. 双重稳健性的乘积误差

先把 nuisance 模型固定，或使用独立训练样本。条件期望为

$$
\mathbb E\left[\widehat m+\frac I{\widehat e}(Y-\widehat m)\mid X\right]
=\widehat m+\frac e{\widehat e}(\mu-\widehat m).
$$

减去真实 $\mu(X)$ 得到

$$
\boxed{\operatorname{Bias}(X)=\left(1-\frac{e(X)}{\widehat e(X)}\right)(\widehat m(X)-\mu(X)).}
$$

观察概率正确，预测可以错；预测均值正确，观察概率模型可以错。前提仍是可忽略缺失、正覆盖与适当的数据分离。实际拟合模型时通常说一致性或渐近性质，不能一概说任何有限样本都无偏。

这也解释 STOPD 中最可能的收益：原始保留概率是我们设计并精确知道的，第一项误差本来就是零。DR 此时主要提供**残差控制变量降低方差**，不是修复原本未知的 propensity。

## 4. 一个数字例子

真实信号 $Y=10$，以 $e=0.1$ 概率观察，预测 $m=9$：

| 情况 | IPW | DR |
|---|---|---|
| 未观察，概率 0.9 | 0 | 9 |
| 观察，概率 0.1 | 100 | $9+10(10-9)=19$ |
| 期望 | 10 | 10 |
| 方差 | 900 | 9 |

这是一个设定下的精确计算，不是 OPD 实验。若预测很差，DR 可以比 IPW 方差更大。

## 5. 方差公式告诉我们何时有收益

对于已知 $e$、条件可忽略缺失、固定 predictor，设 $\sigma^2(X)=\operatorname{Var}(Y\mid X)$。有

$$
\operatorname{Var}\!\left(m+\frac Ie(Y-m)\mid X\right)
=\frac{\sigma^2(X)}{e(X)}+\frac{1-e(X)}{e(X)}(\mu(X)-m(X))^2.
$$

IPW 是 $m=0$ 的特例。理想 $m=\mu$ 去掉第二项，但未消除真正不可预测的噪声。不能只因名称为 DR 就保证更高效。

## 6. treatment effect 版本与 OPD 的区别

因果 ATE 的标准 AIPW 为

$$
\widehat\tau=\frac1n\sum_i\left[
\widehat m_1(X_i)-\widehat m_0(X_i)
+\frac{A_i}{\widehat e(X_i)}(Y_i-\widehat m_1(X_i))
-\frac{1-A_i}{1-\widehat e(X_i)}(Y_i-\widehat m_0(X_i))\right].
$$

$A$ 是 treatment，$m_1,m_0$ 分别是两种干预下的均值。它需要额外的因果识别假设。STOPD 不比较“执行该 token”和“不执行该 token”的任务结果；它只恢复缺失的教学贡献，因此不必为了迁移 estimator 强行定义动作因果效果。

## 7. 对教程的核对

教程的 DR 组合思想适合作为入门，但有两个地方需要谨慎：

- §2.4 的 ATE IPW 式使用 treated/control 组样本数 $n_1,n_0$ 作分母。一般 population ATE 的 Horvitz–Thompson 形式应以总样本数 $n$ 平均；Hájek 形式则使用对应权重和。不能直接把组样本数版本移植。
- Appendix A 将残差项中的观察指示替换成其条件均值，需要可忽略性和正确 propensity 等条件。不能只凭 $\mathbb E[I\mid X]=e(X)$，对任意相关随机变量都这样替换。

本文件的乘积误差推导显式写出条件，因此也能看清 prediction-correct 分支与 propensity-correct 分支。教程的模拟数字不用于推断 OPD 的实验效果。

## 8. 为什么不是立刻把 STOPD 的 IPW 换成 DR

形式上想写

$$
\widehat G_{\rm naive}=\sum_{t\le T}\left[\widehat h_t+\frac{I_t}{p_t}(h_t-\widehat h_t)\right].
$$

但截断后 $T$、后段 token、后段 prefix 都未知，无法遍历并计算所有 $\widehat h_t$。只对已生成位置写这个式子也不对：baseline 部分被一同删掉，破坏完整补偿。

可执行的迁移必须先回答：**未观察单位的预测项能从哪里算出来？** 见 [03](03-DR用于OPD教师评分的实现.md) 与 [04](04-序列DR能否减少STOPD的rollout.md)。
