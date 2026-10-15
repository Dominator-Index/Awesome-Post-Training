# KL for a KL: On-Policy Distillation with Control Variate Baseline 解读

- 论文：[arXiv:2605.07865](https://arxiv.org/abs/2605.07865)，v1，2026-05-08。
- 方法：vOPD；[官方代码](https://github.com/holi-lab/vOPD)，本目录 `main` 指向 `/usr/data/vOPD`。
- 目录的 NeurIPS2026 标签沿用既有归档；本次没有独立核验录用状态。
- **认知：同一前缀下，teacher 与 student 的平均分歧不必成为每次抽样动作的噪声；监督可以围绕这个平均值居中。**
- **方法：sampled-token reverse-KL 梯度使用 $r+D_{\rm KL}(p\|q)$ 作为 detach 后的系数。**

## 1. 这篇文章到底在解决什么问题

sampled-token OPD 每次只抽一个动作，因此便宜，但 reward 和梯度可能有很大的动作采样方差。全词表 OPD 可消掉固定前缀的动作采样方差，但全词表反向计算更贵。论文试图以 baseline 保留 sampled-token backward 的效率，改善梯度稳定性。

它不解决 rollout 随机截断，不需要任务终局 reward，也没有提出因果意义上的 teacher usefulness 判别。

## 2. Motivation: 为什么需要这个方法

对高 mismatch token，$\log q-\log p$ 可以很负。一次 sample 的更新可能被它主导。老师和学生在这个前缀下整体分歧大时，单独这个动作的负反馈既包含总体偏移，也包含动作相对偏好。baseline 去掉总体偏移，减少抽样系数的噪声。

这不意味着语义上错误 token 都被识别，也不意味着 baseline 可以任意逐动作扣减。

## 3. 背景和已有方法的问题

令前缀 $c$，动作 $a$，学生 $p_\theta$，老师 $q$。讨论论文的局部 sampled-token surrogate，用上升方向统一符号：

$$
r(c,a)=\log q(a\mid c)-\log p_\theta(a\mid c),\quad
v(c,a)=\nabla_\theta\log p_\theta(a\mid c),\quad g^+_{\rm OPD}=rv.
$$

$r$ 是即时密集反馈，而非 trajectory return。局部 reverse KL 的负梯度满足

$$
-\nabla_\theta D_{\rm KL}(p_\theta\|q)=\mathbb E_{a\sim p_\theta}[rv].
$$

额外的常数项因 score 的期望为零抵消。完整 sequence-level KL 一般还涉及前缀分布的导数，不能把即时 token surrogate 自动说成完整轨迹目标的全部梯度。

## 4. 方法总览

1. 学生生成 rollout，teacher 对这些前缀给 logits。
2. 算 sampled-token $r$。
3. 从同一前缀的两个词表分布算 KL baseline。
4. $r$ 减 baseline，detach 系数，更新学生 sampled-token log probability。
5. top-k 版本只近似 baseline，减少计算开销。

官方 `src/opd_trainer.py` 静态阅读确认：student top-k indices、双分布重归一化、KL baseline、`reverse_kl_advantage = -centered_gap.detach()`。本次未运行该仓库训练。

## 5. 数学推导和关键公式

### 5.1 baseline

$$
b(c)=\mathbb E_{a\sim p}r(c,a)=-D_{\rm KL}(p(\cdot\mid c)\|q(\cdot\mid c)).
$$

$b$ 是当前前缀的平均即时 reward，没有 future return 或另一个 critic。论文 Eq.9–10 的更新为

$$
\boxed{g^+_{\rm vOPD}=(r-b(c))v=(r+D_{\rm KL}(p\|q))v.}
$$

对应可实现 surrogate：

$$
\mathcal L_{\rm vOPD}=-\sum_t\operatorname{sg}[r_t-b(c_t)]\log p_\theta(a_t\mid c_t).
$$

系数 stop-gradient；若把 KL baseline 当成可微 loss 的另一项求导，得到的是另一个方法。

### 5.2 为什么平均更新不变

$$
\mathbb E_p[b(c)v]=b(c)\sum_a\nabla p(a\mid c)=0.
$$

任意 action-independent state baseline 都成立，包括近似值。它保持条件动作均值，不保持每一条固定抽样轨迹的更新。

### 5.3 top-k baseline

取由学生分布确定的集合 $\mathcal K(c)$，分别归一化为 $\bar p,\bar q$：

$$
\widehat b(c)=-\sum_{a\in\mathcal K(c)}\bar p(a\mid c)\log\frac{\bar p(a\mid c)}{\bar q(a\mid c)}.
$$

这是 baseline 的近似，不是 distillation target 的 top-k 截断；条件于前缀时集合不随当前动作 sample 改变，因此均值性质仍在。论文常用 $k=20$。

### 5.4 精确最优值和近似方差式

令 $Z=\|v\|^2$。最优 scalar baseline：

$$
b^*=\frac{\mathbb E[rZ]}{\mathbb E Z}.
$$

它由最小化 $\mathbb E[(r-b)^2Z]$ 得出。任意 baseline 的方差减少为

$$
\operatorname{tr Var}(rv)-\operatorname{tr Var}((r-b)v)
=2b\mathbb E[rZ]-b^2\mathbb E Z.
$$

当 $\mathbb E[rZ]\approx\mathbb E r\,\mathbb E Z$，选择 $b=\mathbb E r$ 得论文 Eq.11 / Appendix A.3 的近似减少量 $D_{\rm KL}(p\|q)^2\mathbb E Z$。该条件不能省略，baseline 并非无条件最优。

## 6. 为什么这个方法 work

动作 score 本身有零均值，因此可加入一个零均值控制项而保持局部平均方向。若 baseline 与最优值接近，sampled reward 中的共同偏移被抵消，二阶矩更小。它不需要训练一个 value network，但需要 teacher 的分布信息；黑盒仅返回 sampled-token log probability 时无法直接计算 full KL。

## 7. 为什么其他方法不行：它们忽略了什么

vanilla sampled-token OPD 没利用零均值 score 控制项；full-vocabulary OPD 更直接地平均动作，但 backward 成本较高；top-k loss 为降低成本改变目标，而 top-k baseline 可保持条件动作平均梯度。

这些不是其他方法普遍无效的结论。full-vocabulary 方法在部分实验仍很好，baseline 预测不好也可能增加方差。

## 8. 这篇文章真正的 contribution

把当前前缀的 negative reverse KL 识别为无需额外训练的即时 baseline，并提供 top-k 近似实现。主要创新点在已有 control variate 思路的低成本 OPD 实例化，不是新 DR 定理。

与 STOPD/DR 的区别见 [统计方法笔记 07](../2026-STOPD/markdown/07-vOPD-Control-Variate与DR的区别.md)：动作采样、query missingness、随机停止必须分别分析。

## 9. 实验、消融和证据链

数学包括 Qwen3-1.7B→1.7B-Base、Qwen3-4B→4B-Base、Qwen3-1.7B→0.6B-Base，以及 Olmo-3-7B-Think→7B-Base。主数学训练使用 DAPO-Math17K 子集约 14K；另有 SciKnowEval 化学与 GPQA Diamond。

Table 2：Qwen3-1.7B 的表内平均指标 OPD 29.8、full-vocabulary OPD 33.0、vOPD top-k 33.0、vOPD full baseline 33.1；MATH500 Avg@8 为 58.7→64.9。平均指标提升支持实用性，不能简化为所有任务提升同样幅度。

LoRA rank 64 / alpha 128，batch 64，训练 response 上限 2048；Qwen 学习率 $10^{-5}$。训练 sampling temperature 1、top-p 1，评估 temperature 0.6 / top-p 0.9。top-k baseline 大幅降低相对 full-vocabulary backward 的成本；论文报告最大 57.7% wall-time 节省，依赖具体实现和对照。

消融考察 baseline、k、梯度与 reward 分布。它们支持局部降方差机制，尚未证明 code generation、超大模型或所有 teacher/student mismatch 都有效。

## 10. 局限和适用边界

- 原文说 advantage 在 heavy-tail token 上保持 bounded；加一个有限常数不构成严格有界性证明。应理解为经验上的 damping，不是数学保证。
- $b=\mathbb E r$ 不是任意参数空间的最优 baseline。
- $a\sim p$ 是 score 零均值所用分布；若 rollout 用 temperature/top-p 后的另一个 behavior 分布，直接套用需重新检查或校正。
- PPO 多步 reuse 和 clipping 属非线性目标，不能自动继承以上无偏结论。
- top-k baseline 仍要计算 teacher 前缀；少取 logits 不等于同幅度少 FLOPs。
- baseline 可以改变单 token 系数符号。它不表示 teacher 在语义上认可错误答案。

## 11. 总结：完整故事线

**sampled-token OPD 用随机动作近似局部 KL 更新，噪声大；score 的零均值允许扣掉当前前缀平均 feedback；negative KL 提供可计算 baseline；top-k 近似保持这个平均性质，以较低成本改善更新。**

最简关键式：

$$
\boxed{r\nabla\log p\quad\longrightarrow\quad(r+D_{\rm KL}(p\|q))\nabla\log p.}
$$

代码以官方源码快照取得：git HTTP 拉取超时，使用 GitHub archive；来源与哈希见 `main/CODE_SOURCE.json`，并非完整 git checkout。
