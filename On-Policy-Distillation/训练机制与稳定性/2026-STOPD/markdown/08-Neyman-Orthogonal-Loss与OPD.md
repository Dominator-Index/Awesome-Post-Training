# 08｜Neyman Orthogonal Loss：让辅助模型误差不在一阶污染 OPD

**认知：辅助反馈模型难免错；可以设计一个更新方程，让小的辅助模型误差只产生二阶影响。**

**方法：把预测项与逆概率残差校正写成正交 score，并在独立数据上学习 nuisance。**

主要依据：[Chernozhukov 等 Double/Debiased ML](https://arxiv.org/abs/1608.00060)、[Foster–Syrgkanis Orthogonal Statistical Learning](https://arxiv.org/abs/1901.09036)。下面的 OPD score 是本笔记的迁移推导，不是这些论文验证过的 LLM 算法。

## 1. 什么叫 orthogonal

令 $\beta$ 为目标，$\eta$ 为辅助模型（nuisance），估计方程 $\mathbb E[\psi(Z;\beta,\eta)]=0$。Neyman orthogonality 要求在真实参数处

$$
\boxed{D_\eta\mathbb E[\psi(Z;\beta_0,\eta_0)][\delta\eta]=0.}
$$

$D_\eta$ 是对辅助函数的一阶方向导数。它不是要求神经网络两个向量夹角 90 度，也不是把 loss projection 到另一个梯度上。

对于 loss $L(\beta,\eta)$，对应条件是目标梯度对 nuisance 的一阶扰动消失，即 $D_\eta\nabla_\beta L(\beta_0,\eta_0)=0$。只称 loss “orthogonal”没有意义，必须写出哪个目标梯度。

## 2. AIPW 本身给出了一个正交例子

缺失均值问题的 score 为

$$
\psi=m(X)+\frac I{e(X)}(Y-m(X))-\mu.
$$

对 $m$ 作扰动 $\delta m$，在真实 $e_0$ 下：

$$
D_m\mathbb E[\psi][\delta m]
=\mathbb E[(1-I/e_0(X))\delta m(X)]=0.
$$

对 $e$ 作扰动 $\delta e$，在真实 $m_0=\mathbb E[Y\mid X]$ 下：

$$
D_e\mathbb E[\psi][\delta e]
=-\mathbb E\left[\frac I{e_0(X)^2}(Y-m_0(X))\delta e(X)\right]=0.
$$

这两个等式需要条件可忽略缺失与正覆盖。[02](02-Doubly-Robust数学与教程核对.md) 的 bias 是 nuisance 误差乘积，与一阶导数消失一致。但“一阶正交”与“任一个模型正确都一致”的 DR 性质，在一般模型中不是同义词。

## 3. OPD 中一个具体的目标和 score

在固定 rollout 分布下，令 $v_t=\nabla\log\pi_{\theta_k}(y_t\mid c_t)$，目标为

$$
g^*=\mathbb E[r_t v_t].
$$

把 $X_t$ 取为包含可见 state/action、足以确定 $v_t$ 的特征。$r_t$ 为 teacher feedback，查询指示为 $Q_t$。可写向量 score

$$
\boxed{\psi_t(g;b,e)=\left[b(X_t)+\frac{Q_t}{e(X_t)}(r_t-b(X_t))\right]v_t-g.}
$$

$g$ 是要估计的局部更新向量，不把它和 nuisance 参数混淆。如果是有限批次 token 平均，则先固定批次与 $N$，再讨论 query 随机性；随机长度聚合需另行处理。

对每个参数坐标，这个 score 都有相应正交条件。它自然诱导 [03](03-DR用于OPD教师评分的实现.md) 的 loss，而不是往已有 loss 加一项“orthogonal penalty”。

## 4. STOPD 中能省掉什么、不能省掉什么

STOPD 的 $p_t$ 已知精确，因此无需学习 propensity。teacher-query 设计中 $e_t$ 若也由我们随机化，就同样已知。此时 predictor 错误对**查询期望**已经精确抵消，不只是小误差二阶近似。

当观察机制未知、依赖未记录的在线选择或概率估计有误时，正交和 cross-fitting 更有意义。若真正没有正覆盖，或缺失依赖未观察结果，正交不能创造识别性。

## 5. DML/cross-fitting 的实现注意事项

- 在训练折学习 feedback predictor，在评估折生成 pseudo-outcome / score，再交换折；或在线使用过去 batch 的 lagged predictor。
- 同一回答的 tokens 强相关，分折应按 prompt/完整 rollout，而不是随机把一条轨迹拆到训练与评估。
- predictor 输入 detach，actor 更新系数 detach；不要把 nuisance 模型也一起通过 actor loss 优化，破坏其角色。
- student 更新后轨迹分布改变，经典 IID DML 的 $\sqrt n$ 推断不能直接当长期 LLM 训练保证。
- 一阶正交不保证没有 IPW 的大权重；还要检查 propensity 正覆盖与高阶矩。

## 6. 与“什么应该学”的关系

它解决**估计误差对更新的污染**。如果 target 定义为 teacher log-ratio，它就稳健估计 teacher 教学；若要学最终 task value，需要重新定义 $Y$ 或 $g^*$。不能把 orthogonality 当作自动识别语义重要性的方法。

## 7. 可研究 question

**在线 OPD 的廉价反馈模型在持续分布漂移时，正交 residual update 能否比直接 proxy distillation 更稳定？**

需要比较相同 teacher 查询预算下的 proxy-only、IPW、DR，再故意制造 predictor misspecification 与分布漂移。若概率已知、所有版本已经精确校正均值，创新重点应在方差、可实施性与在线表现，而非重复宣称 double robustness。
