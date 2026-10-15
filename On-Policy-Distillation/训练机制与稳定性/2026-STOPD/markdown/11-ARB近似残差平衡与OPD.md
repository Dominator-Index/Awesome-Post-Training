# 11｜Approximate Residual Balancing：回归解释主要信号，平衡权重校正剩余误差

**认知：辅助模型不必完美预测所有维度；只要回归误差和特征不平衡的乘积足够小，目标均值误差就可能受控。**

**方法：先做 regularized regression，再求稳定的近似平衡权重，对 observed residual 加权。**

依据：[Athey–Imbens–Wager，Approximate Residual Balancing](https://arxiv.org/abs/1604.07125)，发表于 JRSS B 2018。原始结果是高维线性 outcome 结构下的 treatment-effect inference，不是无条件、任意神经网络下的 DR 保证。

## 1. 原文核心式

原文 Eq.4 对 control potential outcome 均值估计为

$$
\boxed{\widehat\mu_c=\bar X_t^\top\widehat\beta_c+
\sum_{i:W_i=0}\gamma_i(Y_i-X_i^\top\widehat\beta_c).}
$$

$\bar X_t$ 是 treated group 特征均值，$\widehat\beta_c$ 是 control outcome 的回归参数，$\gamma$ 将 control 特征向目标 treated group 配平。这与“prediction + weighted residual”有共同结构，但不要求权重逐项等于拟合 propensity 的倒数。

## 2. 权重设计

原文优化平衡与方差的权衡。用概括形式写为

$$
\min_\gamma\;\zeta\|\gamma\|_2^2+(1-\zeta)\|\bar X_t-X_c^\top\gamma\|_\infty^2,
\quad \sum_i\gamma_i=1,\quad 0\le\gamma_i\le n_c^{-2/3}.
$$

具体约束与适用样本数见原文 Algorithm 1；此式保留其核心结构。$\ell_2$ 权重项控制噪声，$\ell_\infty$ imbalance 控制高维最坏坐标偏移。不是只找能分类 treatment 的 logistic regression。

## 3. 最重要的误差分解

若真实 conditional mean 是线性模型 $X^\top\beta_c$，$\varepsilon_i=Y_i-X_i^\top\beta_c$，原文给出

$$
\widehat\mu_c-\mu_c=
(\bar X_t-X_c^\top\gamma)^\top(\widehat\beta_c-\beta_c)+\gamma^\top\varepsilon.
$$

于是

$$
\boxed{|\widehat\mu_c-\mu_c|\le
\|\bar X_t-X_c^\top\gamma\|_\infty\|\widehat\beta_c-\beta_c\|_1
+|\gamma^\top\varepsilon|.}
$$

平衡误差与回归误差相乘，外加加权噪声。它解释为什么模型可以不完全正确，但需要足够好的回归、overlap 和高维 regularity；不能简化成“任意回归或任意权重一个正确就行”。

## 4. OPD teacher-query 迁移

完整 student rollout 的每个 query unit 有可见特征 $\phi_u$，真实 target $Y_u$ 可先取一个梯度坐标/方向投影，而不是直接假定 scalar reward 就代表梯度。target mean 是所有 query units 的均值。

先拟合 $\widehat m(\phi)=\phi^\top\widehat\beta$，再求 queried subset 的 balance weights：

$$
\boxed{\widehat\mu_{\rm OPD,ARB}
=\bar\phi_{\rm all}^\top\widehat\beta+
\sum_{u:Q_u=1}\gamma_u(Y_u-\phi_u^\top\widehat\beta).}
$$

这不是新的已验证 OPD 算法，而是明确 estimand 后的模型辅助构造。缺失真实 teacher feedback，但所有 student features 已可见，因而总体特征均值可计算。

## 5. 实现选择

1. query 单位优先选 response 或可缓存的 chunk，使用低维 features，避免巨大的 QP。
2. 特征可以包含领域、相对位置、student hidden projection 和梯度方向相关投影。
3. 回归用 ridge/lasso 等小模型，按 prompt/rollout 分折或使用过去数据。
4. 在 queried units 上解近似 balancing weights，约束最大权重，记录 imbalance。
5. 先检查 heldout 梯度投影误差；reward MSE 低不证明梯度准确。

如果直接以 $r_u$ 作 outcome，再乘 $v_u$ 更新，要把 $r_uv_u$ 的结构也纳入分析。对 reward 的 scalar mean 平衡，不能自动成为 actor vector gradient 的误差保证。

## 6. 与 STOPD 的难点

未生成 suffix 的 hidden features 不存在，$\bar\phi_{\rm all}$ 不可计算；因此上述完整 rollout ARB 不能直接节省 student generation。若只用 prompt/position 预先可见 features，则可以建一个弱模型辅助估计，但需要判断这些 features 能否解释梯度，而非仅解释长度。

原始 STOPD 已有精确设计概率。ARB 可能改善有限样本方差，但放弃 arbitrary outcome 的设计无偏，转为 outcome 结构与近似 balance 支撑的保证；这是不同统计取舍。

## 7. 好 question 是什么

**低预算 teacher 查询时，能否不拟合复杂 propensity，也不训练强大的 proxy，而依靠少量梯度相关特征的近似残差平衡，获得稳定更新？**

它有明确 motivation：LLM hidden representations 高维，直接预测 teacher 每个 token 很难；query 子集又可能偏向前段、某领域或高 uncertainty。ARB 给了可控制的 bias–variance 分解。

必须验证 outcome 近似线性/稀疏结构是否合理，低覆盖时 QP 是否可行，以及平衡特征与 task success 的联系。否则只是经典统计方法换了应用名称。
