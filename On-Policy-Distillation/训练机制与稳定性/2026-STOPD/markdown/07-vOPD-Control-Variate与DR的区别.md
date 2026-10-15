# 07｜vOPD：先区分三种随机性，再谈 control variate 与 DR

**认知：一次动作抽样、一次教师查询和一次 rollout 截断，产生的是不同的方差。减掉动作 baseline 与预测缺失反馈不是同一个操作。**

**方法：vOPD 将 sampled-token feedback 减去同一前缀下的期望反馈；DR 提供预测，再逆概率校正观测残差。**

论文：[KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865)。完整解读见 [analysis.md](../../NeurIPS2026-KL-Control-Variate/analysis.md)。

## 1. vOPD 原文的关键式

固定前缀 $c$，学生 $p$，老师 $q$，$a\sim p$，令 $r=\log q(a\mid c)-\log p(a\mid c)$，$v=\nabla\log p(a\mid c)$。

$$
b(c)=\mathbb E_{a\sim p}r=-D_{\rm KL}(p\|q),\qquad
\boxed{g^+_{\rm vOPD}=(r-b(c))v=(r+D_{\rm KL}(p\|q))v.}
$$

这里采用上升方向；实现的 loss 取负。由于 $\mathbb E_p v=0$，任何与当次动作无关的 state baseline 都不改变条件平均梯度。$r,b$ 作为系数 detach，保留 actor log probability 梯度。

## 2. 为什么不是 DR 的那个预测器

vOPD 的 baseline 必须在固定前缀下与抽样动作无关。它预测的是该前缀的平均反馈。DR 的 $m(X,a)$ 可以使用已生成动作，预测这个动作的 teacher feedback，再用随机查询校正。

$$
\boxed{g^+_{\rm query}=
\left[m(X,a)+\frac Qe(r-m(X,a))\right]v.}
$$

$e>0$ 是真实查询概率，先条件于已生成动作求查询期望，就得到 $rv$，不需要 $m$ 是动作无关的 baseline。

| 方法 | 校正哪种随机性 | 保持什么期望 |
|---|---|---|
| vOPD | 学生下一个动作抽样 | 固定前缀，对学生动作分布的平均局部梯度 |
| teacher-query DR | 缺失真实 teacher feedback | 固定完整 rollout，对随机查询的平均梯度 |
| STOPD IPW | suffix 被随机停止 | 固定潜在完整 rollout，对停止的梯度和期望 |
| sequential DR | 随机停止后的未来梯度 | 精确停止概率下，同上；预测准确时还可讨论整体轨迹期望 |

## 3. top-k 的区别

vOPD 在学生 top-k 集合上重归一化 $p,q$，计算 approximate KL baseline；集合由前缀分布决定。只要它仍不依赖当次抽样动作，近似 baseline 不改变条件动作平均梯度。

这不同于把原来的 distillation loss 截断成 top-k loss，后者一般改变目标。vOPD 仍对 sampled token 反向传播，并非 full-vocabulary loss 的全词表反向传播。

## 4. 方差保证的细节

最优 scalar baseline 是

$$
b^*(c)=\frac{\mathbb E_p[r\|v\|^2]}{\mathbb E_p\|v\|^2}.
$$

$\mathbb E_p r$ 只有在奖励与 score norm 的相关性较弱等条件下接近最优。论文 Appendix A.3 的方差减少是近似式，不是任意配置都更好的定理。减去有限常数也不能让原本无界的 $r$ 变成严格 bounded variable。

## 5. 和 STOPD 一起使用时，究竟保留哪个目标

$$
\widehat G^+_{\rm stop,vOPD}=\sum_t\frac{I_t}{p_t}(r_t-b(c_t))v_t.
$$

先对停止求期望，得到完整 **vOPD** rollout 的梯度和，并不等于该条固定 rollout 的 vanilla OPD 梯度和。再对学生动作求期望，baseline 才抵消。两层条件不能混淆。

## 6. 一个可继续研究的具体结合点

如果教师查询概率 $e(c,a)$ 随动作改变，考虑 $Q(r-b(c))v/e$。它仍保持原来的局部条件动作平均梯度；其最优动作无关 baseline 变成

$$
\boxed{b^*_{\rm query}(c)=
\frac{\mathbb E_p[r\|v\|^2/e(c,a)]}
{\mathbb E_p[\|v\|^2/e(c,a)]}.}
$$

当 $e$ 只依赖前缀时，它退化为通常最优 baseline。**非均匀缺失反馈改变了最优 baseline 的几何权重**，这是值得研究的 motivation；公式本身属于经典加权 control variate，不能据此宣称新算法。

主候选仍见 [06](06-候选研究问题-把教师预算用于预测残差.md)：先验证真实预算下的 residual correction，再决定是否增加 baseline。不要为凑方法而叠加组件。
