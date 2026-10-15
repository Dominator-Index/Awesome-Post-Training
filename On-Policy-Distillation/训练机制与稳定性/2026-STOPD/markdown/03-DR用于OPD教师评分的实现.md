# 03｜DR 在 OPD 中较直接的实现：生成完整回答，只抽样昂贵 teacher 评分

**认知：廉价预测提供常规教学，真实 teacher 负责校正预测残差。**

**方法：对所有已生成 token 使用 proxy 系数；对随机查询位置加 $1/e_t$ 倍的 teacher–proxy 差。**

这是本笔记依据 AIPW 构造的候选迁移，不是 STOPD 原文算法，也未宣称新颖性或完成实验。基础理论见 [02](02-Doubly-Robust数学与教程核对.md) 与 [Bang–Robins](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1541-0420.2005.00377.x)。

## 1. 为什么先选择 teacher feedback 缺失

student 已生成完整 $y$，所有前缀、student log-prob 和 score gradient 都可见。缺的是昂贵 teacher 的评分，所以可以为每个未查询位置计算便宜 proxy。这满足静态 AIPW 对可见 $X$ 的要求。

它不减少 student rollout，不能称为 STOPD 的完整加速版。先从这个子问题验证估计器，可以避免同时引入未生成前缀与预测未来两种困难。

## 2. 公式

定义 ascent 方向

$$
r_t=\log q_T(y_t\mid c_t)-\log\pi_S(y_t\mid c_t),\qquad
v_t=\nabla_\theta\log\pi_\theta(y_t\mid c_t).
$$

廉价预测 $b_t=b_\psi(X_t)$ 近似 $r_t$；$X_t$ 可以包含前缀、sampled token 与 detached student 特征。这里 **允许 $b_t$ 依赖动作**，因为它是预测真实反馈，不是 vOPD 的零均值 state baseline。

以已知 $e_t>0$ 随机查询，$Q_t\sim\operatorname{Bernoulli}(e_t)$，查询随机性在 $X$ 给定后不依赖未知真实 teacher 随机反馈。定义

$$
\boxed{\widetilde r_t=b_t+\frac{Q_t}{e_t}(r_t-b_t),\qquad
\widehat g^+_{\rm DR}=\frac1N\sum_t\operatorname{sg}[\widetilde r_t]v_t.}
$$

完整 $N$ 可见，无需 STOPD 的 lagged normalizer。对固定完整 trajectory 与预先固定的 predictor，有

$$
\mathbb E_Q[\widetilde r_t]=r_t,\qquad
\boxed{\mathbb E_Q[\widehat g^+_{\rm DR}]=N^{-1}\sum_t r_t v_t.}
$$

无论 predictor 是否正确，都保持该批次局部梯度的期望；predicted coefficient 与 observed coefficient 都 detach。

## 3. surrogate loss

$$
\mathcal L_{\rm DR}(\theta)=-\frac1N\sum_t\operatorname{sg}\left[b_t+\frac{Q_t}{e_t}(r_t-b_t)\right]\log\pi_\theta(y_t\mid c_t).
$$

对未查询位置，代码只需 $b_t$，不能先计算全部真实 $r_t$ 再 mask。形式等价于“dense proxy 更新 + sparse teacher residual 校正”，没有重新定义 student 的目标老师。

### 伪代码

```python
# Fixed student batch; teacher is queried only after the random mask is drawn.
student_logp, features = student_forward(full_rollouts)
with no_grad():
    proxy = frozen_predictor(features, sampled_tokens)
    query_prob = positive_query_prob(features).detach()
    queried = draw_bernoulli(query_prob)
    coefficient = proxy.clone()
    true_reward = teacher_score_selected(queried) - student_logp[queried]
    coefficient[queried] += (true_reward - proxy[queried]) / query_prob[queried]
loss = -(coefficient.detach() * student_logp)[response_mask].mean()
loss.backward()
```

这是接口级伪代码，不声称通用 transformer 支持以零开销随机查询单 token。prompt/EOS/padding 规则与 baseline 保持一致。

## 4. 为什么可以降低查询方差

若在固定 trajectory 上，各 $Q_t$ 独立，查询方差的 trace 为

$$
\operatorname{trVar}_Q(\widehat g^+_{\rm DR})
=\frac1{N^2}\sum_t\frac{1-e_t}{e_t}(r_t-b_t)^2\|v_t\|^2.
$$

IPW 对应把残差 $r_t-b_t$ 换成 $r_t$。预测好时只有小残差被放大。比较的是梯度误差，不是只看 reward prediction MSE：不同 token 的 $\|v_t\|^2$ 会改变贡献。

固定数量或 block 级查询会引入位置协方差；上式不能原封不动套用。

## 5. 实现成本决定它值不值得做

- student 完整 generation 和更新仍发生；稀疏 teacher 查询不等于稀疏 student backward。
- 自回归 teacher 要到达一个后段 token，仍需处理它的全部前缀。随机取 10% 位置通常不会省掉 90% transformer FLOPs。
- 可先以整条 response 为查询单位：查询一条时拿到所有 teacher log-prob，不查时所有位置使用 predictor。给该 response 同一 $Q_i/e_i$；正确估计仍成立，query 方差应按 response 梯度残差计算。
- block 查询需要 KV cache、前缀处理与输出投影成本分析；API 计费、teacher-input tokens、实际 wall time 应分别记录。
- predictor 用过去 batch 训练，当前查询前冻结；当前 teacher 标签用于下一轮。若先用同一批标签拟合再评估同批 AIPW，finite-sample 设计无偏性不自动成立。
- $e$ 是实现中的随机机制，不需要再拟合 propensity model；记录真实概率，设置正的最低覆盖。

## 6. 与 vOPD、DIAL 的区别

vOPD 的 $b(c)$ 不能依赖 sampled action，依靠 $\mathbb E_a[b(c)\nabla\log\pi(a\mid c)]=0$。这里的 $b(c,a)$ 可以依赖动作，但要保留所有位置的预测项，再逆概率校正残差。

DIAL 保留 top-K、删除其他教学位置，没有上述 residual correction，因此通常改变监督分配。DR 的目标是在预算下降时近似原始完整教学方向；它并未判断哪些 teacher 偏好更符合任务正确性。

## 7. 哪些能保证、哪些不能

能保证：查询随机性下局部梯度方向期望保持，条件如上。不能保证：predictor 好、总训练更快、最终正确率更高、Adam 更新无偏，或 PPO clipping 后仍保持同一目标。把 AIPW 系数塞进非线性的 clipped surrogate，需要另行推导。
