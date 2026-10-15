# 01｜先用 STOPD 原文公式 formulate IPW

**认知：少观察后段，可以用更大的权重补偿后段的覆盖；要恢复的是完整局部 OPD 的梯度和。**

**方法：独立随机截断，保留 token 的梯度乘其被保留概率的倒数，再使用上一批长度估计作分母。**

依据：[本地 STOPD PDF](../paper.pdf) §3.1–3.2、Appendix A.1；[官方仓库](https://github.com/YouAreSpecialToMe/STOPD)。下面保留原文 $L,K,T_i,S_i,M_i,p_t,h_{i,t},G,N,C_k$ 的定义与 1-based token 索引。扩展讨论见后续编号。

## 1. 完整训练本来要计算什么

固定第 $k$ 个训练步的 student 参数 $\theta_k$。第 $i$ 个 prompt 的完整回答为 $y_i$，长度 $T_i\le L$；$L$ 为最大生成长度。

原文使用一般局部 loss $\ell_{i,t}(\theta)$，不将 IPW 限制为某一种 KL。定义

$$
h_{i,t}=\nabla_\theta\ell_{i,t}(\theta_k),\qquad
G_k=\sum_{i=1}^{B}\sum_{t=1}^{T_i}h_{i,t},\qquad
N_k=\sum_iT_i,\qquad g_k^V=G_k/N_k.
$$

$G_k$ 是参数向量；$N_k$ 是 token 数；$g_k^V$ 是 Vanilla 的 batch token 平均梯度。这是 STOPD 原文 Eqs.3–4 的比较对象。

若选用即时 sampled-token OPD surrogate，可具体写为

$$
c_{i,t}=(x_i,y_{i,<t}),\qquad
r_{i,t}=\log q(y_{i,t}\mid c_{i,t})-\log\pi_{\theta_k}(y_{i,t}\mid c_{i,t}),
$$

$$
\ell_{i,t}(\theta)=-\operatorname{sg}[r_{i,t}]\log\pi_\theta(y_{i,t}\mid c_{i,t}),\qquad
h_{i,t}=-r_{i,t}\nabla_\theta\log\pi_\theta(y_{i,t}\mid c_{i,t})\big|_{\theta_k}.
$$

$\operatorname{sg}$ 表示系数冻结。本文的 $h$ 是最小化 loss 的梯度，后文若使用 ascent 方向 $r\nabla\log\pi$，会明确改用 $v$ 或 $g^+$。

即时条件 KL 梯度和完整轨迹 KL 的梯度不是无条件相同的：前者不对采样前缀分布求导。STOPD 的保持结论针对**给定 baseline 的局部 loss**，并不替它证明序列目标。

## 2. 怎样产生随机停止位置：原文 Eq.1

$$
\boxed{A_i\sim\operatorname{Uniform}\{0,\ldots,L-1\},\qquad
S_i=\min\{K(\lfloor A_i/K\rfloor+1),L\},\qquad M_i=\min(T_i,S_i).}
$$

$K$ 是 block 宽度；anchor 向上取整到 block 末端；$M_i$ 是实际生成长度。遇到自然 EOS 就结束，不强制补到 $S_i$。完整与截断生成可在相同随机 token 抽样下耦合，保留前缀相同。

## 3. 保留概率与权重：原文 Eq.2

$$
\boxed{p_t=\Pr(S_i\ge t)=\frac{L-K\lfloor(t-1)/K\rfloor}{L}>0,\qquad
w_t=1/p_t.}
$$

这里 $p_t$ 是位置 inclusion probability，**不是 student token probability，也不是 teacher/student likelihood ratio**。本系列其他文件把位置或查询概率记为 $\pi_t^{\rm obs}$ 或 $e_t$，避免冲突。

当 $K\mid L$，最大权重为 $L/K$。block 较大能减少极端权重，但增加保留的生成 token。

## 4. IPW 恢复什么：原文 Eqs.6–8

设 $I_{i,t}=\mathbf1\{S_i\ge t\}$，完整轨迹上的有效位置满足 $t\le T_i$。截断估计为

$$
\boxed{\widehat G_k=\sum_{i,t\le T_i}\frac{I_{i,t}}{p_t}h_{i,t},\qquad
\widehat N_k=\sum_{i,t\le T_i}\frac{I_{i,t}}{p_t}.}
$$

实现只遍历 $t\le M_i$，数学上写出未观察项不代表计算了它们。固定完整轨迹与 teacher 随机性，仅对 stop 随机性取期望：

$$
\mathbb E_S\left[\frac{I_{i,t}}{p_t}h_{i,t}\right]=h_{i,t},\qquad
\boxed{\mathbb E_S[\widehat G_k]=G_k,\quad\mathbb E_S[\widehat N_k]=N_k.}
$$

这来自线性期望，不要求同一回答内的位置独立。实际上它们高度相关：看到后段必然已看到前段。

## 5. 一个能直接算清楚的例子

$L=8,K=2$，四个 block 的保留概率分别为 $1,3/4,1/2,1/4$，权重为 $1,4/3,2,4$。假设完整回答有 8 个 token：

| block | 保留概率 | 梯度贡献 | 观察到时加权 |
|---|---|---|---|
| 1 | 1 | $H_1$ | $H_1$ |
| 2 | 3/4 | $H_2$ | $4H_2/3$ |
| 3 | 1/2 | $H_3$ | $2H_3$ |
| 4 | 1/4 | $H_4$ | $4H_4$ |

每项的期望都是 $H_b$。如果所有 token 的某个梯度坐标都为 1，四种停止长度 2/4/6/8 等概率，IPW 和分别为 $2,14/3,26/3,50/3$，均值为 8。恢复期望，单次仍很波动。

## 6. 分母：原文 Eqs.4–5、9

$$
\widehat L_k(\theta)=\frac1{C_k}\sum_{i,t\le M_i}w_t\ell_{i,t}(\theta),\qquad
\widehat g_k=\widehat G_k/C_k.
$$

$C_k$ 在当前 stop 采样前固定。做 $W$ 步完整 warmup 后，

$$
C_{W+1}=N_W,\qquad C_k=\widehat N_{k-1}\quad(k>W+1).
$$

因此

$$
\boxed{\mathbb E_S[\widehat g_k]=G_k/C_k=(N_k/C_k)g_k^V.}
$$

梯度和无偏，不等于当前 batch 的 token 平均梯度严格无偏。用 $\widehat G_k/\widehat N_k$ 也不能凭两个分别无偏就得到比值无偏；Adam 处理估计梯度后的参数更新同样不能直接套用线性无偏结论。

## 7. 结论成立的条件

1. 同一前缀的 next-token 分布不因剩余 budget 改变；不能把“快到截断，赶快给答案”之类额外提示混入。
2. 局部 loss 不依赖未生成未来；终局正确率、future return 不能直接当已知局部系数。
3. stop 与完整轨迹及 teacher 随机性独立，各位置正概率，权重准确，权重和分母不反传。
4. 原文比较以局部梯度为对象，不证明长期训练轨迹或最终准确率完全相同。

若改成根据前缀自适应停止，条件 3 已经变化，需要新的条件概率与序列推导，见 [04](04-序列DR能否减少STOPD的rollout.md)。

## 8. IPW 来自哪里

逆 inclusion probability 的思想在抽样统计中已有经典的 [Horvitz–Thompson 1952](https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446)。因果推断常用它处理 treatment assignment；STOPD 用它处理计算造成的缺失。STOPD 附录明确联系 [McLeish 2011](https://arxiv.org/abs/1005.2228) 与 [Beatson–Adams 2019](https://arxiv.org/abs/1905.07006) 的随机截断估计。

所以借鉴相关理论有依据；但是“用了 IPW”还不意味着已经测出了一个 token 的因果价值。
