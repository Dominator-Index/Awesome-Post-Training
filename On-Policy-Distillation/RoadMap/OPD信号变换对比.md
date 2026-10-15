# TOP-D、PowerOPD、DIAL-OPD：同一类认知，不同位置的变换

## 1. 先回答“这几个是不是几乎一样”

**可以把它们放在同一条研究分支：不直接接受原始 teacher–student log-ratio，而是重新决定教学强度。** 数学形式都很小，共享避免不合适分歧主导更新的动机。若要提出新 idea，单纯再换一个平滑函数或插值参数，和这条分支已有工作距离很近。

但是“差不多”应指研究方向接近，不能直接推成算法等价。其中 TOP-D 的完整算法还加入未来 return、长度归一化、token 组标准化和内部迭代，不能只用它的插值式代表所有变化。

来源：[TOP-D §3](https://arxiv.org/pdf/2607.04751)、[PowerOPD §4](https://arxiv.org/pdf/2606.17199)、[DIAL-OPD §4](https://arxiv.org/pdf/2610.11659)、[DN-MOPD §2](https://arxiv.org/pdf/2609.35347)。以下同一符号下重写原文核心式，额外推导明确标注。

## 2. 将所有方法写在一个更新框架中

令 $p_t=\pi_S(y_t\mid h_t)$、$q_t=\pi_T(y_t\mid h_t)$，即时局部更新可写成

$$
g=\frac1Z\sum_t\operatorname{sg}[m_t w_t f(p_t,q_t)]\nabla_\theta\log\pi_\theta(y_t\mid h_t).
$$

$m_t$ 决定是否教，$w_t$ 决定额外权重，$f$ 决定基础教学系数，$Z$ 决定各位置如何聚合。此式用于比较局部教学信号；TOP-D 的 return/归一化、DN-MOPD 的 PPO clipping 还需继续接到后面的组件。

| 方法 | 认知 | 核心改动 | 主要位置 |
|---|---|---|---|
| Vanilla OPD | 按老师/学生相对支持差异教 | $f=\log(q/p)$ | A/F |
| TOP-D | 老师目标太远，先变近再教 | $q_\lambda=(1-\lambda)p+\lambda q$，然后取 $\log(q_\lambda/p)$ | T/A；完整算法还有 N/C/L |
| PowerOPD | 相同比值的低概率分歧不该同样强 | $f=q^\alpha-p^\alpha$ | A/F/W |
| DIAL-OPD | 固定监督预算下，大分歧还需有概率支持 | 用插值分数选 top-K；保留 $f=\log(q/p)$ | W；伴随 N/L |
| DN-MOPD | 不同领域老师的反馈尺度需要校准 | $w_d=\operatorname{clip}(\sigma_{\rm all}/\sigma_d,0.25,4)$ | W/N |

## 3. TOP-D：真正的分布线性插值

$$
\boxed{q_\lambda(a\mid h)=(1-\lambda)p(a\mid h)+\lambda q(a\mid h),\quad
f_{\rm TOP}(p,q)=\log\!\left(1-\lambda+\lambda\frac qp\right).}
$$

$0<\lambda<1$ 时插值后的完整词表分布仍归一化。其负向系数有下界 $\log(1-\lambda)$，避免老师极低概率给出无限负惩罚；正向仍可能无界。它只依赖比值，同倍缩小 $p,q$ 不改变信号。

因此 TOP-D 主要解决“负向信号过猛”，并没有仅凭插值区分两个都低概率的分歧。不要写成双侧 bounded reward。

## 4. PowerOPD：单调幂变换后做差

$$
\boxed{f_{\rm Power}(p,q)=q^\alpha-p^\alpha,\quad \alpha>0.}
$$

这里没有直接构造插值 teacher。一个 sampled token 的两个概率各做幂变换，然后相减。它同时保持正负号、限制系数到 $[-1,1]$，并满足

$$
f_{\rm Power}(\epsilon p,\epsilon q)=\epsilon^\alpha f_{\rm Power}(p,q).
$$

因此它与 TOP-D 的重要区别是：不仅改比值的敏感度，还引入绝对概率尺度。对整词表 $q^\alpha$ 再归一化会得到不同算法，不能当作这篇的公式。

### 一个进一步的数学理解（本笔记推导）

固定前缀与 teacher 时，定义

$$
D_\alpha(p,q)=\sum_a\frac{p_a^{\alpha+1}-(\alpha+1)p_a q_a^\alpha+\alpha q_a^{\alpha+1}}{\alpha+1}.
$$

这是凸函数 $\sum_a p_a^{\alpha+1}/(\alpha+1)$ 的 Bregman divergence，非负且 $p=q$ 时为零。其负梯度为

$$
-\nabla_\theta D_\alpha
=\sum_a p_a(q_a^\alpha-p_a^\alpha)\nabla_\theta\log p_a.
$$

所以 PowerOPD 的 sampled-token 即时更新也可理解为这个局部散度的采样梯度。$\alpha=1$ 时 $D_1=\frac12\sum_a(p_a-q_a)^2$，相当于概率空间平方距离的局部更新。这是对方法的数学重表达，不是论文额外给出的实验结论；冻结上下文的限制仍然存在。

这说明“用 power”可以改变所对应的局部几何，不能只当成 TOP-D 的另一种同义插值。

## 5. DIAL-OPD：插值的是选词分数

$$
L(p,q)=\int_0^1p^{1-u}q^u\,du,
$$

$$
\boxed{s_\beta=L(p,q)^\beta|\log(q/p)|
=|\log(q/p)|^{1-\beta}|q-p|^\beta.}
$$

$0\le\beta\le1$ 时，确实可以称为相对分歧与绝对分歧分数之间的几何插值。$\beta>1$ 是外推，不是两端之间的凸插值。它不是 TOP-D 的 teacher/student 分布混合。

$$
m_t=\mathbf1\{t\in\operatorname{TopK}(s_\beta)\},\qquad
\boxed{f_{\rm effective}=m_t\log(q_t/p_t).}
$$

选中以后仍使用原始 log-ratio，因此选词分数越有界，不代表最终更新系数也有界。

### 与 PowerOPD 的精确联系

$$
\boxed{s_{\rm DIAL,\beta=1}=|q-p|=|f_{\rm Power,\alpha=1}|.}
$$

这是两个方法之间最直接的重合。它们用同一个绝对概率差表达支持程度，**一个用它决定保留谁，另一个用有符号差值决定每个位置的更新幅度**。

即便 DIAL 保留所有 token，$m_t=1$，也退回 vanilla log-ratio，并不会退回 PowerOPD。只有特定数据和参数下的偶然数值相同，不能建立一般等价关系。

## 6. 一个数字例子看差别

取 TOP-D $\lambda=0.5$、PowerOPD $\alpha=1$、DIAL $\beta=1$。

| student $p$ | teacher $q$ | 原始 log-ratio | TOP-D 系数 | PowerOPD 系数 | DIAL 选择分数 |
|---|---|---|---|---|---|
| 0.08 | 0.8 | 2.303 | 1.705 | 0.72 | 0.72 |
| 0.0001 | 0.001 | 2.303 | 1.705 | 0.0009 | 0.0009 |
| 0.8 | 0.08 | -2.303 | -0.598 | -0.72 | 0.72 |

TOP-D 对前两行给出一样的教学系数。PowerOPD 在第二行直接给极小系数。DIAL 更可能删第二行，但如果它被 top-K 保留，最终系数依然是 2.303。第三行 DIAL 按大分歧选中后给负向监督，不是只保留老师认可的位置。

## 7. DN-MOPD 属于另一粒度的权重校准

$$
\boxed{\widetilde r_t=w_d\log(q_t/p_t),\qquad
w_d=\operatorname{clip}(\sigma_{\rm all}/\sigma_d,0.25,4).}
$$

领域内部每个 token 同乘一个正数，改变的是老师之间的更新强度，并没有 token 级插值。它保留领域内排序，也没有测最终正确率贡献。它可以与上面的信号设计讨论放在同一张图，但不应混成同一种操作。

## 8. 这些方法到底证明了什么能 work

- TOP-D：限制负向教学信号，并配合 return 与受控更新，在其设置中能稳定训练。
- PowerOPD：把绝对概率尺度引入即时系数，在其数学模型组合中比原始 log-ratio 和一些稳定化 baseline 更好。
- DIAL-OPD：监督密度与选择标准值得分开设计，较少监督位置也能优于全监督。
- DN-MOPD：多领域反馈的尺度失衡会影响能力合并，压低过强领域可能比只增加弱领域更有效。

这些是各自论文内的经验，没有统一实验可以据此排列四者优劣，也没有跨论文证明某种变换始终最优。

## 9. 对我们找 idea 的含义

**已有：**对 teacher/student 分歧做插值、单调变换、幅度稳定化、token 筛选、领域缩放。这条研究方向已经相当拥挤。

**这几篇尚未直接解决：**同样的概率分歧里，哪些训练动作真的增加数学或代码任务的最终成功率？高概率的风格偏好可能无用；低概率的修复动作可能关键。概率支持与任务价值之间还差一个论证。

这只是对这几篇覆盖范围的判断，不能单凭这四篇宣布全球 novelty。未来 idea 若围绕这个缺口，仍需对照 AOPD、PIVOT、CC-OPD 等相关工作的任务收益或反事实定义，再做新一轮检索。
