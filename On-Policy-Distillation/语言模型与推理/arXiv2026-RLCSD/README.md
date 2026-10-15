# RLCSD：用正确与错误参考抵消特权信息带来的风格漂移

完整题名：**RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation**。作者：Leyi Pan、Shuchang Tao、Yunpeng Zhai、Lingzhe Zhang、Zhaoyang Liu、Bolin Ding、Aiwei Liu、Lijie Wen。

- [论文页面](https://arxiv.org/abs/2606.11709) · [本地 PDF](paper.pdf)，v2，2026-09-14，24 页，首次提交 2026-06-10。
- [官方代码](https://github.com/THU-BPM/RLCSD)；[main](main) 指向 `/usr/data/RLCSD`。
- 按 arXiv 年份归档，未确认会议状态。

## 核心问题

OPSD 将普通输入下的 student 分布与带参考解法的 teacher 分布对齐。但参考信息既提供任务知识，也改变语言风格：模型变得更直接、更简短，某些措辞差异产生很强的 token 信号。

作者称此为 privilege-induced style drift，特权信息引起的风格漂移。若直接使用全部概率差，训练可能主要改变风格、缩短回答，而未充分纠正任务关键内容。

## 关键思路：正确参考与错误参考作对比

在同一 student 前缀上，分别加入正确参考 $y_c^*$ 和错误参考 $y_w^*$：

$$
e_{c,t}=\log\pi_T(y_t\mid x,y_c^*,y_{<t})-\log\pi_S(y_t\mid x,y_{<t}),
$$

$$
e_{w,t}=\log\pi_T(y_t\mid x,y_w^*,y_{<t})-\log\pi_S(y_t\mid x,y_{<t}).
$$

两者相减后 student 项抵消：

$$
e_{\mathrm{ctr},t}=e_{c,t}-e_{w,t}
=\log\pi_T(y_t\mid x,y_c^*,y_{<t})-\log\pi_T(y_t\mid x,y_w^*,y_{<t}).
$$

正确和错误参考使用完全相同的包装模板，甚至错误参考也被放入同样的“Reference Solution / Correct final answer”字段。这样，两边可能共享“看过参考后更直接”的风格变化，而在内容上不同；相减尝试保留与正确性相关的差别。

这里不是训练 student 接受错误答案。错误参考只是构造对照分布。共享风格被抵消是设计动机与经验现象，不是对任意提示和任务都成立的严格保证。

## 两个实现细节

1. 从同组错误 rollout 中取 $K$ 个负参考，先平均概率再取对数，降低单个错误类型带来的噪声：

$$
e_{\mathrm{ctr},t}
=\log\pi_T(y_t\mid x,y_c^*,y_{<t})
-\log\left[\frac1K\sum_{k=1}^K\pi_T(y_t\mid x,y_{w,k}^*,y_{<t})\right].
$$

这不是对 log-probability 直接取平均。

2. 排除目标 rollout 自身作为它的参考，避免 teacher 因看到待评分原文而产生复制式过度自信。正参考可来自其他正确 rollout 或数据集真实解法。

## 最终怎样训练？

它不直接用对比信号替代正确性奖励。先用验证器计算 GRPO 的组相对 advantage $A_{\mathrm{ORM}}$，再将对比信号压缩为：

$$
r_t=\lambda\tanh(e_{\mathrm{ctr},t}/\tau),\qquad m_t=\mathbf1[|r_t|>\delta].
$$

较弱的信号不调节，选中位置的 advantage 改为：

$$
\tilde A_t=
\begin{cases}
\max(0,A_{\mathrm{ORM}}+r_t),&A_{\mathrm{ORM}}\ge0,\\
\min(0,A_{\mathrm{ORM}}+r_t),&A_{\mathrm{ORM}}<0.
\end{cases}
$$

这确保不反转验证器确定的符号。未选中的 token 用原始 advantage，选中的 token 用修正后的 advantage，分别组成 PPO 式裁剪目标。

重要限制：失败 rollout 的 token 仍不因此获得正向 advantage；这不是直接用价值函数给某个有益纠错 token 独立赋予正信用。

## 数据与适用边界

它需要组中同时有正确和错误 rollout；论文丢弃全对或全错组。由此不能把它当作解决“全部失败、GRPO 没有梯度”的冷启动方案。即使提供真实解法，负参考和组相对奖励仍有采样要求。

额外的正、负参考打分增加开销；负参考代表性、模板对称性及验证器质量影响结果。形式上更干净的对比信号也不自动等于因果 token 贡献。

作者在 Qwen3-1.7B/4B/8B 与 Olmo-3-7B-Think 上测试数学和逻辑推理，并提供 agentic 任务补充结果；报告比 GRPO 和所测 OPSD 方法更好的表现与较稳定的生成长度。本文档未独立复现实验。

## 与近期阅读的关系

| 工作 | 核心处理对象 |
|---|---|
| Rethinking OPSD | 揭示特权信息可能抑制长推理分支与纠错 |
| ReDraft | 把失败文本修订成验证通过的目标，再 SFT |
| Dr.OPD | 用奖励梯度对齐来选择 teacher 指导强度 |
| RLCSD | 用正负参考对比削弱共同风格变化，再调节 GRPO advantage |

因此，“teacher 看参考解法后产生的概率差未必都是有用的任务信号”已是明确研究问题。RLCSD 给出了一种对照构造方式，与 ReDraft 的完整文本修订及 Dr.OPD 的梯度评价不同。
