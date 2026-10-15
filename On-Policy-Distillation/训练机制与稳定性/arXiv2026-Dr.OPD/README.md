# Dr. OPD：根据任务收益决定哪些 teacher 信号值得学习

完整题名：**Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation of Large Language Models**。作者：Zhenyu Wang、Tianze Wang、Linjun Zhang、Yifan Hu。

- [论文页面](https://arxiv.org/abs/2609.38025) · [本地 PDF](paper.pdf)，v1，2026-09-29，22 页。
- [官方代码](https://github.com/zywang0701/Dr-OPD)；本地 [main](main) 指向 `/usr/data/Dr-OPD`。
- 按 arXiv 年份归档，未确认会议投稿或录用状态。

## 核心问题

普通 OPD 对各 token 的 teacher 信号采用相同权重。但修正关键推理错误与偏好不同措辞，对最终任务结果的影响不同。Teacher 的偏好告诉我们“它更喜欢什么”，未必告诉我们“student 学这个方向能不能提高正确率”。

Dr.OPD 用最终任务奖励来评价蒸馏方向：**与提高奖励的参数更新方向一致的 teacher 信号加强，冲突的信号减弱。**

## 方法与公式

对 student 采样 token $y_t$，定义 teacher 修正向量：

$$
g_t=\log\frac{p_T(y_t\mid s_t)}{p_S(y_t\mid s_t)}\nabla_\theta\log p_S(y_t\mid s_t).
$$

它表示局部向 teacher 靠近的更新方向。再计算它与奖励梯度的对齐程度：

$$
\hat c_t=g_t^\top\hat r,
$$

其中 $\hat r$ 是用当前 batch 的最终奖励、组相对 advantage 估计的奖励梯度。正 credit 增强指导，负 credit 降低指导强度。权重为：

$$
\hat w_t=\operatorname{clip}\left(1+\lambda\frac{\hat c_t}{\max(\sigma,\epsilon)},w_{\min},w_{\max}\right).
$$

$\sigma$ 为 credit 的 RMS 尺度，$\lambda$ 控制调节强度。权重保持正值，所以负 credit 会削弱 teacher 修正，而非直接反转其方向。

逐 token 构造参数梯度很昂贵，作者用一次 Jacobian–vector product（JVP）计算各 token 沿奖励梯度方向的导数。每轮采样、评分、计算 credit 和权重，最后进行一次加权 OPD 更新。

论文并非简单对 reverse-KL 的各词表项乘权重。其加权散度为：

$$
D_w(p\|q)=\sum_v w_v\left[p_v\log\frac{p_v}{q_v}-p_v+q_v\right].
$$

附加的 $-p_v+q_v$ 保证 token 依赖权重下的非负性，$w\equiv1$ 时恢复普通 KL。

## 理论与实践的区别

理论将选权重写为双层优化：内层做加权蒸馏，外层最大化更新后 student 的奖励。实际不完整求解内层，而用一步更新近似；理论 credit 使用普通 OPD 更新后的奖励梯度，实践进一步用当前模型的奖励梯度近似。

论文证明的单步奖励改善依赖平滑性、步长和权重约束，不能直接视为带采样误差、近似 JVP 和多轮训练的无条件保证。

## 实验与边界

Qwen3-4B teacher 到 Qwen3-1.7B Instruct student 的强到弱蒸馏中，五项数学基准均值从普通 OPD 的 25.3% 提高到 35.0%，增加 9.7 个百分点，teacher 均值为 34.1%。Base student 的数学增益较小，为 1.8 个百分点；不能将 9.7 推广到所有配置。

论文还测试代码生成及同规模 RL teacher 蒸馏。它每 10 步在所评估数据集上评估、报告最佳 checkpoint，这种选择协议可能影响结果估计。奖励梯度在极低正确率或全部同奖励组中缺乏信息，credit 也会受采样噪声与 batch 构成影响。

官方仓库当前提供数学训练路径；README 说明代码生成实验专用 launcher 未包含在该快照中。本文档是方法笔记，未独立复现实验。

## 与我们此前讨论的联系

它直接对应“teacher 的指导是否真的提高 student 任务表现”这一问题，通过梯度对齐评价局部指导，而不是逐个关键前缀展开分支续写。因此，基于最终奖励选择蒸馏信号的研究需要明确对比 Dr.OPD。
