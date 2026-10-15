# AsyncOPD

**认知：** 异步训练能提高吞吐，但 loss 方向和打分缓存决定旧数据是否还能用。

**方法：** 比较 FKL/RKL 的陈旧性，在 learner 端重算当前 student 信号，并用多样本 MC 降低单采样方差。

**改动位置：** rollout 与前缀分布；散度、估计器与梯度路径；约束、更新幅度与训练调度。

## 关键公式

公式性质：结构示意。

$$
\widehat g_M=\frac1M\sum_{m=1}^M g(a_m;\pi_\theta,T),\quad a_m\sim\mu
$$

**适用边界：** μ 是真实提议分布；需要论文对应的重要性校正，不能简单重采样就宣称无偏。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2606.24143) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及估计器部分。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
