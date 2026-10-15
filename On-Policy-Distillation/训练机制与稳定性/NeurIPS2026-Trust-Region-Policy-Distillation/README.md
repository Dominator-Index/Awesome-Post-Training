# TOP-D

**认知：** teacher 太远时先构造较近的目标，再用受控更新逐步接近它。

**方法：** teacher/student 概率插值，使用平滑 log-ratio、长度归一化未来 return、token 组标准化与 PPO 式内部迭代。

**改动位置：** teacher 来源与额外信息；散度、估计器与梯度路径；奖励、return、baseline 与 critic；advantage / loss 归一化；约束、更新幅度与训练调度；token length、截断与预算。

## 关键公式

公式性质：原文核心式。

$$
T_\alpha=\alpha T+(1-\alpha)S;\quad\widetilde r_t=\log\left(\alpha\frac{T(y_t)}{S(y_t)}+1-\alpha\right)
$$

**适用边界：** 不只做插值；概率空间插值与 log 概率缩放不同。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2607.04751) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；§3, Eq.5、7–8。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
