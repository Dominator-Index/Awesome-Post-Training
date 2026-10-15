# Many Faces of OPD

**认知：** teacher 强不够，还要检查前缀兼容性、TopK 梯度与特权信息是否能迁移。

**方法：** 诊断三类失败，比较 stop-gradient TopK、RLVR teacher 与 SFT 稳定化。

**改动位置：** teacher 来源与额外信息；散度、估计器与梯度路径；约束、更新幅度与训练调度。

## 关键公式

公式性质：结构示意。

$$
g=\sum_{a\in\mathcal K}\operatorname{sg}\!\left[S(a)\log\frac{T(a)}{S(a)}\right]\nabla\log S(a)
$$

**适用边界：** TopK surrogate 与完整 KL 不能无条件等同；这是诊断加稳定化研究。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.11182) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要、TopK 梯度部分。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
