# Rethinking OPSD

**认知：** teacher 看答案后不再需要探索，但 student 仍需要；这种偏好未必适合长思考。

**方法：** 改变特权信息和生成预算，分析 fork suppression、纠错与分支行为。

**改动位置：** teacher 来源与额外信息；token / trajectory 权重与门控；token length、截断与预算。

## 关键公式

公式性质：原文诊断量。

$$
\Delta_t=\log T(y_t|h_t,z)-\log S(y_t|h_t)
$$

**适用边界：** 诊断论文，没有一个已经验证的通用修复算法。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2607.05184) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；本地 analysis.md；长预算及 fork 诊断。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
