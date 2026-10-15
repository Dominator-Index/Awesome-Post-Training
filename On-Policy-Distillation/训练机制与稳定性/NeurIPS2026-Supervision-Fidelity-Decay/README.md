# LGR / Supervision Fidelity Decay

**认知：** 学生前缀越长，teacher 的纠正能力可能越弱，不能默认后段监督始终有效。

**方法：** 对候选 token 看一步后 teacher confidence，构造组相对奖励，并用 entropy-triggered tree attention 控制开销。

**改动位置：** teacher 来源与额外信息；奖励、return、baseline 与 critic；token / trajectory 权重与门控；token length、截断与预算。

## 关键公式

公式性质：结构示意。

$$
r_t(a)\ \propto\ \operatorname{Conf}_T(h_ta)\quad\longrightarrow\quad\text{group-normalized reward}
$$

**适用边界：** 下一步置信度是监督强度代理，不等于最终答对概率。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.30833) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及 LGR 方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
