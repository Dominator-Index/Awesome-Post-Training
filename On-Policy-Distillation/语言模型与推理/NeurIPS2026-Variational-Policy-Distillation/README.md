# VPD / Variational Policy Distillation

**认知：** 会阅读反馈的 teacher 也需要继续进步，固定的反馈解释能力可能成为瓶颈。

**方法：** 用变分 EM：E-step 根据结果与语言反馈更新 teacher，M-step 蒸馏给 student。

**改动位置：** teacher 来源与额外信息；约束、更新幅度与训练调度；奖励、return、baseline 与 critic。

## 关键公式

公式性质：结构示意。

$$
T_{k+1}\leftarrow\operatorname{E-step}(R,\text{feedback});\quad S_{k+1}\leftarrow\operatorname{M-step}(T_{k+1})
$$

**适用边界：** 不是单纯把反馈字符串加进固定 teacher prompt。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.15113) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及 EM 方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
