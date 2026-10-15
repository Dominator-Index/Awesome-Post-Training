# OPHSD / Harness Self-Distillation

**认知：** 额外推理流程可以只在训练中当脚手架，再把它的帮助内化到模型里。

**方法：** 用当前模型加 draft–verify、plan–solve harness 构造 teacher，再在学生轨迹上蒸馏。

**改动位置：** teacher 来源与额外信息。

## 关键公式

公式性质：结构示意。

$$
T=\operatorname{Harness}(S);\quad L=\mathbb E_{h\sim d_S}D(T_h\|S_h)
$$

**适用边界：** 部署时能否移除 harness 取决于任务与实际内化效果。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.08741) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及 teacher 构造。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
