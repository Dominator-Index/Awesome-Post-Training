# Vision-OPD

**认知：** 同一个模型放大关键区域后看得更清楚，这种感知优势可以教给看全图的 student。

**方法：** 裁剪区域作 privileged teacher 输入，全图 student 生成 rollout，再匹配两者分布。

**改动位置：** teacher 来源与额外信息。

## 关键公式

公式性质：原文机制式。

$$
T_t=S_{\bar\theta}(\cdot|x,I_{\rm crop},y_{<t});\quad S_t=S_\theta(\cdot|x,I_{\rm full},y_{<t})
$$

**适用边界：** 模型不一定更大；核心是输入信息差。区域获取也需要说明。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.18740) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及条件策略。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
