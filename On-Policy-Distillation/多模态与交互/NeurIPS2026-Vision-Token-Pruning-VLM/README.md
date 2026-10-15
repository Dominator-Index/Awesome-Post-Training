# Fewer Tokens, Fewer Layers

**认知：** 视觉输入 token 数与模型层数，也会影响蒸馏后的推理成本。

**方法：** 题名提到 vision token pruning 与 OPD；未取得正文，不写具体剪枝规则。

**改动位置：** 监督接口与动作单位；token length、截断与预算。

**适用边界：** 视觉 token 数与回答 token 长度应分开记录。

**正文状态：** PDF尚未取得；不把本条标记为已精读。

[原始来源](https://openreview.net/forum?id=T7WbWqjpqm) · [RoadMap](../../RoadMap/README.md)

整理依据：仅题名；OpenReview 访问受限；论文题名。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
