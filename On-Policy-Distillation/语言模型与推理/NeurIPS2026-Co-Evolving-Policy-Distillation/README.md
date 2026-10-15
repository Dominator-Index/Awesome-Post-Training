# CoPD / Co-Evolving Policy Distillation

**认知：** 专家还在训练时就互相教，可能比专家完全分化后再合并更容易吸收。

**方法：** 并行 RLVR 训练领域专家，同时进行双向 OPD，使能力与行为模式共同演化。

**改动位置：** teacher 来源与额外信息；约束、更新幅度与训练调度。

## 关键公式

公式性质：结构示意。

$$
\{T_j^{(k+1)}\}_j\leftarrow\text{RLVR}+\text{mutual OPD}(\{T_j^{(k)}\}_j)
$$

**适用边界：** 是训练结构改变，不能归为只修改某个 token 权重。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2604.27083) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
