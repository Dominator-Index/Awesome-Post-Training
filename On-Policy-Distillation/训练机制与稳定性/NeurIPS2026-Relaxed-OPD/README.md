# REOPOLD / Relaxed OPD

**认知：** 强烈负监督应适度，低信息 token 不必占据同样的学习预算。

**方法：** 限制负 log-ratio 尾部，先按奖励门控、后按学生 entropy 选 token，分阶段探索与精炼。

**改动位置：** 散度、估计器与梯度路径；token / trajectory 权重与门控；约束、更新幅度与训练调度；advantage / loss 归一化。

## 关键公式

公式性质：arXiv 相关版本原文式。

$$
\widehat r_t=\max\left(\operatorname{sg}(r_t),\frac{\log\lambda}{1-\lambda}\right)
$$

**适用边界：** 依据 arXiv:2603.11137；与原列表及 workshop 版本的逐字对应尚未核实。

**正文状态：** 已补齐 [arXiv:2603.11137v1 PDF](paper.pdf)。该版本题名为 *Scaling Reasoning Efficiently via Relaxed On-Policy Distillation*；用户提供的 workshop OpenReview 版本（NPhEUvT1jH）仍受403限制，不能认定两个版本逐字相同。

[原始来源](https://arxiv.org/abs/2603.11137) · [RoadMap](../../RoadMap/README.md)

整理依据：RoadMap/sources 中的同方法 arXiv PDF；原列表版本待核对；§4, Eq.5–10。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
