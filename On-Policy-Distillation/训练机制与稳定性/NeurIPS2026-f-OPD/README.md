# f-OPD

**认知：** 旧 rollout 可以复用，但应按新鲜程度决定它还值得信任多少。

**方法：** 根据策略漂移、teacher 上下文漂移和更新滞后降低旧样本权重，并加入 rollout 锚定及缓冲区刷新。

**改动位置：** rollout 与前缀分布；token / trajectory 权重与门控；约束、更新幅度与训练调度。

## 关键公式

公式性质：原文核心式。

$$
f_i=\frac{\exp[-\alpha\sqrt{D_i^{\rm roll}}-\beta\sqrt{D_i^{\rm sup}}]}{1+\tau_i};\quad w_i=[f_i-\xi]_+
$$

**适用边界：** 新鲜度是偏差代理，不是精确因果可靠性。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.17862) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；§3–4。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
