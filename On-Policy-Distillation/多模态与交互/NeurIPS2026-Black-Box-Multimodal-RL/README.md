# PRISM / Black-box Multimodal RL

**认知：** 拿不到 teacher logits 时，也可以先对齐响应分布再做 RL。

**方法：** 在 SFT 与 RLVR 之间插入 response-level adversarial alignment，用感知/推理 MoE discriminator 提供信号。

**改动位置：** teacher 来源与额外信息；散度、估计器与梯度路径；约束、更新幅度与训练调度。

## 关键公式

公式性质：原文流程式。

$$
\text{SFT}\ \rightarrow\ \text{adversarial pre-alignment}\ \rightarrow\ \text{RLVR}
$$

**适用边界：** 属于黑盒、响应级对齐，不能塞进普通逐 token KL 的同一实现。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2604.28123) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及 PRISM 方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
