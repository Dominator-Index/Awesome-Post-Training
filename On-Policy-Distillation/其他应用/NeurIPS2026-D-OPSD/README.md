# D-OPSD / Few-step Image Diffusion

**认知：** 少步扩散模型继续学新概念时，应沿自己的少步轨迹监督，避免破坏原生成方式。

**方法：** student 仅看文本，self-teacher 同时看文本与目标图像特征，匹配学生 rollout 上的预测。

**改动位置：** teacher 来源与额外信息；监督接口与动作单位；rollout 与前缀分布。

## 关键公式

公式性质：结构示意。

$$
L=\mathbb E_{z_k\sim S}\sum_kD\big(T_k(\cdot|\text{text,image}),S_k(\cdot|\text{text})\big)
$$

**适用边界：** 图像扩散轨迹不是文本 CoT；具体预测对象以原文为准。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.05204) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
