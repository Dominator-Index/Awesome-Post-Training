# d-OPSD / Self Future dLLMs

**认知：** 扩散语言模型的监督单位应跟去噪步骤对齐，而不是硬套自回归前缀。

**方法：** 以自生成答案作 suffix 条件构造 self-teacher，在学生去噪轨迹上做 step-level distillation。

**改动位置：** teacher 来源与额外信息；监督接口与动作单位。

## 关键公式

公式性质：结构示意。

$$
L=\mathbb E_{s_k\sim S}\sum_kD\big(T(\cdot|s_k,\text{suffix}),S(\cdot|s_k)\big)
$$

**适用边界：** 这里的 step 不是左到右的 token 位置；与图像 D-OPSD 不同。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2606.18195) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
