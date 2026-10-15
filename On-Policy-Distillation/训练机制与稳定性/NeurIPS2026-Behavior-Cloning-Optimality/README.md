# Noisy Expert / NAIL

**认知：** 当专家带噪声时，学生访问自己的状态再问专家，可能比离线模仿更有样本效率。

**方法：** 建立 noisy-expert 模型，推导离线与在线 imitation 的复杂度分离，并提出鲁棒在线 NAIL 变体。

**改动位置：** rollout 与前缀分布；散度、估计器与梯度路径。

## 关键公式

公式性质：原文设置式。

$$
\pi_\eta^*=(1-\eta)\pi^*+\eta\nu
$$

**适用边界：** 多项式或 horizon-free 结论有噪声与 domination 条件；不是所有 SFT 都指数困难。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2606.30923) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；§2、§3–5。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
