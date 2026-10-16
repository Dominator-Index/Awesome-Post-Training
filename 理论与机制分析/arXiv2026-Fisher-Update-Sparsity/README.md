# Fisher与后训练参数更新稀疏性

**论文：** Understanding LLM Parameter Update Sparsity through the Lens of Fisher

**作者：** Yufan Zhang、Sagnik Mukherjee、Hao Peng（UIUC）。

**来源：** [arXiv:2609.36262v1](https://arxiv.org/abs/2609.36262v1)，2026-09-28，预印本。

[本地PDF](paper.pdf) · [原文PDF](https://arxiv.org/pdf/2609.36262v1) · [分类目录](../README.md) · [仓库首页](../../README.md)

## 分类

- **主分类：** 后训练 → 理论与机制分析。
- **研究类型：** 理论解释 + 实证机制分析。
- **主题标签：** 参数更新稀疏性、对角模型Fisher、梯度集中、后训练动力学。
- **覆盖范围：** 理论讨论SFT、on-policy RL与OPD；实验主要验证RL与OPD。

**分类理由：** 核心贡献是解释后训练中的更新为何集中在少数参数上。Fisher固定掩码训练用于检验解释，因此适合独立的跨方法机制类别，并与[OPD训练机制与稳定性](../../On-Policy-Distillation/训练机制与稳定性/README.md)关联。分类判断依据原文§2–4。

## 核心认知与证据边界

**认知：** 在学习信号或分布差异尺度受控时，低Fisher限制单个参数的期望梯度；训练前的Fisher还能识别支持后续学习的稀疏参数子集。

原文Proposition 1（§3.1）的界限为：

$$
F_i=\mathbb{E}_{x\sim D_x,\,y\sim\pi_\theta(\cdot\mid x)}\!\left[(\partial_i\log\pi_\theta(y\mid x))^2\right],
\qquad
|g_i|\leq\sqrt{\mathbb{E}[a^2]}\sqrt{F_i}.
$$

其中$x$来自固定提示分布，$g_i$是期望更新梯度，$a$是随训练目标变化的有效信号。该式约束固定策略下的单步期望梯度；长期参数变化还依赖优化器、梯度噪声和学习率。

§4观察到高概率候选token的logit往往具有相近的参数敏感度，并通过概率加权的敏感度方差解释低Fisher。§3与附录C以梯度位置预测、固定掩码训练验证其作用。

**证据限制：** 固定掩码实验仍使用稠密计算，未建立实际显存或运行时间收益；归档与分类不代表实验复现。见原文§5。

## 归档记录

- 整理日期：2026-10-16。
- 固定版本：arXiv:2609.36262v1。
- PDF页数：19；已核对文件头、PDF元数据与前两页题名、作者、版本标记。
- SHA256：`373512b9c3f9cd0654880ef4b272e9d958cf0798a64164cd4e22c83feb30a958`。
