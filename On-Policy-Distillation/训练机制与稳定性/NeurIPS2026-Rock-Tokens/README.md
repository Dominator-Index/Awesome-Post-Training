# Rock Tokens

**认知：** 高 loss 不一定意味着值得学；持续难以纠正的风格残差可能消耗大量梯度。

**方法：** 识别持续高 loss token 及关联窗口，降低其权重；Rock-Freeze 将这些区域权重设为零。

**改动位置：** token / trajectory 权重与门控。

## 关键公式

公式性质：原文特例。

$$
w_t=\begin{cases}0,&y_t\in\mathcal R\ \text{or}\ t\in\mathcal W_R\\1,&\text{otherwise}\end{cases}
$$

**适用边界：** 需要持续性与干预诊断；不能把所有高 loss token 都删掉。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.09253) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；§4.2, Eq.19。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
