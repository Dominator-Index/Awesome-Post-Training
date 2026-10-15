# PW-OPSD / Teacher Token Reliability

**认知：** 高 entropy 既可能是坏分支，也可能是多种好解法；可靠性不能只看 entropy。

**方法：** 先做分支可行性诊断，再在训练中按相对位置给递增权重，弱化早期监督。

**改动位置：** token / trajectory 权重与门控；token length、截断与预算。

## 关键公式

公式性质：结构示意。

$$
L=\frac{\sum_t w(t/|y|)\ell_t}{\sum_t w(t/|y|)},\quad w\ \text{随位置递增}
$$

**适用边界：** 位置是结构代理；训练不需要逐次做 branch intervention，非精确可靠性标签。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.21606) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；本地 analysis.md §4–5。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
