# GUI-SD

**认知：** GUI 定位需要把监督集中到可靠且影响坐标的数字位置。

**方法：** teacher 看目标 bounding box 与 Gaussian soft mask，按数字位意义及 teacher confidence 加权。

**改动位置：** teacher 来源与额外信息；token / trajectory 权重与门控；监督接口与动作单位。

## 关键公式

公式性质：结构示意。

$$
L=\sum_t w_{\rm digit}(t)\,w_{\rm conf}(t)\,D(T_t\|S_t)
$$

**适用边界：** target 区域只供训练 teacher；confidence 仍是代理。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.00642) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及 entropy-guided objective。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
