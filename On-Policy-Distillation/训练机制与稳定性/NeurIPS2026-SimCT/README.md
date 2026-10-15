# SimCT

**认知：** 两模型 tokenizer 不同时，先找到可比较的文本单位，比直接比较 token id 更重要。

**方法：** 除共享 token 外，纳入双方都能表达的短 multi-token continuations，扩展监督接口。

**改动位置：** 监督接口与动作单位。

## 关键公式

公式性质：结构示意。

$$
\mathcal U=\mathcal V_{\rm shared}\cup\mathcal C_{\rm jointly\ tokenizable}
$$

**适用边界：** 扩大接口不等于强行对齐不同 token id；序列概率需按各自 tokenizer 计算。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.07711) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及监督接口部分。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
