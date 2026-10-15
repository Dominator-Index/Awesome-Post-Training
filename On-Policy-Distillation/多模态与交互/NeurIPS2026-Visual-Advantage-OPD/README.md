# VA-OPD / Visual Advantage

**认知：** 真正视觉相关的 token 很少，按所有语言 token 平均会稀释视觉监督。

**方法：** 比较 teacher 有/无精细视觉的 logprob，重加权 rollout，并分高低 VA 两组分别平均 KL。

**改动位置：** teacher 来源与额外信息；token / trajectory 权重与门控；advantage / loss 归一化。

## 关键公式

公式性质：原文核心机制。

$$
v_t=\log T(y_t|I_{\rm fine},h_t)-\log T(y_t|I_{\rm coarse},h_t);\quad L=L_{\rm highVA}+L_{\rm lowVA}
$$

**适用边界：** VA 是输入干预的概率变化，不是 RL 的 Q−V；组大小归一化是关键。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.21924) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；§3.1–3.2。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
