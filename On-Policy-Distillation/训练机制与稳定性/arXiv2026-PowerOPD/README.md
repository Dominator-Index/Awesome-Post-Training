# PowerOPD：Bounded Power Transformation 解读

来源：[论文](https://arxiv.org/pdf/2606.17199) · [代码](https://github.com/EIT-NLP/PowerOPD)。按 arXiv 2026 归档。

**认知：相同比值可能来自有支持的分歧，也可能来自两个极小概率；教学强度应该考虑绝对概率，而不是让 log-ratio 的极值主导更新。**

**方法：将即时教学系数从 $\log q-\log p$ 改成 $q^\alpha-p^\alpha$，保持鼓励/惩罚的符号，原生限制系数幅度。**

- [PDF](paper.pdf)
- [完整数学与证据分析](analysis.md)
- [TOP-D / PowerOPD / DIAL-OPD 对比](../../RoadMap/OPD信号变换对比.md)
- 代码：[main](main) → `/usr/data/PowerOPD`

本次归档与原文分析，不代表实验复现。
