# DN-MOPD：Beyond Teacher Assignment 解读

来源：[论文](https://arxiv.org/pdf/2609.35347) · [项目](https://lixin.ai/DN-MOPD/) · [代码](https://github.com/LiXin97/DN-MOPD)。目录按 arXiv 2026 归档，不推定会议录用。

**认知：多个领域的老师共同教一个学生，反馈声音最大的老师可能占据更新；分配老师后，还要校准各领域的反馈尺度。**

**方法：保持按领域选老师的规则，用领域 log-ratio 标准差的倒数缩放该领域的 advantage，不减均值，保持正负方向。**

- [PDF](paper.pdf)
- [完整数学与证据分析](analysis.md)
- [TOP-D / PowerOPD / DIAL-OPD 对比](../../RoadMap/OPD信号变换对比.md)
- 代码：[main](main) → `/usr/data/DN-MOPD`

本次归档与原文分析，不代表实验复现。
