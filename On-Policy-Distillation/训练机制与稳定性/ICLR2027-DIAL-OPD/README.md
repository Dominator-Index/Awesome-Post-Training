# DIAL-OPD：Learning More from Fewer Tokens 解读

来源：[论文](https://arxiv.org/pdf/2610.11659)。按用户提供信息标记 **ICLR 2027**，不表述为已录用。用户提供的[代码地址](https://github.com/EIT-NLP/DIAL-OPD)在 2026-10-10 归档时网页与 GitHub API 均返回 404，因此目前没有可克隆的公开 checkout，也没有 main 链接；404 不能区分尚未公开、私有或地址改变。

**认知：分歧大不等于值得监督；两个模型都几乎不支持的 token，也能产生很大的 log-ratio。**

**方法：在相对分歧与绝对概率差之间调节选词分数，每条回答只保留 top 比例 token，然后对保留位置使用原始 OPD 更新。**

- [PDF](paper.pdf)
- [完整数学与证据分析](analysis.md)
- [TOP-D / PowerOPD / DIAL-OPD 对比](../../RoadMap/OPD信号变换对比.md)
- 代码：公开代码地址当前返回 404；未创建代码 checkout。

本次归档与原文分析，不代表实验复现。
