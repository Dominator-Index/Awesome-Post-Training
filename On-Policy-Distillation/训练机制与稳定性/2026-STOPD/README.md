# STOPD：随机截断 rollout，并补偿后段监督的缺失

**Does On-Policy Distillation Need to Complete Every Rollout?**

- 作者：Zhengxin Zhang、Maosen Tang、Ning Wang、Changhao Li、Jingren Zhou、Bo Dai、Sainyam Galhotra、Claire Cardie。
- [官方仓库](https://github.com/YouAreSpecialToMe/STOPD)，仓库内提供原生 PDF `paper/stopd.pdf`，本目录保存为 `paper.pdf`，43 页。
- 官方 README 尚未给出 arXiv 编号，未确认会议状态，因此使用 `2026-STOPD`，没有自行标注 arXiv 或 ICLR。
- 代码 `/usr/data/STOPD`，本目录 `main` 为软链接；快照 commit 和 PDF 哈希见 `SOURCE.txt`。
- [数学与方法解读](analysis.md)，并已纳入 [RoadMap](../../RoadMap/README.md) 的长度、采样和归一化组件。

它并非仅在固定位置截断：随机选择停止 block，让每个 token 都有非零保留概率，再用逆概率加权补偿。它不需要 critic 预测未生成后段，也不需要拿到本条轨迹的最终奖励。但无偏性严格针对局部梯度之和，使用滞后长度分母后，归一化梯度仍有尺度差异。

本次仅归档与静态阅读；未安装依赖或运行仓库测试、训练。
