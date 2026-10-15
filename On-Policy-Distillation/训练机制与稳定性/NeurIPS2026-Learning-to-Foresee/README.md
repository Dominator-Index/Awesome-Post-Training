# EffOPD / Learning to Foresee

**认知：** 如果早期更新方向已经接近最终方向，可以沿当前方向多走一点来加速。

**方法：** 分析模块分配与低秩更新方向，采用自适应 extrapolation 步长。

**改动位置：** 约束、更新幅度与训练调度。

## 关键公式

公式性质：结构示意。

$$
\theta_{k+1}=\theta_k+\alpha_k\Delta\theta_k
$$

**适用边界：** 外推倍数必须控制；稳定方向的经验现象不是全局收敛保证。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.11739) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及 EffOPD 方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
