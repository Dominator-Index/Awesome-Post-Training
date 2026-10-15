# SEAD

**认知：** 监督要匹配学生当前能力：太难的前缀和已掌握的 token 都可能浪费训练。

**方法：** 联合 teacher/student entropy 分区选择 loss 或跳过，随阶段从 FKL 过渡到 RKL，配合由易到难课程。

**改动位置：** 题目与训练数据；散度、估计器与梯度路径；token / trajectory 权重与门控；约束、更新幅度与训练调度。

## 关键公式

公式性质：结构示意。

$$
\ell_{t,k}=m_{t,k}\big[\alpha_k D_{\rm FKL}+(1-\alpha_k)D_{\rm RKL}\big]
$$

**适用边界：** 该式抽象三项组件；实际分区和调度以原文为准，不能只独立套其中一项。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2606.28562) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及方法。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
