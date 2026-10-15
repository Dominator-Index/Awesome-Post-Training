# CorrGRPO：多奖励的相关性归一化

完整题名：**CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning**。作者：Wenbin Hu、Huihao Jing、Haochen Shi、Yuxuan Liu、Haoran Li、Yangqiu Song。

- [论文页面](https://arxiv.org/abs/2609.36820) · [本地 PDF](paper.pdf)，v1，2026-09-29，40 页。
- [官方代码](https://github.com/HKUST-KnowComp/CorrGRPO)；[main](main) 指向 `/usr/data/CorrGRPO`。
- 按 arXiv 年份归档，未确认会议状态。这是多奖励 RL 方法，作为与 OPD 联合优化有关的阅读资料保存在此。

## 核心问题

多个奖励分别衡量正确性、效率、工具参数或安全性。普通 GRPO 对奖励求和，再用组内总奖励标准差归一化。总方差包含所有奖励的协方差；数值尺度大的奖励可能主导分母，使小尺度奖励之间的相关关系在归一化中几乎不起作用。

CorrGRPO 保留总奖励的中心化分子，将分母中的协方差换成 Pearson 相关系数。

## 关键公式

设 $R_i^l$ 为同一题目第 $i$ 条 rollout 的第 $l$ 个加权奖励，$\bar R^l$ 为组均值。固定奖励权重已吸收进 $R^l$。

$$
A_i^{\mathrm{GRPO}}=
\frac{\sum_l(R_i^l-\bar R^l)}{\sqrt{\sum_{l,m}\widehat{\operatorname{Cov}}(R^l,R^m)+\epsilon}}.
$$

因为总和的方差等于协方差矩阵所有元素之和，所以分母同时受奖励依赖性和尺度影响。CorrGRPO 改为：

$$
\hat\rho_{lm}=\frac{\widehat{\operatorname{Cov}}(R^l,R^m)}{\hat\sigma_l\hat\sigma_m},
\qquad
A_i^{\mathrm{CorrGRPO}}=
\frac{\sum_l(R_i^l-\bar R^l)}{\sqrt{\sum_{l,m}\hat\rho_{lm}+\epsilon}}.
$$

每个非零方差奖励的相关矩阵对角项为 1；其他项描述奖励间依赖。零方差奖励对应行列设为零，包括对角项。实际代码对相关系数总和取非负下限，再加 epsilon，防止浮点误差。

## 为什么可能有效，不能解决什么

在中心化总奖励固定时，强正相关增大分母、减小 advantage；弱相关或负相关使分母较小。更换分母后，这种调节不再被单个奖励的大方差直接主导。

**分子仍保留原奖励尺度和权重。**因此它没有让全部奖励对最终梯度等量贡献，也不是对每种奖励分别标准化后相加。共同缩放奖励仍会缩放 CorrGRPO 的 advantage，这与普通总奖励标准差归一化不同。

它主要改变同组 rollout 的更新幅度，不自动识别错误 token，也不解决 teacher 可靠性或逐步信用分配。所有奖励组内恒定时，中心化分子仍为零。高度负相关、小组样本估计误差或分母接近零也需要考虑数值稳定性。

## 实验结果

作者在代码生成、工具调用、agent 安全上评估 0.5B–8B 模型。官方仓库列出的代表性结果包括：Qwen2.5-Coder-7B 四基准代码 Pass@1 均值从 GRPO 的 47.28% 到 51.49%；Qwen2.5-7B 工具调用 all-exact 从 63.38% 到 67.61%；agent 安全 joint accuracy 从 31.38% 到 47.88%。这些是选定比较，不表示所有指标或模型都提高，未独立复现。

## 代码归档说明

仓库按用户约定克隆到 `/usr/data`，论文目录用软链接访问。克隆时跳过 Git LFS 大文件自动下载，因此 LFS 数据可能仅为指针；运行需要的数据应按官方说明另行获取。未安装依赖或启动训练。

## 对 OPD 研究的联系

如果 OPD 与含多个奖励的 GRPO 联合训练，它可以作为奖励分支的 advantage 归一化选项。不能未经实验就认定其同样适用于蒸馏 loss，或认定它会改善 teacher token 监督。
