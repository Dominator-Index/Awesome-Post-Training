# RWOPD / NL-to-SVA

**认知：** 语法像标准答案还不够，应让真正性质等价的 SVA 决定哪些监督值得用。

**方法：** 用 SymbiYosys+Z3 的性质等价检查器筛选学生 rollout，再做奖励加权 FKL。

**改动位置：** 奖励、return、baseline 与 critic；token / trajectory 权重与门控。

## 关键公式

公式性质：结构示意。

$$
L=\mathbb E_{y\sim S}\left[w_{\rm PEC}(y)\sum_tD_{\rm KL}(T_t\|S_t)\right]
$$

**适用边界：** 奖励权重与验证器可用性相关；这是 rollout 加权，不等于 AOPD 动作 advantage。

[本地PDF](paper.pdf)

[原始来源](https://arxiv.org/abs/2605.13501) · [RoadMap](../../RoadMap/README.md)

整理依据：本地 PDF / 阅读笔记；摘要及 verifier-weighted objective。这是已有阅读材料的归纳；不代表本次重新完成全文精读或实验复现。
