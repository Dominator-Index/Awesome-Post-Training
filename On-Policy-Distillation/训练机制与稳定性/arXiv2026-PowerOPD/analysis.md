# PowerOPD：Bounded Power Transformation 解读

来源：[论文](https://arxiv.org/pdf/2606.17199) · [代码](https://github.com/EIT-NLP/PowerOPD)。按 arXiv 2026 归档。

**认知：相同比值可能来自有支持的分歧，也可能来自两个极小概率；教学强度应该考虑绝对概率，而不是让 log-ratio 的极值主导更新。**

**方法：将即时教学系数从 $\log q-\log p$ 改成 $q^\alpha-p^\alpha$，保持鼓励/惩罚的符号，原生限制系数幅度。**

## 1. 这篇文章到底在解决什么问题

sampled-token OPD 只采样一个词表动作，log-ratio 系数可能极端，训练出现梯度尖峰。作者关注的是**概率到教学信号的映射是否适合稳定优化**。它不直接检测数学步骤正确性，不训练 critic，也不修复错误前缀。

## 2. Motivation：为什么需要这个方法

标准 log-ratio 只知道老师与学生支持度的比例，不知道双方支持度都多低。$(0.2,0.01)$ 与 $(0.002,0.0001)$ 有相同比例，但后一对落在概率更小的区域。对 sampled-token 估计，极端系数可能对某个罕见动作产生很大的单次贡献。

罕见动作在期望中也罕见，因此“系数无界”并不自动证明实际方差无穷；论文主要用观察到的训练尖峰支持设计动机。

## 3. 背景和已有方法的问题

full-vocabulary KL 通过词表求和降低 sampled-token 噪声，但有额外输出投影与内存成本。对 log-ratio 做 clipping/tanh 能限制系数，但它们仍只依赖比值，无法区分上述两对概率。Z-score 减均值可能翻转原始教学符号。

统一记号：$h_t=(x,y_{<t})$，$p_t=\pi_S(y_t\mid h_t)$ 是 student 对自己采样 token 的概率，$q_t=\pi_T(y_t\mid h_t)$ 是 teacher 在同一前缀上的概率。以下公式采用这个约定；PowerOPD 原文部分段落使用相反的 $p,q$ 命名。

即时 sampled-token OPD 的更新方向为

$$
g=\frac1Z\sum_t\operatorname{sg}[\log(q_t/p_t)]\nabla_\theta\log\pi_\theta(y_t\mid h_t).
$$

$\operatorname{sg}$ 表示系数不反传，$Z$ 是聚合分母。严格地说，这是冻结采样前缀分布时的条件 KL 更新。完整轨迹 KL 的精确梯度还包含早期动作改变未来前缀的项；不要将两个梯度无条件等同。

## 4. 方法总览

完整 student rollout → teacher 逐 token 评分 → 从 log-prob 算幂概率差 → 将差值 detach → 乘 student token log-likelihood 更新。现有生成、评分流程与即时 policy-gradient surrogate 都保留。

## 5. 数学推导和关键公式

### 5.1 同一个函数作用于两概率，再相减

$$
r_t^h=h(q_t)-h(p_t).
$$

原始 OPD 对应 $h(x)=\log x$。若 $h$ 严格递增，$q>p$ 时系数为正、$q<p$ 时为负；若 $h$ 在 $[0,1]$ 上有界，差值也有界。这给出一类可控制符号与幅度的设计。

### 5.2 Box–Cox 与实现中的缩放

$$
h_\alpha(x)=\frac{x^\alpha-1}{\alpha},\qquad
h_\alpha(q)-h_\alpha(p)=\frac{q^\alpha-p^\alpha}{\alpha},\qquad\alpha>0.
$$

$\alpha$ 控制概率区域的敏感度。论文将常数 $1/\alpha$ 视为可吸收进学习率，实际使用

$$
\boxed{r_t^{\rm Power}=\operatorname{sg}[q_t^\alpha-p_t^\alpha]\in[-1,1],\qquad
\mathcal L_{\rm Power}=-\frac1Z\sum_t r_t^{\rm Power}\log\pi_\theta(y_t\mid h_t).}
$$

$Z$ 在查看的官方 trainer 中是有效训练 token 数；系数 detach，student loss log-prob 反传。代码使用 `exp(alpha * teacher_logp) - exp(alpha * student_logp)` 计算幂差。

### 5.3 不要混淆两个极限

$$
\lim_{\alpha\to0^+}\frac{q^\alpha-p^\alpha}{\alpha}=\log(q/p),\qquad
\lim_{\alpha\to0^+}(q^\alpha-p^\alpha)=0.
$$

只有带 $1/\alpha$ 的 Box–Cox 差值收敛到 log-ratio。实现中的 rescaled reward 不会直接收敛成它；想比较需要对应调整学习率。

### 5.4 绝对概率带来的变化

$$
r_\alpha(\epsilon p,\epsilon q)=\epsilon^\alpha r_\alpha(p,q),\qquad
\operatorname{sign}(r_\alpha)=\operatorname{sign}(q-p).
$$

概率同倍下降，原始比值不变，幂差却下降。$\alpha=1$ 为绝对概率差；更大 $\alpha$ 使双方低概率的位置更接近零。$h'_\alpha(x)=\alpha x^{\alpha-1}$：$\alpha>1$ 将敏感度推向高概率区域，但 $0<\alpha<1$ 对低概率变化仍可敏感，不能仅因 reward 有界就宣称映射全局 Lipschitz。

## 6. 为什么这个方法 work

它在保留老师与学生支持差异的方向时，减少 low-low token 对单次更新的极端系数，并通过 $\alpha$ 调整实际学习集中在哪个概率区域。理论能保证系数有界和符号一致，实验才支持特定任务上的稳定性和正确率收益。

有界 reward 不等于参数梯度一定有界，因为 $\nabla\log\pi$ 也进入梯度；符号一致也不等于任务收益一定增加，因为老师可能错。

## 7. 为什么其他方法不行：它们忽略了什么

clipping/tanh 并非数学上不能稳定训练：它们同样可以有界、保符号。准确的区别是它们只看 $q/p$，PowerOPD 还看绝对尺度。在作者设置下后者更好，不足以证明所有 post-hoc 变换必然失败。

full-vocab 与 sampled-token 的计算量、估计方差不同；PowerOPD 本身也改变教学方向，不能把全部改进归结为对同一 KL 的更好无偏估计。

## 8. 这篇文章真正的 contribution

把 teaching reward 的几何作为可设计组件，提出“单调变换后相减”的有界家族。RoadMap 主要是 A/F/W：系数本身改变，token 权重随之改变。它没有构造归一化的 $q^\alpha$ 新 teacher 分布；对整词表幂变换后重新归一化，是另一个算法。

## 9. 实验、消融和证据链

- Qwen3-0.6B-Base/1.7B-Base student，4B/8B teacher，四组合；DeepScaleR，训练 1.5k steps，BF16，learning rate $5\times10^{-7}$，batch 32，rollout cap 1024；结果平均三个运行。
- §5 称评估六个数学 benchmark，AIME24/25、MATH500、AMC23、Minerva、Olympiad；主表另有 GSM8K 列。复用作者均值时应保留其指标与聚合说明，避免自行混用任务集合。
- §6.1 报告 4B→0.6B 的 benchmark 平均 Avg@8 从 19.13 到 25.50，最高 +6.37 点；这是 best PowerOPD variant 的结果，不是固定 $\alpha=1$。
- 与 full-vocab 相比，特定效率设置为 22.14→9.03 秒/step、78.99→60.72 GiB，降低 59.2% 时间与 23.1% peak memory（Table 4，memory batch 8）。不能解释为比 vanilla sampled-token OPD 省这些资源。
- reward 平面图说明绝对尺度差别；梯度轨迹显示 sampled Power 更稳定；不同 $\alpha$ 的结果支持概率区域重新分配。但部分指标的 best-$\alpha$ 不相同，不能声称越大总是越好。

以上是论文报告，本次没有复现。

## 10. 局限和适用边界

固定学习率下，改变 $\alpha$ 同时改变全局 reward 尺度和相对 token 权重。梯度范数缩小本身不能完全区分“更合理地分配监督”和“等效步长变小”。更大 $\alpha$ 也可能导致大多数 token 信号接近零、浮点下溢或遗漏低概率有效纠错。

实验主要是数学、同家族模型，代码生成推广尚待验证。Appendix C 用序列 KL 记号推到即时局部 surrogate 时，没有充分保留轨迹分布导数；本文按局部训练方向理解，完整轨迹梯度的 return-to-go 不能省略。

## 11. 总结：完整故事线

log-ratio 会把低支持分歧放大；PowerOPD 直接更换概率到更新系数的映射，用单调有界幂差稳定教学。它与 TOP-D 共享“不要让原始教学信号极端”的动机，但 TOP-D 插值分布后取比值，PowerOPD 在标量概率上变换后相减。

代码在 `/usr/data/PowerOPD`，本目录 [main](main) 为符号链接；具体见 `opd_dynamics/trainer.py`。对照阅读：[OPD 信号变换对比](../../RoadMap/OPD信号变换对比.md)。
