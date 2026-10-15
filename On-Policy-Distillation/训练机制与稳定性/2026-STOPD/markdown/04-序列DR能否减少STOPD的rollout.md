# 04｜序列 DR：能否真的减少 student rollout？

**认知：未来前缀没生成时，不能为每个未来 token 单独补监督；应该在已观察前缀上预测剩余贡献，再校正每个已观察 chunk 带来的预测误差。**

**方法：将剩余预测写成 telescoping sum，对实际看到的增量残差做 survival IPW。**

以下是本笔记的推导，不是 STOPD 已有定理。思想联系 [Jiang–Li 2016 sequential DR](https://proceedings.mlr.press/v48/jiang16.html) 和 [随机 telescoping](https://arxiv.org/abs/1905.07006)；目标特化为 OPD 的局部梯度和，需要验证其工程价值。

## 1. chunk 表述与停止概率

将最多 $L$ 个 token 划成 $J$ 个 chunk。$H_j$ 为生成第 $j$ 个 chunk 前的可见状态，$d_j$ 是这个 chunk 的 loss gradient 和，$H_{j+1}$ 是其结束状态，$j=0,\ldots,J-1$。

自然 EOS 后进入吸收态，后续 $d_j=0$。不强迫 EOS 之后继续生成。完整目标是

$$
G=\sum_{j=0}^{J-1}d_j.
$$

给定到达 $H_j$，以已知 $q_j(H_j)>0$ 继续，定义 $C_j$ 为第 $j$ 个 chunk 被生成的指示，

$$
\Pi_j=\prod_{l=0}^{j}q_l(H_l).
$$

$\Pi_j$ 是累计到达概率，不是当前继续概率 $q_j$，也不是新旧 student 策略比。初始 chunk 可设 $q_0=1$。随机决定须在生成下一 chunk 前作出，不能使用尚未观察的真实梯度或结局。

STOPD 独立均匀 stop 是特例：当 $L=JK$，第 $j$ 个 chunk 的 $\Pi_j=(J-j)/J$。也可直接从原文 $p_t$ 得到 block 的累计覆盖概率。

## 2. 剩余梯度预测

令 $U_j(H_j)$ 是廉价的**剩余梯度向量预测**，$U_J=0$；自然 EOS 状态上也令 $U=0$。理想情况为

$$
U_j^*(H_j)=\mathbb E\left[\sum_{l=j}^{J-1}d_l\mid H_j\right].
$$

预测的是未完成 OPD 更新，不是“最终答对概率”。预测模型在本轮停止抽样前固定，且只使用可见输入。数百万维参数梯度难以直接建模，这是实际难点。

## 3. 构造残差估计器

定义

$$
\delta_j=d_j+U_{j+1}(H_{j+1})-U_j(H_j).
$$

$d_j$ 为真实 chunk 贡献；前后 $U$ 表示对剩余贡献认识的变化。完整轨迹有恒等式

$$
U_0(H_0)+\sum_{j=0}^{J-1}\delta_j=G.
$$

于是可构造

$$
\boxed{\widehat G_{\rm seq}=U_0(H_0)+\sum_{j=0}^{J-1}\frac{C_j}{\Pi_j}
\left[d_j+U_{j+1}(H_{j+1})-U_j(H_j)\right].}
$$

未生成 chunk 不计算残差。最后一个已生成 chunk 的 $H_{j+1}$ 已可见，因此可以计算其边界预测，即使之后立即停止。

## 4. 哪一部分能证明

固定完整潜在轨迹及预先固定的预测函数，假设停止 coins 在给定历史后独立于未观察未来，且完整与截断的生成分布可耦合。此时

$$
\mathbb E_C[C_j/\Pi_j\mid\text{完整潜在轨迹}]=1,
$$

$$
\boxed{\mathbb E_C[\widehat G_{\rm seq}\mid\text{完整潜在轨迹}]=G.}
$$

预测错也不会改变该设计期望，因为误差会由 observed residual 校正。$U=0$ 时退化为 chunk 形式 STOPD/IPW。

如果用错的 $\widehat\Pi$，但 $U=U^*$，则 $\mathbb E[\delta_j\mid H_j]=0$。在预测/权重预先固定、可忽略停止和有限矩等条件下，残差的总体均值仍为零。这是这里的第二稳健分支，但它是**对轨迹分布取期望**的结论，不是对每条固定轨迹恢复精确梯度。

序列 DR 的正规统计结果需要更完整的 nuisance 条件与数据分离；不能从这里直接推出复杂训练闭环的所有渐近性质。

## 5. loss 和梯度的实现连接

可以先构造可微局部 surrogate $\ell_j(\theta)$，再构造廉价可微剩余 surrogate $M_j(H_j;\theta)$，令

$$
U_j=\nabla_\theta M_j(H_j;\theta_k).
$$

保持采样 token 固定，并冻结预测器参数及停止权重，则

$$
\widehat L_{\rm seq}(\theta)=M_0(\theta)+\sum_j\frac{C_j}{\Pi_j}
[\ell_j(\theta)+M_{j+1}(\theta)-M_j(\theta)]
$$

的梯度对应上面的 $\widehat G_{\rm seq}$。所有 $M$ 必须定义为同一套潜在轨迹上的局部可微函数，终端 $M_J=0$，才能 telescoping；不能只在停止时随意插入一个不相容的补项。

**不能把 $M$ 全部 detach 掉**，否则其梯度为零，控制变量没有作用。冻结 predictor 参数和停止机制，不代表切断 actor 到 surrogate 的可微路径。

## 6. 最容易出错的实现

- 只学习未来 KL 数值，并不保证 $\nabla M$ 近似未来梯度。value prediction 准，不代表 gradient prediction 准。
- 只在 stop 后直接加一次 critic prediction，没有 residual correction，会变成有模型偏差的 bootstrap。
- 只除当前 $q_j$，漏掉之前的累计 survival probability，会产生偏差。
- 有些 chunk 对齐或 task return 使用终局未来，截断时不可见；这时 $d_j$ 并非原定理要求的局部贡献。
- 当前 batch 先训练预测器再用其校正同一批停止样本，可能引入依赖性；使用 lagged predictor 或按完整 rollout 分折。
- 还要处理 STOPD 的长度分母。分子 DR 并未让未知 $N$ 自动可见，使用 $C_k$ 仍有 $N/C_k$ 的尺度差。

## 7. 是否值得做

**数学上有明确连接，实现上比 teacher-query DR 更难。** 最有价值的实证问题是：一个足够便宜的、梯度相关的剩余预测，能否以更少完整 rollout 降低 IPW 的噪声和 wall time？

若只加入昂贵 critic 而没有降低总计算，就没有解决 STOPD 的痛点。若只是重现经典 sequential DR 式，也没有建立 OPD 新贡献。建议先检验固定模型上的残差方差，再训练。
