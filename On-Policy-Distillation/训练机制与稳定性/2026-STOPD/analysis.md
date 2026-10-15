# STOPD：Does On-Policy Distillation Need to Complete Every Rollout? 解读

依据官方仓库内的 `paper/stopd.pdf`（43 页）和源码。本地归档为 `paper.pdf`，官方尚未提供 arXiv 编号，不据此认定会议录用。

## 1. 最值得记住的认知

**局部 OPD 监督不必等回答完成才可取得；随机截断后的后段覆盖减少，可以通过逆概率加权补偿。**

方法是提前抽取一个随机停止 block，只生成到该 block 或自然 EOS，让 teacher 打分保留的前缀，对每个 token 乘上其被保留概率的倒数。

这里最重要的不是“把答案截短”，而是 **每个位置保留非零概率，并纠正不均匀的覆盖**。

## 2. 为什么固定截断与只 mask loss 不够

- 完整 rollout 后才 mask 一些 loss，没有省掉已发生的 actor generation。
- 固定在同一长度停止，后段位置永远没有监督。它可以作为不同训练目标，但一般不是完整 OPD 梯度的无偏估计。
- 随机停止但不重加权，早期 token 更经常出现，依然改变了原局部监督的相对权重。

STOPD 选择随机停止加 IPW，把 actor decoding、teacher scoring 和 student update 计算都减少，同时尽量保留完整局部梯度的期望。

## 3. 数学形式：生成、保留概率和加权

使用论文的 1-based token 索引。最大长度 $L$，block 宽度 $K$，完整生成本会有 $T_i\le L$ 个 token。独立抽 anchor：

$$
A_i\sim\operatorname{Uniform}\{0,\ldots,L-1\},\qquad
S_i=\min\{K(\lfloor A_i/K\rfloor+1),L\},\qquad
M_i=\min(T_i,S_i).
$$

$S_i$ 是随机停止点，$M_i$ 是实际保留长度；如果模型自然提前结束，不会为了凑满 stop 而继续生成。

$$
\boxed{p_t=P(S_i\ge t)=\frac{L-K\lfloor(t-1)/K\rfloor}{L},\qquad w_t=1/p_t.}
$$

$p_t$ 是位置抽样概率，不是 teacher 或 student 对该 token 的概率。后段 $p_t$ 较小，所以保留时权重更大。若 $K\mid L$，最大权重为 $L/K$；相比按单 token 停止的最大权重 $L$，block 限制极端权重，但也会多生成一些 token。

例如 $L=8,K=2$：block 保留概率为 $1,3/4,1/2,1/4$，IPW 为 $1,4/3,2,4$。

## 4. 哪个量是无偏的？

令 $h_{i,t}=\nabla_\theta\ell_{i,t}$ 是 token 的局部梯度，$I_{i,t}=\mathbf1[S_i\ge t]$。固定完整轨迹和局部 teacher feedback：

$$
G=\sum_{i=1}^B\sum_{t=1}^{T_i}h_{i,t},\qquad
\widehat G=\sum_{i=1}^B\sum_{t=1}^{T_i}\frac{I_{i,t}}{p_t}h_{i,t}.
$$

因为 $\mathbb E_S[I_{i,t}]=p_t$，有：

$$
\boxed{\mathbb E_S[\widehat G]=G,\qquad
\mathbb E_S\left[\widehat N\right]=N,\quad
\widehat N=\sum_{i,t}\frac{I_{i,t}}{p_t}.}
$$

这是 §3 与 Appendix A 的核心证明：线性加总恢复梯度和、token 数的期望。它没有要求某条截断回答最终正确，也不估计未生成未来的 task reward。

条件包括：随机 stop 与完整轨迹、teacher 随机性独立；相同前缀的 next-token 分布不因为剩余 budget 不同而改变；保留位置的局部 loss 不依赖被删除的未来。这些条件之外不能直接套用无偏结论。

## 5. 分母为什么不能随手用本批截断 token 数？

Vanilla token 平均梯度为 $g^V=G/N$。本批被截断，完整 $N$ 不可见；STOPD 在当前 stop 抽取前固定分母 $C_k$：

$$
\widehat L_k=\frac1{C_k}\sum_{i,t\le M_i}\frac{\ell_{i,t}}{p_t},\qquad
\widehat g_k=\widehat G_k/C_k.
$$

先执行 $W$ 个完整 warmup steps，第一步 STOPD 用最后 warmup 的真实 token 数，之后用上一批的 IPW count：

$$
C_{W+1}=N_W,\qquad C_k=\widehat N_{k-1}\quad(k>W+1).
$$

这让 $C_k$ 相对当前停止随机性固定。权重和分母也都不参与求导。

$$
\boxed{\mathbb E_S[\widehat g_k]=\frac{G_k}{C_k}
=\frac{N_k}{C_k}g_k^V.}
$$

所以归一化梯度仍有尺度差异。论文明确分解：

$$
\widehat g_k-g_k^V
=\frac{\widehat G_k-G_k}{C_k}
+\frac{N_k-C_k}{C_k}g_k^V.
$$

第一项是随机截断误差，第二项是 lagged normalizer 的尺度误差。两个随机估计量分别无偏，不能推出它们的比值也无偏。

## 6. 为什么可以 work？理论保证到哪里

block 控制最大 IPW；独立的 rollout 停止点让 batch averaging 缓解截断噪声。Appendix A 给出方差与 lagged-count 浓缩分析；论文还对相同初始参数和优化器状态下 SGD、Adam、Muon 的一步更新差异给出条件化界。

这些是局部梯度与一步更新误差结果，不是“随机截断后的长期训练与完整训练逐步一样”，也不保证实际任务准确率总不下降。Adam/Muon 非线性处理梯度，因此梯度和无偏不能直接推出最终参数更新无偏。

## 7. 实验支持什么

主要设置使用 $L=16384,K=128$，经过完整 warmup 后再切换 STOPD。跨模型规模、数学、代码、指令遵循和优化器比较：

| 主设置领域 | Vanilla 每步时间 | STOPD 每步时间 | 加速 |
|---|---:|---:|---:|
| 数学 | 941s | 513s | 1.8× |
| 代码 | 566s | 260s | 2.2× |
| 指令遵循 | 104s | 80s | 1.3× |

论文报告生成 token 减少 25–48%，准确率与 Vanilla 相近，不能写成每个指标完全相同或全部提高。长任务的生成吞吐也有变化，所以速度收益不只等于 token 数比例。具体硬件、服务、并行和长度分布会影响效果。

## 8. 与 AC2、AOPD 和其他长度方法的关系

| 方法 | 为什么可以少做完整 rollout | 代价 / 条件 |
|---|---|---|
| STOPD | 已有局部 teacher feedback 可立即训练，随机缺失用 IPW 补偿 | 局部性、独立停止、IPW 方差和长度分母 |
| AC2 | 用末端 critic value 估计未生成后半段收益 | 需要训练与校验 critic，并保留完整结果通路 |
| 固定截断 OPD | 只训练前缀目标 | 后段零覆盖，一般改变完整目标 |
| 只 mask 完整 rollout 的 loss | 只少做部分监督 | 不自动节省已经发生的生成 |
| AOPD | 用终局 reward 判断 teacher 动作价值 | 估计器仍需本条终局 reward，不能直接截断后保持原式 |

将 STOPD 与 AOPD、RWOPD 或 reward-aware 筛选组合时，局部 loss 会依赖终局 reward。未来被删掉后该量不再可见，需要新的奖励估计或抽样推导。不能用“STOPD 支持任意 per-token loss”忽略原定理的局部性条件。

作者讨论它对 OPSD、multi-teacher OPD 和 SFT 的扩展潜力：保留 teacher 所需完整参考信息，只截断 student rollout 或参考 loss 的计算。这些延伸应区分作者讨论与已经完成的实验。

## 9. 代码对应与限制

- `src/stopd/core.py`：budget、token weights、loss reduction、lagged denominator 和 checkpoint 状态。
- `src/stopd/truncation.py`：stop 与 inclusion probability。
- `src/stopd/normalizer.py`：逐步更新分母。
- `src/stopd/integrations/trl.py` 与 verl hooks：框架接入。

代码使用 0-based response position，所以 $\lfloor t/K\rfloor$ 与论文 1-based 的 $\lfloor(t-1)/K\rfloor$ 是索引约定差异。分布式 microbatch / data-parallel reduction 必须保持全局 sum 与同一个 $C_k$，不能被框架额外平均。

普通批量 `transformers.generate` 可能等最长序列结束，单行较早停止不一定带来对应 GPU 时间节省；真正减少解码与 teacher/update 时间需要服务实现配合。仓库 README 对支持的 vLLM 模式与集成边界已有说明。

## 10. 总结

**STOPD 改的是 rollout 长度、位置权重和 loss 分母；原有局部蒸馏 loss 保持不变。** 它的价值在于用统计补偿减少采样计算，而不必训练一个预测未来的 critic。能否保留目标和节省时间，分别取决于估计条件与系统实现。

本次读取论文主要方法、理论及附录相关结论并静态核对源码，未运行仓库测试或训练。
