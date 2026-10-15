# DuoOPD：Learning from Joint Teacher–Student Outcomes for Multi-Task On-Policy Distillation 解读

- 论文：[arXiv:2609.33711v1](https://arxiv.org/abs/2609.33711v1)，2026-09-27；本地 [PDF](paper.pdf)。按 arXiv2026 归档，不推定会议状态。
- 作者：Ao Yu、Weibo Gao、Heng Zhou、Linan Yue、Rui Li、Suyi Liu、Yu Yan、Yizhong Zhang、Qi Liu。
- [官方代码](https://github.com/YongYuanDeAo/DuoOPD)；[项目页](https://yongyuandeao.github.io/DuoOPD/)。
- 阅读覆盖正文、Appendix A–D，包括实现、成本和失败边界；本次未复现训练。

> **High-level 认知：学生已经做对时，不应因老师不喜欢它的写法而惩罚它；学生做错时，应该用老师可验证的成功帮助定位纠错。谁答对，决定谁提供可信的支持。**
>
> **方法：student 的 verifier 决定监督正负号；teacher/student 的四种联合结果决定 token 权重来自普通评分、参考答案条件评分，还是同任务共享正权重。**

## 1. 这篇文章到底在解决什么问题

标准 sampled-token OPD 使用 teacher/student 的 log probability ratio，决定强化或压低学生生成的 token。但是 teacher 的偏好、teacher 在这道题上能否答对、student 的回答能否通过检查，是三件不同的事。

学生可能最终答对，老师却更不喜欢这个表达；老师也可能独立答错，却仍对学生的正确回答提供细粒度偏好。多任务训练进一步放大这个问题：师生能力的相对关系随任务和样本变化，不能只按“老师整体更强”处理每条回答。

核心问题：**当 teacher 与 student 的结果不一致时，该由谁决定学习方向，又该如何分配监督力度？**

## 2. Motivation：为什么需要这个方法

只按学生结果改正负号还不够。

1. **老师对、学生错**：老师在独立完成题目时会做，但在学生的错误前缀上评分时，仍可能被错误思路带偏。论文把老师自己的已验证回答加入老师的评分上下文。
2. **老师错、学生对**：即使强制正反馈，失败老师仍可能把关键正确 token 的权重压到几乎零。论文取消这个子集内的逐 token 权重差别，用同任务共享系数强化完整学生回答。

这与你之前担心的“teacher 基于 student 错误前缀继续教”直接相关：DuoOPD 保留学生原始错误前缀，却让评分老师额外看到一个成功参考，降低前缀误导。它没有把学生 rollout 改写成正确答案。

## 3. 背景和已有方法的问题

### 3.1 原始 OPD：偏好系数不保证与正确性一致

原文 Eq.1：

$$
d_{0,i,t}=\log\pi_T(o_{i,t}\mid q_i,o_{i,<t})-\log\pi_{\theta_{\rm old}}(o_{i,t}\mid q_i,o_{i,<t}).
$$

$q_i$ 是题目，$o_i$ 是旧学生生成的回答。老师只给学生实际抽到的 token 评分，不是在此阶段重新生成 teacher answer。$d_0>0$ 在 surrogate 中强化该 token，$d_0<0$ 抑制它。

原文 Eq.2 使用**先对每条回答的有效 token 求均值，再对回答求均值**：

$$
\widehat g_{\rm OPD}=-\frac1B\sum_{i=1}^B\frac1{T_i}\sum_{t=1}^{T_i}d_{0,i,t}\nabla_\theta\log\pi_\theta(o_{i,t}\mid q_i,o_{i,<t}).
$$

这是 loss gradient，优化器做梯度下降；与上升方向的 policy gradient 相差负号。$d_0$ 冻结，梯度只走学生 log probability。它与 STOPD 原文的全 batch token-count 分母不是同一个归一化。

### 3.2 OPDVR：正确回答只保留正反馈，错误回答只保留负反馈

原文 Eq.3：

$$
A_{\rm gate}(d,r_S)=\begin{cases}
\max(d,0),&r_S=1,\\
\min(d,0),&r_S=0.
\end{cases}
$$

$r_S$ 是学生是否通过 verifier。它修正方向，但与方向冲突的 token 被置零，也没有根据 teacher 自己是否成功改变 teacher context。

### 3.3 不要误读“OPD 压低正确回答”

论文在 DuoOPD 的训练轨迹上，观察到正确回答的原始 $d_0$ 平均为负。这是局部反馈统计；不能由此推出一次共享参数更新必然降低每条正确回答的概率，或 OPD 在所有任务上损害正确率。论文 OPD baseline 本身也明显改善初始学生。

## 4. 方法总览：四种结果分别怎么处理

训练前，每个训练问题缓存**一条**独立生成的 teacher response $o_i^T$，经同任务 verifier 得到 $r_{T,i}$。训练时学生每步重新生成一条回答，得到 $r_{S,i}$。

| Teacher | Student | 目标 | 反馈来源 |
|---|---|---|---|
| 对 | 对 | 强化学生成功 | 原始 context 的正 Softplus 权重 |
| 对 | 错 | 抑制学生失败，并让纠错更有依据 | teacher 看到成功参考后，重新评分学生错误轨迹 |
| 错 | 对 | 保留学生自己的成功 | 同任务、同联合结果子集的共享正权重 |
| 错 | 错 | 抑制失败轨迹 | 原始 context 的负 Softplus 权重 |

teacher outcome 指这条缓存回答是否通过检查，不等于老师对这道题所有可能采样的能力。学生输入始终不含参考。

## 5. 数学推导和关键公式

### 5.1 方向和力度分离：原文 Eq.4

$$
\boxed{A_{i,t}=\sigma_i m_{i,t},\qquad\sigma_i=2r_{S,i}-1.}
$$

学生对，$\sigma=+1$；学生错，$\sigma=-1$。$m>0$ 决定力度：

$$
m_{i,t}=\begin{cases}
\operatorname{sp}(\sigma_i d_{0,i,t}),&r_{T,i}=r_{S,i},\\
\operatorname{sp}(-d_{T,i,t}),&r_{T,i}=1,\ r_{S,i}=0,\\
\bar m_{k(i)},&r_{T,i}=0,\ r_{S,i}=1.
\end{cases}
$$

$k(i)$ 是任务类别，$\operatorname{sp}(x)=\log(1+e^x)$。正负方向由 verifier 控制，不由 $d$ 的符号控制。

### 5.2 Softplus 相比 gating 加了什么：原文 Eq.5

$$
A^+(d)=\operatorname{sp}(d)=\max(d,0)+\log(1+e^{-|d|}),
$$

$$
A^-(d)=-\operatorname{sp}(-d)=\min(d,0)-\log(1+e^{-|d|}).
$$

这是 gate 的原贡献，再加一个方向正确的非零项。$d=0$ 时幅度为 $\log2\approx0.693$。但 $d$ 与 verified direction 强烈冲突时，权重仍可非常小；数学上的非零不等于有效强化。

例如学生回答正确、$d=-3$：OPD 负反馈为 $-3$；gate 为 0；Softplus 为约 $0.0486$。学生错误、$d=3$：gate 为 0；negative Softplus 为约 $-0.0486$。

### 5.3 老师对、学生错：原文 Eq.6

$$
\boxed{d_{T,i,t}=\log\pi_T(o_{i,t}\mid q_i,o_i^T,o_{i,<t})-\log\pi_{\theta_{\rm old}}(o_{i,t}\mid q_i,o_{i,<t}).}
$$

只有 teacher 的 context 多了 verified reference。仍评分学生原回答的同一组 token，使用 $A=-\operatorname{sp}(-d_T)$。若 reference 使某个错误 token 的 teacher probability 更低，$d_T$ 更负，对它的抑制更强。

**例：学生写 $2+3=6$，teacher 的独立参考写 $2+3=5$。** reference-conditioned teacher 可以降低对学生 token `6` 的评分。但是它并没有修复学生已出现的 `6`，也不保证后续每个 token 都获得语义上正确的纠错信号。

### 5.4 老师错、学生对：原文 Eq.7

在完整 rollout batch 内定义 $\mathcal I_k=\{i:k(i)=k,r_{T,i}=0,r_{S,i}=1\}$，非空时：

$$
\boxed{\bar m_k=\operatorname{sg}\left[\frac1{|\mathcal I_k|}\sum_{i\in\mathcal I_k}\frac1{T_i}\sum_{t=1}^{T_i}\operatorname{sp}(d_{0,i,t})\right].}
$$

每条回答先求 token 平均，再求同任务子集平均。这个子集内，所有有效 token 共用 $\bar m_k>0$。它消掉失败 teacher 在子集内部对 token/response 的力度差异，**但任务尺度仍来自 teacher log-ratio**，并未完全去掉 teacher 的影响。

若仅一条回答落入该子集，仍可计算自身 token 平均；若子集为空，就没有需要该权重的回答。必须在 microbatch 切分之前求全局子集均值，不能每个 microbatch 各算一个值。

### 5.5 最终 surrogate：原文 Eq.8

$$
\boxed{\mathcal L_{\rm DuoOPD}=-\frac1B\sum_i\frac1{T_i}\sum_t\operatorname{sg}[A_{i,t}]\log\pi_\theta(o_{i,t}\mid q_i,o_{i,<t}).}
$$

每步冻结权重，只有学生 log probability 接收梯度。student rollout 和 student update 都不看参考；不是在 teacher answer 上做 SFT。学生成功分支更像对已验证学生回答的加权 imitation，失败分支是负权重的 likelihood 更新。

### 5.6 实现为什么用 policy ratio 也能对应此式

Appendix B.2 使用一个 rollout batch、一个 minibatch、一个 epoch。on-policy branch 的 denominator 为 detached current log probability。令 $\ell=\log\pi_\theta(o_t\mid c_t)$：

$$
u=e^{\ell-\operatorname{sg}[\ell]}=1,\qquad
\nabla_\theta[-\operatorname{sg}[A]u]=-\operatorname{sg}[A]\nabla_\theta\ell.
$$

因此 actor ratio loss 与 Eq.8 在此设置下梯度相同，scalar loss 值可以不同。更换成多 epoch、旧策略 denominator 或额外 clipping 后，不能自动沿用这个等价性。

## 6. 为什么这个方法 work

**方向机制**：错误回答不会因 teacher 相对喜欢某个 sampled token 就得到正系数；正确回答不会因 teacher 不喜欢写法就得到负系数。这是 surrogate 系数层面的保证，不是最终正确率保证。

**参考机制**：独立成功参考让 teacher 评分时不必完全沿学生错误思路继续，提供有依据的 suppression allocation。

**共享机制**：失败 teacher 对正确答案的关键选项可能给很弱的正权重，共享后使完整成功回答都得到同任务尺度的 reinforcement。

论文没有给出一般性收敛、无偏保留原始 OPD 梯度或语义 token identification 定理。它有意改变监督目标；机制主要由消融与案例支持。

## 7. 为什么其他方法不行：它们忽略了什么

| 对照 | 处理了什么 | DuoOPD 进一步处理什么 |
|---|---|---|
| 原始 OPD | teacher/student 局部分布匹配 | 偏好方向可能违背 verified success |
| OPDVR | 按 student success 门控 | 加入 teacher outcome，改变评分上下文与权重分配 |
| OPSD / privileged-context OPD | teacher 可看参考 | 条件性地给外部 teacher 自己的成功回答；只用于 T对S错 |
| DN-MOPD | 跨领域 feedback 尺度 | 同领域内不同 teacher/student outcomes 的可信来源 |
| TOP-D / PowerOPD / DIAL-OPD | 改分布/反馈强度等 | 本文用 verifier 改系数方向，不是两分布插值 |
| STOPD / DR | 计算预算与缺失信号估计 | 本文改变哪个方向值得强化，不是校正 missingness |

这些是作用位置的区别，不是所有对照都失败。主实验没有与 STOPD、DR、TOP-D 等全部做直接对照。

## 8. 这篇文章真正的 contribution

单独“正确强化、错误惩罚”已有 OPDVR 等背景；单独 privileged context 也已有工作。本文的具体贡献是**四种联合 outcome 下选择可信支持来源**，以及两种 disagreement 的不同处理。

其 multi-task 主要体现在 verifier 和 shared-weight scope，不是训练多个教师，也不估计四种 outcome 的概率 $\rho_k$，没有动态改变任务采样比例。

## 9. 实验、消融和证据链

### 9.1 设置

Qwen3-4B→Qwen3-0.6B、Llama-3.1-8B-Instruct→Llama-3.2-3B-Instruct。60 次 student update、batch 48，每任务16题，一题一条 rollout，上限1024token；AdamW，学习率 $3\times10^{-6}$，6步warmup。训练与评估 temperature1，不做top-p/top-k截断；Qwen3 thinking关闭。每方法3次训练。

主 mixture 为 biology/chemistry/physics；额外为 materials/chemistry-understanding/physics-calculation，以及 physics/IFEval/MBPP。avg@8 是八次回答的**平均成功率**，不是 pass@8；Macro 为三个任务等权平均。

### 9.2 主要结果

| 任务组合 / 模型 | OPD Macro | 最强对照 Macro | DuoOPD Macro |
|---|---:|---:|---:|
| B/C/P，Qwen | $64.39\pm0.45$ | EOPD $65.81\pm0.54$ | $66.97\pm0.26$ |
| B/C/P，Llama | $67.28\pm1.25$ | OPDVR $68.68\pm1.37$ | $73.26\pm1.04$ |
| science，Qwen | $62.33\pm0.39$ | ExOPD $62.62\pm0.01$ | $63.48\pm0.37$ |
| heterogeneous，Qwen | $47.22\pm0.36$ | OPDVR $47.83\pm0.35$ | $49.24\pm0.64$ |

数据来自 Table1–2、Appendix Table8，单位为百分比。Qwen/Llama 主 mixture 比 OPD 分别高2.58/5.98个百分点。heterogeneous 的 MBPP 为32.69 vs OPD31.41，IFEval53.30 vs50.95；但 IFEval 初始学生54.99，说明只是较好保留能力，仍没有超过初始水平。science 也不是所有领域都是本文最高。

### 9.3 哪个组件带来收益

Table4：完整66.97；无sharing66.26；无reference65.48；两者都无64.56，接近OPDVR64.83。支持**两种 disagreement 机制**是主要增益来源，单独 Softplus direction 不足以解释总收益。

Table5：within-task 比 across-task sharing 稍好，三个 mixture 分别差0.23/0.26/0.35个百分点，差距小，不应宣传为很大的普遍优势。

Appendix D 的固定trajectory案例：参考让错误 token `super` 的负权重从约−19.922变为−26.795；sharing让正确选项 `A` 从约0.00019变为0.524。它展示力度改变，仍不是因果识别关键 token 的证据。

### 9.4 真实成本不能遗漏 teacher cache

Table7 中 training loop 不含 cache，Qwen变化−0.7%，Llama+7.3%。但单次实验计入完整cache后 GPU-time分别约+29.0%、+30.3%；cache跨3次运行共用后仍为+9.2%、+15.0%。因此它不是STOPD式加速方法，不能只用“循环成本接近”忽略预计算。

## 10. 局限和适用边界

1. **结果正确不等于逐步正确。** 全部 token 同号可能强化正确选项之前的错误 reasoning，也可能惩罚失败回答中的正确中间步骤。Appendix D.2 甚至明确展示了“正确选项、早期推理有错”的例子。它仍未完整解决你关心的“到底哪些 token 应该学”。
2. **teacher success 是单个缓存采样的结果。** 一次失败不等于老师不会做；一次成功可能碰巧通过。binary verifier 也可能漏检，MBPP只通过有限测试不是全输入正确性证明。
3. **参考只影响评分，不修复已生成前缀。** 它缓解 prefix contamination，不能保证消除。
4. **改变目标，不保留 vanilla OPD 梯度。** positive/negative Softplus 和共享系数不是无偏估计技巧；不要与vOPD baseline或IPW混同。
5. **surrogate 系数同号不保证所有 token 概率同时同向变化。** 参数共享、softmax竞争和不同样本梯度干涉仍在。
6. **同任务平均保留 teacher 的整体尺度。** batch中该子集太少时尺度可能抖动；不是跨任务的自动公平归一化。
7. **有限验证范围。** 短60步、1024token上限、有限模型和相同训练/测试任务类型；跨新任务迁移、长CoT、开放写作不在结论范围。SciKnowEval是作者自行划分train/dev/test，不能把结果与官方test benchmark直接等同。
8. **需要完整 outcome。** 本文一般要完整回答才能verify；不能直接接STOPD截断后继续使用同样$r_S$。若有partial verifier/critic，需要另定义目标与估计。

## 11. 总结：完整故事线

OPD 的teacher偏好不一定与成功一致；只门控student correctness仍缺少teacher能否成功的信息；DuoOPD用student outcome定方向，并在两种disagreement中分别引入verified teacher reference和student-success共享权重，取得额外增益。

$$
\boxed{\text{学习方向由 student verification 决定；支持来源由 joint outcomes 决定。}}
$$

在现有RoadMap的组件中，它主要修改 **T（teacher context）、A（verified outcome）、W（token/response权重）**，保留student rollout与actor更新骨架。

**可借鉴认知：老师的知识应支持任务成功，而不是让学生成功的实现服从老师的偏好。仍开放的问题：如何把最终成功与失败进一步分解成局部可信的教学单元。**

## 代码定位与归档记录

- `main/src/methods/duoopd/objective.py`：Softplus、四种 outcome、同任务权重共享。
- `main/src/methods/duoopd/teacher.py`：verification、teacher reference 路由、保持学生 response token IDs。
- `main/src/methods/duoopd/trainer.py`：整批次计算权重，再交给 actor。
- 以上为静态源码核对，没有安装训练依赖或运行实验。
- `main` → `/usr/data/DuoOPD`。git HTTP/2 失败、HTTP/1.1 拉取超时，改用官方 GitHub 源码 archive；这是可用源码快照，非完整 git checkout。来源、revision 前缀和 SHA256 见 `main/CODE_SOURCE.json`。
