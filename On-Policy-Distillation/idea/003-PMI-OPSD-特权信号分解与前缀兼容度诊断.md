# Idea 003：OPSD 信号的 PMI 分解——特权信息何时可教，前缀何时可信

> 日期：2026-10-22。状态：**理论整理＋待验证假设，尚无实验**。依据：本仓库论文笔记（[组件矩阵](../RoadMap/论文组件矩阵.csv)、各论文 README / analysis.md）；本笔记没有重读全部 PDF，也没有做系统文献检索，novelty 未核实。第 5 节的实验代码尚未建立。

## 0. 证据标签

| 标签 | 含义 | 能否直接当结论引用 |
|---|---|---|
| **[已有]** | 经典结果、论文原文结果或本库统计，附出处 | 可以，但要带上原文条件 |
| **[推导]** | 本笔记在写明的假设下做的数学推导 | 数学上成立；不代表经验上有用 |
| **[假设]** | 待验证的经验判断，附可证伪判据 | 不可以，需第 5 节实验 |

## 1. 先记住认知、方法和一个公式

**认知：特权自蒸馏（OPSD）的局部信号，主要成分是“特权信息 $z$ 让 teacher 更喜欢还是更不喜欢这个 token”，即 token 与 $z$ 的逐点互信息（PMI）。风格漂移、fork suppression、位置可靠性结论互相矛盾，可能都是这一个量的性质。**

**方法：暂不提出新训练算法。先用训练中已有或只需多一次 forward 的 log-prob 算出 PMI 与累积 PMI，在分支续写标签上检验它们能否比“相对位置”更好地预测 teacher 指导是否可靠。**

$$
\boxed{
d_t
=\underbrace{\log\pi_{\bar\theta}(y_t\mid x,z,h_t)-\log\pi_{\bar\theta}(y_t\mid x,h_t)}_{\mathrm{PMI}_t:\ z\ \text{对这个 token 的表态}}
+\underbrace{\log\pi_{\bar\theta}(y_t\mid x,h_t)-\log\pi_{\theta}(y_t\mid x,h_t)}_{\text{快照差}}
}
$$

$$
S_t=\sum_{u<t}\mathrm{PMI}_u=\log\frac{\pi_{\bar\theta}(y_{<t}\mid x,z)}{\pi_{\bar\theta}(y_{<t}\mid x)}
\qquad(\text{前缀与 }z\text{ 的兼容度})
$$

**想回答的核心问题：teacher 在某个学生前缀上的指导是否可靠，能否由“这段前缀与特权信息的兼容度”解释，而不是由“它在第几个位置”解释？**

## 2. 已有理论与结果 [已有]

### 2.1 本库论文中的相关结果

| 论文 | 原文结果或公式 | 与本笔记的关系 |
|---|---|---|
| [Rethinking OPSD](../语言模型与推理/NeurIPS2026-Rethinking-OPSD/analysis.md) | 图 5 固定同一 student 轨迹、只改 teacher 是否看参考：`But` 的 log-ratio 从 +0.95 变为 −1.96，`wait` 从 +0.37 变为 −4.97。完整解法比只给答案退化更强。附录中保护 epistemic 词表的效果与随机等量 mask 接近 | 两列之差就是 PMI（§3.1）；退化随 $z$ 信息量增大；问题出在状态而不在词表 |
| [PW-OPSD](../训练机制与稳定性/NeurIPS2026-Teacher-Token-Reliability/analysis.md) | 正确 spine 上的分支续写诊断：早期位置分数 AUROC 0.83（题目级 bootstrap CI [0.66, 0.95]），teacher 熵 0.51，MC-dropout MI 0.51；正例只有 8 个。分支混合恒等式 $\mathbb E_z\mathrm{KL}(q^z\Vert p)=\mathrm{KL}(q\Vert p)+I_q(Y;Z\mid h)$ | E1 直接复用其标签协议；位置是要被检验的基线 |
| [RLCSD](../语言模型与推理/arXiv2026-RLCSD/README.md) | $\log\pi_T(y_t\mid x,y_c^{\ast},y_{<t})-\log\pi_T(y_t\mid x,y_w^{\ast},y_{<t})$，正负参考用完全相同的模板 | 选 $z^-$＝错误参考（§3.3） |
| [VA-OPD](../多模态与交互/NeurIPS2026-Visual-Advantage-OPD/README.md) | $\log T(y_t\mid I_{\rm fine},h_t)-\log T(y_t\mid I_{\rm coarse},h_t)$ | 选 $z^-$＝粗图 |
| [VC-OPD](../多模态与交互/NeurIPS2026-VC-OPD/README.md) | 同一前缀上原图与损坏图下的 teacher log-prob 之差 | 选 $z^-$＝损坏图 |
| [LGR](../训练机制与稳定性/NeurIPS2026-Supervision-Fidelity-Decay/README.md) | 学生前缀越长，teacher 的纠正能力可能越弱 | 与 PW-OPSD 方向相反；H2 尝试调和 |
| [TrOPD](../训练机制与稳定性/arXiv2026-TrOPD/README.md)、[Many Faces](../训练机制与稳定性/NeurIPS2026-Many-Faces-OPD/README.md) | 用师生解码一致性定义可靠区域；把前缀兼容性列为失败原因之一 | 最接近“前缀兼容度”的已有工作 |

### 2.2 KL 方向与多 teacher 的最优解（标准结论）

对一组 teacher 分布 $\lbrace q_k\rbrace$ 与权重 $w_k$（$\sum_kw_k=1$），在完整概率单纯形上：

$$
\arg\min_p\sum_kw_k\thinspace \mathrm{KL}(p\Vert q_k)\ \propto\ \exp\Big(\sum_kw_k\log q_k\Big),\qquad
\arg\min_p\sum_kw_k\thinspace \mathrm{KL}(q_k\Vert p)=\sum_kw_kq_k.
$$

前者展开为 $\sum_ap(a)\log p(a)-\sum_ap(a)\sum_kw_k\log q_k(a)$，加归一化约束后得到几何平均；后者等价于最大化 $\sum_a\big(\sum_kw_kq_k(a)\big)\log p(a)$，得到算术平均。常说的“RKL mode-seeking、FKL mass-covering”就是这个事实。

### 2.3 机器人模仿学习中的 imitation gap

- [Learning by Cheating](https://arxiv.org/abs/1912.12294)（Chen et al., CoRL 2019）：特权 agent 看到真实道路布局和所有交通参与者位置，再教只看相机的 agent。这类特权状态原则上能从传感器推得。
- [Bridging the Imitation Gap by Adaptive Insubordination](https://arxiv.org/abs/2007.12173)（Weihs et al., NeurIPS 2021）：teacher 依赖学生看不到的信息做决策时，模仿会产生 imitation gap。其方法 ADVISOR 在训练中动态加权模仿损失与 RL 损失，让 agent 在模仿与探索之间切换。

本库此前的笔记没有把 OPSD 的问题与这条文献线连起来。

### 2.4 本库统计（有选择偏差）

- [组件矩阵](../RoadMap/论文组件矩阵.csv)的 45 条中，W（mask／权重）22 条、T 19 条，D 只有 3 条、I 只有 6 条。
- 按 arXiv 编号的月份：2026-05 的 16 条中，把任务结果放进训练目标的只有 RWOPD、VPD 2 条；2026-09/10 的 9 条中有 PIVOT、Dr.OPD、AC2、DuoOPD、CorrGRPO 5 条。AOPD 没有 arXiv 编号，未计入，但同属此类。AC2、CorrGRPO 是特意收录的 RL 对照，去掉后为 3/7。
- 这是本库收录的统计，不是领域的随机样本，只能说明“本库视野内”的趋势。

## 3. 推导 [推导]

### 3.1 信号分解（恒等式）

条件：teacher 与 student 共享 tokenizer；response 以同一串 token id 接在两种 prompt 之后，不能各自重新分词。在 $d_t$ 中加减同一项 $\log\pi_{\bar\theta}(y_t\mid x,h_t)$，即得第 1 节公式。常见配置下各项如下：

| 配置 | 快照差 | 得到 PMI 的额外代价 |
|---|---|---|
| OPSD，teacher 与 rollout 策略同参数（每步同步） | 若信号的 student 项用 rollout 时的 old log-prob，则快照差为 0，$d_t=\mathrm{PMI}_t$；多 epoch 更新时另差一个新旧策略 log 比 | **零**：信号本身就是 PMI |
| OPSD，teacher 冻结在初始 checkpoint（Rethinking 的主设置） | $\log\pi_{\theta_0}(y_t\mid x,h_t)-\log\pi_\theta(y_t\mid x,h_t)$ | 需要初始模型在普通 prompt 上的 log-prob；若训练开了 reference KL 且 ref＝初始 checkpoint，可直接复用 |
| 外部 teacher，带特权 | 变成“无特权 teacher 与 student 的能力差” | teacher 在普通 prompt 上多做一次 forward |
| 外部 teacher，无特权 | 不存在 PMI；对应的兼容度是 teacher 对前缀的似然 $\sum_{u<t}\log q(y_u\mid h_u)$ | 零：训练已经算过 |

Rethinking OPSD 图 5 的两列正好是这两项：无特权那列是能力差，两列之差是 PMI。`wait` 的 PMI 约为 $-4.97-0.37=-5.34$。

### 3.2 累积 PMI 与前缀兼容度

由自回归链式法则，$S_t=\log\pi_{\bar\theta}(y_{<t}\mid x,z)-\log\pi_{\bar\theta}(y_{<t}\mid x)$：

- $S_t>0$：带着 $z$ 的模型比不带时更可能写出这段前缀，前缀在沿 $z$ 的路线走。
- $S_t<0$：前缀与 $z$ 不一致，例如学生走了另一条解法或已经出错。
- $S_t\approx0$：前缀还没显出与 $z$ 的关系，常见于开头。

若把模型的条件分布看作某个一致联合分布 $\pi(z,y\mid x)$ 的条件，由 Bayes 公式 $S_t=\log\pi(z\mid x,y_{<t})-\log\pi(z\mid x)$，即“读完前缀后对 $z$ 的后验提升”。LLM 把 $z$ 放进 prompt 得到的条件分布不保证来自一致的联合分布，所以这只是解读，不是定理。

序列层面：teacher 与 student 同参数时，$\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[S_{T+1}]=-\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\thinspace \Vert \thinspace \pi_\theta(\cdot\mid x,z)\big)$。也就是说，序列级 RKL 型 OPSD 在推高 response 与 $z$ 的 PMI。任何与 $z$ 相关的特征都会被推高，包括“看过参考后的写法”，这就是 RLCSD 所说的风格漂移。

### 3.3 对比基线：RLCSD、VA-OPD、VC-OPD 都在选 $z^-$

$$
\mathrm{PMI}_t(z^+)-\mathrm{PMI}_t(z^-)=\log\pi_T(y_t\mid x,z^+,h_t)-\log\pi_T(y_t\mid x,z^-,h_t).
$$

无特权项与 student 项都被抵消。$z^-$ 的选择决定留下什么：

| $z^-$ | 被抵消的成分 | 留下的成分 | 对应方法 |
|---|---|---|---|
| 空（不给参考） | 无 | 内容＋模板／风格 | vanilla OPSD 的 PMI 部分 |
| 同模板的错误参考 | “看过某个参考”的共同风格 | 与正确答案特有的内容 | RLCSD |
| 粗粒度或损坏的图像 | 语言先验与模板 | 依赖精细视觉证据的部分 | VA-OPD、VC-OPD |

代价：对比信号也丢掉了能力差项。自蒸馏中能力差本来约为 0；外部 teacher 时丢掉的是“teacher 本来就更会”的那部分信号。

### 3.4 单参考与多参考：fork suppression 的一个机制

由 2.2，若同一题有多个正确参考 $z_1,\dots,z_K$：

- **RKL 平均**：最优学生 $\propto\exp\big(\frac1K\sum_k\log q_k\big)$。只要某条解法给分叉 token $a$ 极低概率，$a$ 就被压到接近 0。
- **FKL 平均**：最优学生 $=\frac1K\sum_kq_k$，任一解法支持的分叉都会保留。
- **固定一条参考（OPSD 的常见做法）**：两者都退化为 $q_{z^{\ast}}$，学生在每个前缀上都被拉向这一条路线。
- **JSD（Rethinking 所用）**：介于两者之间，没有简单闭式。

这只解释“解法路线的分叉”被压缩，不解释 `wait`、检查这类 token 的减少。

### 3.5 后验混合 teacher，以及它与 PW-OPSD 恒等式的区别

$$
q_{\rm mix}(\cdot\mid h_t)=\sum_kw_k(h_t)\thinspace \pi_{\bar\theta}(\cdot\mid x,z_k,h_t),\qquad
w_k(h_t)\propto\pi(z_k\mid x)\exp\Big(\sum_{u<t}\log\pi_{\bar\theta}(y_u\mid x,z_k,h_u)\Big).
$$

$w_k$ 是在 $\lbrace z_k\rbrace$ 上、以 $\pi(z_k\mid x)$ 为先验、读完前缀后的后验（与 3.2 有同样的解读限制）。其中的指数项就是每条参考下 teacher 对前缀的似然，做 $K$ 次 teacher forward 时顺带得到。

PW-OPSD 恒等式中的 $\alpha(z\mid h)$ 也是分支后验。在 FKL 下，按 $\alpha(z\mid h_t)$ 抽 $z$ 训练与对混合分布训练有相同的期望梯度，因为互信息项与 student 无关，PW-OPSD 也指出了这一点。**区别在于实际 OPSD 并不按后验抽 $z$：每题固定一个 $z^{\ast}$，无论学生前缀是否与它兼容。** 所以要改的不是“用混合”，而是“让 $z$ 的权重随前缀更新”。在 RKL 下，混合分布与对 $z$ 求平均还会给出不同的最优解（3.4）。

局限：如果每个 $q_k$ 都很少输出 `wait`（知道答案的人不需要检查），混合也恢复不了它。

### 3.6 可推得性：模仿特权 teacher 何时是“摊销”，何时是“假装知道”

设学生的理想策略是只看 $(x,h)$ 时的 Bayes 最优决策。

- 若 $z$ 能由学生以可接受的代价从 $x$ 推出（裁剪区域的像素本就在全图里；harness 只是同一模型多算几步），则 $\pi(\cdot\mid x,z,h)$ 是学生“多算一点就能达到”的分布，模仿它相当于把计算摊销进参数。
- 若 $z$ 在可接受代价内推不出（难题的答案），$\pi(\cdot\mid x,z,h)$ 中就不存在信息收集动作，而学生的最优策略需要这些动作。这就是 2.3 的 imitation gap。

这一段是概念论证，“可接受代价”没有形式化定义。

## 4. 待验证假设 [假设]

### H1：可推得性决定 OPSD 收益的正负

| $z$ 的类型 | 学生能否推出 $z$ | 任务是否需要试错 | 例子 | 预测 |
|---|---|---|---|---|
| 计算型 | 能 | 任意 | [Vision-OPD](../多模态与交互/NeurIPS2026-Vision-OPD/README.md) 裁剪、[OPHSD](../语言模型与推理/NeurIPS2026-Harness-Self-Distillation/README.md) harness、[d-OPSD](../语言模型与推理/NeurIPS2026-Self-Future-dLLMs/README.md) self-future | 正收益 |
| 信息型 | 不能 | 不需要（短输出、单步定位） | [GUI-SD](../多模态与交互/NeurIPS2026-GUI-Grounding-Self-Distillation/README.md) bbox、短预算 instruct 模型 | 正收益 |
| 信息型 | 不能 | 需要（长推理） | thinking 模型＋参考解法 | 退化；$z$ 信息量越大退化越强 |

已有的一致证据：Rethinking 中完整解法比只给答案退化更强；thinking 模型的 fork rate 明显下降，instruction 模型变化较小。待核对：Vision-OPD 的裁剪区域从哪里来。若来自人工标注，“看哪里”本身就是信息型特权。

**判据**：若计算型特权在长预算 thinking 模型上同样出现同等幅度的 fork rate 下降，或信息型特权在长推理上稳定带来正收益，则 H1 不成立。

### H2：可靠性跟随前缀兼容度，而不是位置

预测分三段：

1. $S_t\approx0$（前缀还没显出与 $z$ 的关系，多在早期）：teacher“跑在学生前面”，指导不可靠。对应 PW-OPSD 的早期不可靠。
2. $S_t>0$（沿 $z$ 路线）：指导可靠。对应 PW-OPSD 的后期可靠。
3. $S_t<0$（已偏离 $z$）：teacher 依据不兼容的 $z$ 给纠正，指导失真。对应 LGR 的长前缀衰减。

按 H2 的解释，位置在 PW-OPSD 中有 0.83 的 AUROC，是因为它只用正确 spine，而正确 spine 越往后越沿 $z$ 路线，位置与 $S_t$ 高度相关。判据见 5.9。

### H3：反思 token 的 PMI 符号随兼容度翻转

在与 $z$ 兼容的前缀上，`wait`、`but`、`let me check` 等的 PMI 为负（知道答案的人不检查）；在已经出错（$S_t<0$）的前缀上，同类 token 的 PMI 接近 0 或为正（知道答案的人会要求回头）。若成立，说明 fork suppression 主要发生在兼容前缀上：应按状态屏蔽，而不是按词表屏蔽。这也能解释 Rethinking 中词表保护无效。

## 5. 零成本诊断实验方案

### 5.1 “零成本”指什么

- **特征零成本或近零成本**：PMI 与 $S_t$ 来自训练中已有的 log-prob，或每条 rollout 多一次 forward（见 3.1 的表）。
- **标签不是零成本**：验证特征需要分支续写，得到 student 的后续成功率。这是一次性的诊断开销，不进训练循环。
- 诊断成立之后，训练时只需要特征，不需要标签。

### 5.2 每个位置计算的特征

对一条 student spine $y$（题目 $x$、参考 $z$），做两次 teacher-forcing forward：一次用含 $z$ 的 teacher 模板，一次用不含 $z$ 的 student 模板，response 用同一串 token id。$L$ 为 response 长度，$q_z$、$q_0$ 分别为两种模板下的 teacher 分布。

| 编号 | 特征 | 定义 | 额外代价 |
|---|---|---|---|
| F0 | 相对位置（基线） | $(t-0.5)/L$ | 无 |
| F1 | teacher 熵（基线） | top-16 截断熵，口径同 PW-OPSD | 无 |
| F2 | 当前 PMI | $\log q_z(y_t)-\log q_0(y_t)$ | 普通 prompt 一次 forward |
| F3 | 累积 PMI | $S_t=\sum_{u<t}\mathrm{PMI}_u$ | 同上 |
| F4 | 平均累积 PMI | $S_t/(t-1)$ | 同上 |
| F5 | 局部特权散度 | $\mathrm{KL}(q_z\Vert q_0)$，在两边 top-64 并集加一个剩余桶上计算 | 同上 |
| F6 | 候选 PMI | teacher top-3 候选 $a$ 的 $\log q_z(a)-\log q_0(a)$，取最大值与均值 | 同上 |
| F7 | 对比累积 PMI | $\sum_{u<t}[\log q_{z^+}(y_u)-\log q_{z^-}(y_u)]$，$z^-$ 为同模板的错误答案 | 再一次 forward |

F7 用来区分“内容兼容”与“模板风格”。

### 5.3 数据与模型

- **锚点复现**：先按 PW-OPSD 附录 A 的协议，在 Qwen3-4B（或资源允许的同系列模型）上复现位置 AUROC，确认标签口径一致，再比较新特征。换模型时不能拿 0.83 直接比，以同一批数据上重测的位置 AUROC 为基线。
- **题目**：先让 student 每题采样 8 次估计通过率，保留通过率在 [0.2, 0.8] 的题。PW-OPSD 正例只有 8/279，题目太容易可能是原因之一；中等难度的题应能产生更多 real-uncertain 正例。目标是至少 40 个正例。
- **特权条件**：$z$＝最终答案（与 PW-OPSD 诊断一致）和 $z$＝完整参考解法，两者作为因素比较。
- **spine**：每题采样 student 回答，正确 spine 用于 E1，错误 spine 用于 E2。

### 5.4 E0：恒等式与工程正确性检查（不需要标签）

1. 抽 20 条 spine，检查 $\sum_t\mathrm{PMI}_t$ 等于两种 prompt 下整段 response 的 log-prob 之差，每 token 平均误差小于 $10^{-3}$。
2. 检查两种 prompt 下的 response token id 完全一致。
3. 在 teacher 同步的 OPSD 配置下，检查训练日志里的 $d_t$ 与离线算出的 PMI 一致。

### 5.5 E1：正确 spine 上的可靠性预测（复现并扩展 PW-OPSD）

- **标签**：位置选择、候选、续写与分类完全沿用 PW-OPSD。用 top-16 截断熵选高歧义位置，每题最多 5 个；取 teacher top-3 候选；每个候选强制接上后，用 student 模板（不含 $z$）续写 6 次；按成功率分为 diversity、real-uncertain、gray，gray 不参与二分类。
- **特征**：在同一批位置上计算 F0–F7。
- **单变量指标**：题内去均值 AUROC（与 PW-OPSD 同口径），题目级 bootstrap 95% CI（2000 次）。
- **增量价值**：按题目分组做 5 折交叉验证 logistic 回归，比较只用 F0 与 F0＋F$_k$ 的 held-out AUROC 之差，用题目级 bootstrap 给出差值 CI。
- **拆开位置混杂**：按相对位置三等分，在每一层内计算 F3、F5 的 AUROC。

### 5.6 E2：错误 spine 上的 teacher 帮助度（检验 LGR 方向）

- **位置**：每条错误 spine 按相对位置均匀取 5 个位置，不按熵筛选，避免只看高歧义点。
- **标签**：$\Delta_t=\widehat{\mathrm{succ}}(\text{强制 teacher top-1})-\widehat{\mathrm{succ}}(\text{强制学生原 token})$，两者都用 student 模板续写 8 次。$\Delta_t>0$ 表示 teacher 的指导对看不到 $z$ 的学生有用。teacher top-1 与学生原 token 相同时记 $\Delta_t=0$，并单独报告这类位置的比例。
- **分析**：$\Delta_t$ 与 F3 的 Spearman 相关，以及控制 F0 后的偏相关；按 F3 的正负分组比较 $\Delta_t$ 的均值。

### 5.7 E3：反思 token 的 PMI 符号（检验 H3）

- 不需要续写标签。预先固定反思词表（wait、but、hmm、let me check、actually 等），并与 Rethinking 附录的口径对齐。
- 比较两组：正确 spine 与错误 spine 上反思 token 的 PMI 分布；错误 spine 中 $S_t\ge0$ 段与 $S_t<0$ 段的 PMI 分布。
- 对照：位置分布匹配的随机非反思 token。

### 5.8 E4：训练中监控（真正零成本，可直接加进已有 OPSD／STOPD 训练）

不改 loss，只在每步记录：$S_t$ 的分位数、$S_t<0$ 的 token 比例、反思 token 的平均 PMI，并与 Rethinking 的 fork rate、验证标记密度对照。目的是看训练过程中 PMI 分布怎样变化，以及 fork suppression 出现之前是否已有可见信号。只有 teacher 同步时 $d_t$ 直接等于 PMI；teacher 冻结时需要 3.1 表中的额外 log-prob。

### 5.9 预先写下的判据

| 结果 | 结论 |
|---|---|
| E1：F3 或 F5 的 AUROC ≥ F0，且 F0＋F$_k$ 的增量 CI 不含 0 | 支持 H2：兼容度提供位置之外的信息 |
| E1：增量 CI 含 0，但位置分层内 F3／F5 的 AUROC 仍 > 0.6 | 兼容度与位置大部分重合，H2 只部分成立 |
| E1：位置分层内 F3／F5 的 AUROC ≈ 0.5 | H2 不成立，位置效应另有原因 |
| E2：控制位置后，F3 与 $\Delta_t$ 的偏相关显著为正 | 支持“偏离 $z$ 导致指导失真” |
| E2：$\Delta_t$ 与 F3 无关 | LGR 的衰减不能用兼容度解释 |
| E3：错误段的反思 token PMI 明显高于兼容段 | 支持 H3，按状态屏蔽值得做 |
| F7 优于 F3 | 风格成分是噪声来源，对比基线是必要的 |

正负结果都要保存；单个模型、单组题目的结果不能当作普遍结论。

### 5.10 开销与分阶段

主要开销是续写：E1 约为 题目数 × 5 位置 × 3 候选 × 6 次，E2 约为 题目数 × 5 位置 × 2 个强制选项 × 8 次。建议分三步：

1. **阶段 A**：60 题，只跑 E0 与 E1，确认锚点（位置 AUROC 与 PW-OPSD 相近）和管线正确。
2. **阶段 B**：扩到能得到至少 40 个正例的题量，跑完整的 E1、E2、E3。
3. **阶段 C**：在已有 OPSD 或 STOPD 训练中加入 E4 日志。

代码计划放在 `/usr/data/OPD-PMI-Diag`，与 001、002 的 `/usr/data` 约定一致；目前尚未建立。

### 5.11 混杂与风险

- **位置混杂**：$S_t$ 是累加量，天然随 $t$ 增长。必须做分层和增量检验，不能只报单变量 AUROC。F4 取平均能减弱这个问题，但不能消除。
- **模板混杂**：teacher 模板本身会改变分布，与 $z$ 的内容无关；F7 用于区分。
- **选择偏差**：PW-OPSD 只用正确 spine。E2 补上错误 spine，但两类 spine 的位置选择规则不同，结果要分开报告。
- **答案与完整解法**：PW-OPSD 诊断给的是最终答案，训练给的可能是完整解法，所以两种都要测。
- **正例稀少**：PW-OPSD 只有 8 个正例；阶段 B 之前不下结论。
- **快照差**：冻结 teacher 与当前 student 的差距随训练增大。E1、E2 用同一模型同时做 teacher 与 student，快照差为 0；结论能否迁移到训练中，需要 E4 检查。

## 6. 诊断成立后再考虑的方法（均未实现）

1. **后验混合 teacher（3.5）**：为每题准备 $K$ 条经过验证的参考（可由 teacher 采样后验证），teacher 分布按前缀似然加权。代价是 $K$ 倍 teacher forward。
2. **兼容度门控**：$S_t$ 低于阈值时降低特权项的权重，退回无特权信号或任务奖励。思路类似 ADVISOR 按 imitation gap 在模仿与 RL 之间切换。
3. **与成本方向结合**：按 2.4 的统计，“teacher 偏好是否带来任务收益”这条线已有 AOPD、Dr.OPD、DuoOPD、PW-OPSD；“便宜地估计 student 后续成功率”这条成本线更空。兼容度若有效，可以作为 [002 DR-OPD](002-DR-OPD-预测更新残差与正交校正.md) 预测器的廉价特征，或用来决定哪些位置值得做分支续写。

## 7. 与已有论文和本库 idea 的关系

| 对象 | 已有内容 | 本笔记的增量（待验证） |
|---|---|---|
| PW-OPSD | 分支续写诊断、位置先验、分支混合恒等式 | 用兼容度解释位置效应；指出“固定 $z$”与“按后验加权”的区别 |
| Rethinking OPSD | fork suppression 诊断，没有修复方法 | 在 PMI 视角下给出机制解释；主张按状态而不是按词表处理 |
| RLCSD／VA-OPD／VC-OPD | 各自在一个模态里使用对比 teacher 信号 | 统一为 $z^-$ 的选择问题，并写出各自丢弃的成分 |
| LGR／TrOPD／Many Faces | 长前缀衰减、解码一致性、前缀兼容性 | 把兼容度定义成可直接计算的连续量 |
| [传统 RL 迁移笔记](传统RL方法迁移到OPD-对比与研究思路.md) | 用 student 后续成功率检验 teacher 指导 | E1、E2 的标签就是这个量；本笔记补的是廉价代理特征 |
| [002 DR-OPD](002-DR-OPD-预测更新残差与正交校正.md) | 预测 teacher 反馈，随机查询残差 | 兼容度可作为预测器特征（未测） |
| [001 BQ-OPD](001-BQ-OPD-蒸馏计算行为而非等价写法.md) | 等价实现分组 | 没有直接关系；两者都在问 teacher 偏好中哪些不该迁移 |

## 8. 证据边界

- 第 2 节的论文结果来自本库笔记，本次没有重读全部 PDF。Rethinking 图 5 的数值、PW-OPSD 表 1 的 AUROC 以对应 analysis.md 为准。两篇机器人论文已核对 arXiv 摘要页的题名、作者与会议信息。
- 第 3 节的推导在写明的条件下成立；其中“后验”“Bayes”的解读依赖一致联合分布的假设，LLM 不保证满足。
- 第 4 节全部是假设；第 5 节尚未运行。
- 没有做系统文献检索：PMI／互信息视角、后验混合 teacher、兼容度门控在 2026 年的文献中是否已有，需要检索之后才能主张 novelty。
