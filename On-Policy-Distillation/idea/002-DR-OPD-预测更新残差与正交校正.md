# 002｜DR-OPD：昂贵教师应当校正更新残差，而不是重复提供可预测的教学信号

> 研究候选，实验代号；与已有 Dr.OPD 论文不同。经典 AIPW / control variate / Neyman orthogonality 不是我们的发明；novelty 尚需和 budgeted OPD、learned control variate、DR distillation 逐项核验。此前自建循环的探索实验见第11节，不属于官方源码复现。当前已改为官方STOPD/verl流程，四组各250步，结果待完成。

> 官方最小扩展：[第14号实施说明](../训练机制与稳定性/2026-STOPD/markdown/14-官方源码四组对照与最小修改.md)。为控制改动，当前采用4维廉价特征与logit-score/IPW加权ridge；它与旧探索实验的高维预测器/梯度sketch不同，旧数值不能替代本轮结果。DR本身正交，第二组是不同nuisance拟合策略。

## 1. 一句话认知

**可预测的反馈由廉价模型承担；真实 teacher 的预算用于校正会使学生更新方向出错的残差。预测器该按“更新误差”来信任，不该只按 token 分数预测误差来信任。**

## 2. Question 与 motivation

STOPD 用随机截断减少生成，并以 IPW 保持完整局部梯度和的期望。它对观测到的整份 token 梯度放大：位置越晚、coverage 越低，放大越大。

另一类方法选择 teacher supervision，却容易混淆三个对象：原始 teacher/student 分歧、teacher feedback 的不可预测部分、不可预测部分对 actor 更新的影响。

研究问题：**固定真实 teacher 查询预算，是否应预测并校正 actor update residual，而不是挑选 raw disagreement 大的 token？**

原始 feedback 很大不代表必须每次查询；如果 cheap model 能准确预测它，重复 teacher scoring 的边际信息低。相反，反馈幅度不大但与梯度 score 相乘后方向关键，也可能值得查询。

本轮只实现一个核心操作：prediction plus randomized residual correction。梯度校准是决定预测项强度的必要部分，不引入额外 reward、teacher 插值或 PPO 多组件。

## 3. 先明确目标：本轮估计原始 OPD，不判别老师语义上是否正确

令 $v_t=\nabla\log\pi_\theta(y_t\mid c_t)$，$r_t=\log q(y_t\mid c_t)-\log\pi_{\theta_{old}}(y_t\mid c_t)$。采用上升方向，完整 batch 的局部目标为 $G=\sum_t r_tv_t$。

$I_t$ 表示 STOPD 保留 token，$p_t$ 是真实 inclusion probability。teacher query 以 response 为单位：$Q_i\sim\operatorname{Bernoulli}(e_i)$，$e_i>0$；只在被查询的 response 上实际运行 teacher forward。

$b_t$ 来自过去 teacher labels 拟合的低成本模型。它使用当前学生可见的 prefix features，但不能使用当前未知的真实 teacher feedback。actor 对 $b$ 不求导。

## 4. 核心公式：两层随机性，分别校正

$$
\boxed{\widehat g^+=\frac1{C_k}\sum_{i,t}
\frac{I_{i,t}}{p_t}\operatorname{sg}\left[
\alpha_k b_{i,t}+\frac{Q_i}{e_i}(r_{i,t}-\alpha_k b_{i,t})
\right]v_{i,t}.}
$$

$C_k$ 为官方 STOPD 的 lagged denominator，当前停止前已经固定；$\alpha_k$ 用过去数据校准，在当前 step 内冻结。

先对 query 求期望，任何预测器与任何固定 $\alpha$ 都恢复 $r_t$。再对 STOPD 随机停止求期望，恢复 $G/C_k$。它与 vanilla token mean 还相差 STOPD 原文已有的分母尺度 $N_k/C_k$，不能省略。

我们不计算 teacher 再 mask：未查询 response 没有 teacher forward。所有 retained student token 仍做 actor update，由预测项与观测残差构成系数。

## 5. Neyman orthogonality 在这里具体意味着什么

固定实际 retained trajectories，定义局部 score

$$
\psi(g;b,e)=\sum_t\frac{I_t}{p_t}
\left[b_t+\frac{Q}{e}(r_t-b_t)\right]v_t/C_k-g.
$$

对 feedback predictor 的扰动 $b\to b+\varepsilon h$：

$$
\left.\frac{d}{d\varepsilon}\mathbb E_Q\psi(g;b+\varepsilon h,e)\right|_{0}
=\sum_t\frac{I_t}{p_t}\mathbb E_Q[1-Q/e]h_tv_t/C_k=0.
$$

**预测器误差不会把局部平均更新推向另一个目标。** 已知 query design 下，该结论对任意冻结 predictor 精确成立，强于只在真值附近的一阶鲁棒性。

若把 $e$ 当作需估计的 nuisance，要讨论另一方向的正交性 / 双重鲁棒性，必须有适当 conditional-mean model、MAR、positivity 与 cross-fitting 条件；只用 hidden projection 预测 $r$ 未必足以保证 vector-gradient score 对 estimated propensity 正交。本轮使用设计已知的 $e=0.5$，不估计 propensity，也不借 DML 的名字声称已有在线整体收敛定理。

## 6. 关键修改：为什么要按梯度几何校准 proxy

对 response query unit，$G_i=\sum_t(I_t/p_t)r_tv_t/C_k$，$B_i=\sum_t(I_t/p_t)b_tv_t/C_k$。独立 query 的 trace variance 为

$$
\operatorname{tr Var}_Q(\widehat g^+)=
\sum_i\frac{1-e_i}{e_i}\|G_i-\alpha B_i\|^2.
$$

因此让 $b$ 的 token MSE 低，不一定让更新方差低：token score 与跨 token 梯度的抵消、相关性都会影响结果。

使用过去独立 calibration responses 的梯度 sketch $\widetilde G_i,\widetilde B_i$：

$$
\boxed{\alpha_k=\operatorname{clip}_{[0,1]}
\frac{\sum_{i\in\mathcal C_{<k}}\widetilde G_i^\top\widetilde B_i}
{\sum_{i\in\mathcal C_{<k}}\|\widetilde B_i\|^2}.}
$$

它最小化过去 calibration 上的 residual-gradient norm。本轮 sketch 使用8个 logit-bias方向；另加真实 actor 最后一层 $o\_proj$ 参数梯度 audit，检查 virtual sketch 是否与真实参数方向一致。若 proxy 方向不相关，$\alpha$ 可回到0，此时 DR 退回 query-IPW。

这是经典 control-variate 系数校准的一种应用；可研究的新点是 **OPD 反馈预测与 actor-gradient residual 的错位，及怎样在真正稀疏 teacher 查询下处理它**，不能把公式本身当 novelty。

## 7. 与已有论文的关系

| 方向 | 已有认识 | 我们要验证的增量 |
|---|---|---|
| STOPD | incomplete rollout + exact IPW | 对保留前缀的 teacher missingness 再做 residual correction |
| vOPD | state baseline 降动作采样方差 | action-dependent prediction + query correction，随机性不同 |
| sequential DR | tail prediction + residual telescope | 本轮尚未实现，未来需要廉价且梯度可信的tail surrogate |
| BAS-OPD / SR-OPD | teacher budget 分配 | dense cheap feedback 与原目标校正是否更有用，需完整方法对照 |
| DR ranking distillation | 蒸馏中已有 AIPW 结构 | 本轮不是首次 DR × distillation，而是 LLM online actor-update efficiency 问题 |
| DuoOPD | verifier 定方向、联合 outcome 定支持 | 本轮仍保留 OPD 目标，不能修正老师本身的错误偏好 |

统计方法来源与边界详见 [STOPD 统计笔记导航](../训练机制与稳定性/2026-STOPD/markdown/00-阅读导航与研究问题.md)。DuoOPD 的对照见 [分析](../训练机制与稳定性/arXiv2026-DuoOPD/analysis.md)。

## 8. 实验设计：真实训练与离线 audit 分开

独立代码 `/usr/data/STOPD-DR-Online`；原始 `/usr/data/STOPD` 不修改。

- 原文主设置最小模型组合：Qwen3-1.7B-Base student、Qwen3-4B-Instruct-2507 teacher。
- 官方数据源：Nemotron-Cascade-RL-Math 随机128题，过滤与MATH-500的精确规范化问题重合；固定随机MATH-500 64题评估。不是论文全数据复现。
- 32步、batch8、$L=512,K=64$、4步共同vanilla warmup、学习率$10^{-6}$；完整模型参数训练，FP32 master / Adam states、BF16 forward。
- STOPD：所有保留前缀均查询teacher。
- STOPD-query-IPW：相同截断、$e=0.5$ teacher query，无预测项。
- STOPD-DR：相同截断与query概率，使用过去数据拟合proxy与calibration。
- Vanilla OPD 作为完整生成参照，同样从common warmup继续，已完成。
- 按用户要求固定一个 training seed：42；数据抽样使用固定seed20261010，evaluation使用固定seed1234。这些是随机过程的分离，不是多个训练seed。

nominal teacher budget 相同不代表最终 token cost完全相同：不同更新导致rollout长度变化。因此同时报告实际 teacher scored tokens、actor generated tokens、query数和时间。

每方法独立入口、run目录与模型；common checkpoint含optimizer/STOPD state和过去proxy资料，避免初始化差异。teacher query masks与STOPD随机预算使用固定、分离的seed。

## 9. 判据与风险

1. 首先验证真实参数梯度的query mean保持和同预算variance，而不仅是reward MSE。
2. 再看固定64题的paired accuracy与长度，保留全部generation供复查。
3. 报告全流程wall time，不能把查询减少50%写成训练加速50%。小teacher评分可能只占少量时间。
4. 当前PyTorch core训练循环是官方framework-independent接口的缩小方法复现；没有复现论文verl/vLLM系统或原数值。
5. 单seed、短训、小测试集不能支撑可靠benchmark结论；失败或持平都应保存。
6. 正交性只控制平均score对预测器扰动的影响，不消除估计方差，也不自动保证Adam、clipping与整段训练轨迹相同。

## 10. 当前进度与后续问题

此前1.5B离线实验：code中query-DR有收益、math初版无收益；弱sequential predictor甚至显著增大variance，见 [最小诊断](../训练机制与稳定性/2026-STOPD/markdown/12-最小验证实验与结果.md)。本轮已经完成真正full-parameter online训练，结果见下节。

**若真实teacher评分只是小成本，下一步最有价值的问题是 gradient-calibrated tail residual 是否能够降低 STOPD 的截断方差，使更激进的停止设计可用。** 需要预测剩余更新，而非只预测剩余loss数值；本轮不把这项尚未实现的结果提前归功于DR。

## 11. 本轮真实训练结果

已完成四个方法的32步在线数学训练。详见 [完整结果、成本与方差图](../训练机制与稳定性/2026-STOPD/markdown/13-在线数学实验-STOPD与正交残差校正.md)。

Vanilla OPD 32/64、全评分STOPD 27/64、query-IPW 26/64、DR 30/64。DR相对STOPD多答对3题，相对query-IPW多答对4题，但仍低于Vanilla；本轮不是确认优于完整OPD的证据。按用户要求仅使用一个training seed，不追加多seed。配对检验保留作为描述，不作为继续实验的门槛。

真实最后一层参数的query variance在两次audit下降约32%与24%，teacher评分token相对STOPD减少46.3%，但DR loop未加速。下一版应优先解决生成占主要成本的tail-gradient residual问题。

## 12. 一个应排除的方向：仅校正分母可能没有训练效果

本轮四个方法的每一步更新都触发norm1的gradient clipping。若改动只是把同一个梯度和$G$除以另一正数$C$，且仍触发clipping，则

$$
\operatorname{clip}_{1}(G/C)=\frac{G/C}{\|G/C\|}=\frac{G}{\|G\|}.
$$

因此，单独用DR改进STOPD的token-count分母，在这个优化设置下会被clipping消掉，不能据此期待准确率改善。这个结论要求$C>0$且两种归一化后都触发clipping；浮点误差与阈值附近情况另论。

**更有价值的问题是：预测未生成后缀对更新方向的贡献，而不是只预测剩余token数或剩余loss。** 本轮的query-DR能改变有噪声的梯度方向；未来的序列残差方法也应按真实更新方向校准。该后续方法尚未实现。
