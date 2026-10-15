# 13｜STOPD 与正交残差校正：真正的在线数学训练验证

> 状态更正：本页是自建PyTorch循环的探索实验，用户要求的官方训练脚本复现尚未由本页完成。官方复现另存于 `../experiments-upstream`，必须以该目录的实际日志和结果为准。

## 1. 本轮结论

**DR 残差校正确实降低了两组固定轨迹上真实参数梯度的查询方差；这次短训中正确率较高，但差异不显著，整体训练也没有因少查 teacher 而更快。**

这轮完成 full-parameter student 更新，与前次只有离线梯度投影的实验不同。外部 teacher 是固定4B模型，不给student或teacher额外参考答案。本文的DR指doubly robust，与归档的Dr.OPD论文不是同一方法。

- [Idea 与完整公式](../../../idea/002-DR-OPD-预测更新残差与正交校正.md)
- [代码、模型、数据与原始日志](../experiments-DR-online) → `/usr/data/STOPD-DR-Online`
- [统计理论导航](00-阅读导航与研究问题.md)

## 2. 到底复现了什么

使用官方STOPD的 **framework-independent core**：原始采样、blockwise IPW、`loss`和lagged denominator均直接导入，官方仓库不修改。复现的是缩小设置中的方法，不是论文原数值或verl/vLLM系统速度。

| 项目 | 本轮 | STOPD论文主设置 |
|---|---|---|
| student / teacher | Qwen3-1.7B-Base / Qwen3-4B-Instruct-2507 | 同一组合 |
| 数学训练来源 | Nemotron-Cascade-RL-Math，128题 | 14,476题 |
| 评估 | 固定随机MATH-500 64题，avg@1 | 完整数学benchmark组合，按原协议多次采样 |
| batch / steps | 8 / 32 | 256 / 250 |
| common OPD warmup | 4步 | 25步 |
| rollout上限 / block | $L=512,K=64$ | $L=16384,K=128$ |
| actor更新 | 全参数、AdamW、$10^{-6}$、clip norm1 | 主设置AdamW、$10^{-6}$ |
| 实现 | PyTorch紧凑解码、BF16 autocast、FP32 master/state | verl / vLLM |

完整rollout参照指生成至EOS或共同512-token上限，并不代表每次都自然结束。评估允许1024token，temperature1、无top-p/top-k截断；使用verl `math_reward.compute_score`的boxed答案规范化匹配。不是pass@8或论文avg@3。

数据固定且未按结果挑选。排除与全部500道测试题的精确规范化题目重合；character-5gram Jaccard≥0.8检查也未发现近重复。两条超出1024prompt-token的训练题被替换，原文件和替换记录保留；已核对4步共同warmup的32个样本完全未变。这不能排除预训练污染或所有语义近重复。

## 3. 四个独立方法

1. **Vanilla OPD**：完整到EOS/cap，所有token真实teacher反馈，普通token均值。
2. **STOPD**：随机截断，所有保留token真实teacher反馈，官方IPW及$C_k$。
3. **STOPD query-IPW**：相同截断，以response为单位用$e=0.5$随机teacher查询；未查询系数为0，查询系数$r/e$。
4. **STOPD query-DR**：相同截断与query概率，未查询用预测项，查询时加残差校正。

后两者共用query masks和停止预算；四方法共用同一warmup model/optimizer state与后续问题顺序。不同更新使后续rollout长度变化，因此实际token成本不完全相同。query-IPW与DR的预算公平性是相同sampling design，实际成本仍单独报告。

## 4. 最关键公式及Neyman含义

$$
\widehat g^+=\frac1{C_k}\sum_{i,t}\frac{I_{i,t}}{p_t}
\operatorname{sg}\left[\alpha_k b_{i,t}+\frac{Q_i}{e_i}(r_{i,t}-\alpha_k b_{i,t})\right]
\nabla\log\pi_\theta(y_{i,t}\mid c_{i,t}).
$$

$b$为过去teacher labels拟合的ridge predictor；$\alpha$只用过去calibration responses的8维梯度sketch校准。fit/calibration按prompt hash划分。当前step内权重冻结，auxiliary predictor不接收actor loss梯度。

已知query概率时，对任意冻结预测器都保持条件query平均梯度。对预测扰动$h$：

$$
D_b\mathbb E_Q\widehat g[h]=
\frac1{C_k}\sum_{i,t}\frac{I_{i,t}}{p_t}\mathbb E_Q[1-Q_i/e_i]h_{i,t}v_{i,t}=0.
$$

这就是本轮用到的observation score正交性，不是另一个独立的“Neyman模型”。不估计propensity，不自动继承DML整体训练收敛结论，也不保证Adam和gradient clipping后的每次更新无偏。

## 5. 正确率结果

初始学生在相同precision与采样协议下答对 **10/64（15.62%）**。

| 方法 | 答对 | 正确率 | 95% Wilson区间 |
|---|---:|---:|---:|
| Vanilla OPD | 32/64 | 50.00% | [38.1%, 61.9%] |
| STOPD | 27/64 | 42.19% | [30.9%, 54.4%] |
| STOPD query-IPW | 26/64 | 40.62% | [29.5%, 52.9%] |
| STOPD query-DR | 30/64 | 46.88% | [35.2%, 58.9%] |

DR相对query-IPW：6题从错变对、2题从对变错；配对McNemar exact双侧 $p=0.289$。**不足以确认可靠正确率提升。**

按用户要求，本轮固定一个training seed（42），不追加其他seed。数据抽样seed20261010与evaluation seed1234只用于分离随机过程。Wilson区间只描述这64题的采样不确定性，不能覆盖training seed不确定性。没有完整500题、没有长CoT验证，也没有code训练；这轮只做math。

Vanilla OPD为32/64，高于STOPD的27/64，也高于DR的30/64。因此当前结果支持“稀疏teacher查询中残差校正有作用”的初步判断，不能写成STOPD在本轮保持了完整OPD的正确率，或DR已经优于完整OPD。

## 6. 教师预算与真实耗时

以下仅统计共同warmup之后的28步，排除初始化、模型保存、evaluation与独立audit；共有4步warmup耗时另计。

| 方法 | student生成token | teacher评分token | 评分/生成 | loop时间(s) |
|---|---:|---:|---:|---:|
| Vanilla OPD | 89,866 | 89,866 | 100.0% | 501.5 |
| STOPD | 51,103 | 51,103 | 100.0% | 434.3 |
| STOPD query-IPW | 51,747 | 28,381 | 54.8% | 402.5 |
| STOPD query-DR | 50,547 | 27,459 | 54.3% | 451.5 |

在本轮缩小实现中，STOPD相对Vanilla观测到的生成token减少约43.1%，post-warmup loop约1.15倍快；这不是论文系统速度复现。

DR查询了130/224条response；名义概率0.5，真实评分token比例54.3%。相对全评分STOPD，teacher评分token减少约46.3%，但其loop反而长约4.0%。

**少查询teacher没有带来整步加速。** 4B teacher scoring每步只耗约0.2–0.4秒，学生generation约十几秒，是主要瓶颈。方法的rollout行为、最大活跃解码轮数以及不同GPU/concurrent workloads也会影响时间，当前数字不是严格控制设备负载的性能benchmark。不能把50%query概率写成2倍训练加速。

所有方法都包含相同student feature/gradient-sketch diagnostics。它们不额外生成未来token，但会有前向计算开销；时间表包含这些开销。共同warmup、独立offline audit和模型下载成本都在日志中可查看，未将offline audit算作加速。

## 7. 真实actor参数梯度的机制验证

测量最后一层 `model.layers.27.self_attn.o_proj.weight` 的完整2048×2048参数梯度，其他参数冻结用于audit。它是实际训练过的actor参数，不是此前只测的virtual logit-bias方向；但仍不等于全模型梯度。

对8条固定retained responses，缓存全部teacher labels，枚举$2^8=256$种独立query组合，$e=0.5$，精确设计方差：

$$
\operatorname{tr Var}_{Q,\rm IPW}=\sum_i\frac{1-e}{e}\|G_i\|^2,
\qquad
\operatorname{tr Var}_{Q,\rm DR}=\sum_i\frac{1-e}{e}\|G_i-\alpha B_i\|^2.
$$

| 模型阶段 | IPW variance | DR variance | DR/IPW | 减少比例 |
|---|---:|---:|---:|---:|
| 共同warmup | 0.675628 | 0.460746 | 0.682 | 31.8% |
| 最终DR模型，另一组8题 | 0.083192 | 0.062989 | 0.757 | 24.3% |

两个audit中枚举mean与完整teacher梯度的差异norm均为0（当前浮点表示下）；独立联合STOPD+query toy check也通过，predictor扰动的score期望导数为0。audit不参与actor或proxy训练，不代表真实部署省去了这些offline teacher labels。

两次方差的绝对值不同，因为参数、轨迹和固定分母不同；可比较各自DR/IPW比值，不应把跨checkpoint方差下降全部归于DR。两组8题也不足以证明全任务、所有参数、所有checkpoint都降方差。

## 8. 对idea的判断与修正

本轮支持：**预测残差比完整教学信号更易估计；用已知query概率校正可以保留局部方向，并在真实参数上降低query variance。**

本轮尚不支持：稳定正确率提升、全流程加速、全模型variance保证、首次DR蒸馏或完成原论文复现。

最重要的motivation修正是：**若生成占大头，仅改teacher missingness不是最高价值的预算位置。下一版应研究可校准的gradient-to-go predictor，用序列残差校正支撑更激进的随机停止。**

另一个排除项：本轮全部112个post-warmup更新都触发norm1 clipping。仅把梯度和$G$改用另一个正分母$C$，若仍触发clipping，有$\operatorname{clip}_1(G/C)=G/\|G\|$。所以单纯DR校正token-count分母不会有效改变本轮优化方向；未来应预测和校正方向性残差。详见idea第12节。

不能直接拿scalar remaining loss的critic当gradient critic：数值预测准确不意味着对actor参数的导数准确。可先以LoRA参数或低秩真实梯度basis为目标，预测剩余更新，并以少量完整rollout做独立audit。这个后续版本尚未实现，不能归入本轮结果。

## 9. 图与复现入口

![准确率、teacher预算与时间](../experiments-DR-online/online-results.png)

![真实参数梯度方差](../experiments-DR-online/gradient-variance.png)

独立PDF：[结果图](../experiments-DR-online/online-results.pdf)、[方差图](../experiments-DR-online/gradient-variance.pdf)。

```bash
cd /usr/data/STOPD-DR-Online
CUDA_VISIBLE_DEVICES=0,1 venv/bin/python train.py --method warmup --no-eval
CUDA_VISIBLE_DEVICES=0,1 venv/bin/python train.py --method stopd --resume runs/warmup/checkpoint
CUDA_VISIBLE_DEVICES=2,3 venv/bin/python train.py --method stopd-ipw-query --resume runs/warmup/checkpoint
CUDA_VISIBLE_DEVICES=4,5 venv/bin/python train.py --method stopd-dr --resume runs/warmup/checkpoint
CUDA_VISIBLE_DEVICES=0,1 venv/bin/python train.py --method vanilla --resume runs/warmup/checkpoint
```

重跑使用新的`--run-dir`，避免向旧metrics追加。完整Vanilla入口、固定seed、依赖版本、数据hash与官方STOPD commit见独立目录README和`reproducibility.json`。最初BF16-only初始评估单独保留；最终表采用与在线final evaluation一致的FP32 master/BF16 autocast初始评估。

官方STOPD仓库无修改；没有安装或改变共享环境的依赖。每方法可独立运行。训练、数据处理、已知debug失败和最终完整generations均保留供复查。
