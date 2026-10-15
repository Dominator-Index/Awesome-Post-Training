# DN-MOPD：Beyond Teacher Assignment 解读

来源：[论文](https://arxiv.org/pdf/2609.35347) · [项目](https://lixin.ai/DN-MOPD/) · [代码](https://github.com/LiXin97/DN-MOPD)。目录按 arXiv 2026 归档，不推定会议录用。

**认知：多个领域的老师共同教一个学生，反馈声音最大的老师可能占据更新；分配老师后，还要校准各领域的反馈尺度。**

**方法：保持按领域选老师的规则，用领域 log-ratio 标准差的倒数缩放该领域的 advantage，不减均值，保持正负方向。**

## 1. 这篇文章到底在解决什么问题

数学、代码、指令遵循三个专家来自相同初始模型，但分别经过 RL 训练。学生在自己的回答上接收对应领域专家的 token 反馈。即便三个领域题目数量相近，梯度贡献也可能失衡：反馈更分散的领域会产生更大的更新系数，压过其他领域。

论文关注的是**多教师反馈的可比尺度**。老师分配正确，不代表每个老师的信号强度适合共同训练。

## 2. Motivation：为什么需要这个方法

初始批次中，指令遵循专家的 log-ratio 标准差达到 pooled 标准差的 2.3–4.4 倍；数学约为其一半。4B 诊断中，等权时 IF 贡献了作者所定义组合梯度贡献的 94%（Appendix D.5）。这说明直接平均 loss 不一定平均了学习力度。梯度方向、序列分母仍会影响实际贡献，不能仅由标准差推算占比。

## 3. 背景和已有方法的问题

Label baseline 用领域标签选择对应老师，但不校准信号。随机分配或 winner-take-all 改的是老师来源，无法独立解决来源之间的尺度问题。Open-MOPD 利用平均 reward 幅度设置领域权重，DN-MOPD 则选择当前批次的离散程度，并限制缩放幅度。

统一记号：$h_t=(x,y_{<t})$，$p_t=\pi_S(y_t\mid h_t)$ 是 student 对自己采样 token 的概率，$q_t=\pi_T(y_t\mid h_t)$ 是 teacher 在同一前缀上的概率。以下公式采用这个约定；PowerOPD 原文部分段落使用相反的 $p,q$ 命名。

即时 sampled-token OPD 的更新方向为

$$
g=\frac1Z\sum_t\operatorname{sg}[\log(q_t/p_t)]\nabla_\theta\log\pi_\theta(y_t\mid h_t).
$$

$\operatorname{sg}$ 表示系数不反传，$Z$ 是聚合分母。严格地说，这是冻结采样前缀分布时的条件 KL 更新。完整轨迹 KL 的精确梯度还包含早期动作改变未来前缀的项；不要将两个梯度无条件等同。

## 4. 方法总览

1. 每道题带 math/code/IF 标签，学生生成完整回答。
2. 对应老师为回答逐 token 评分。
3. 用缓存的 rollout student log-prob 统计领域与全批次标准差。
4. 用正的领域乘数缩放 actor 重新计算的教学 advantage。
5. 沿用 Label 的 PPO 式 clipped 更新与 response 平均方式。

没有增加老师调用，没有学习 router，没有改变各领域题目数，也没有提前停止 rollout。

## 5. 数学推导和关键公式

### 5.1 统计使用 rollout 概率

$$
r_{i,t}^{\rm roll}=\ell^T_{i,t}-\ell^{\rm roll}_{i,t},\qquad
\sigma_d=\operatorname{Std}_{i:d_i=d,t\in V_i}(r_{i,t}^{\rm roll}),\qquad
\sigma_{\rm all}=\operatorname{Std}_{i,t\in V_i}(r_{i,t}^{\rm roll}).
$$

$V_i$ 是有效 response token 集合。标准差采用总体约定，除以 token 数；pooled 标准差由拼接所有领域 token 计算，不能用领域标准差的平均代替。因此长回答也更多影响尺度统计。

### 5.2 正乘数校准

$$
\boxed{w_d=\operatorname{clip}\!\left(\frac{\sigma_{\rm all}}{\sigma_d},0.25,4\right),\qquad
\widetilde A_{i,t}=\operatorname{sg}\!\left[w_{d_i}(\ell^T_{i,t}-\ell^{\rm actor}_{i,t})\right].}
$$

$\ell^{\rm actor}$ 是 actor 在更新前重新计算的 student log-prob，和估计尺度用的缓存 rollout 值分开。任何标准差为零或观察数不足两项时，使用 $w_d=1$。不减领域均值，避免把原本应该鼓励的 token 变成惩罚。领域内部的相对排序保持不变。

未触发 clipping 时，$\operatorname{Std}(w_d r_d^{\rm roll})=\sigma_{\rm all}$。这个恒等式只保证测得信号的标准差一致，不保证 actor 梯度范数、方向或学习收益相等。

### 5.3 沿用 clipped loss

$$
\rho_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid h_{i,t})}{\exp(\ell^{\rm actor}_{i,t})},
$$

$$
\mathcal L=-\frac1B\sum_i\frac1{|V_i|}\sum_{t\in V_i}
\min\!\left(\rho_{i,t}\widetilde A_{i,t},\operatorname{clip}(\rho_{i,t},0.8,1.2)\widetilde A_{i,t}\right).
$$

$\rho$ 是新旧 student 比，和 teacher/student 教学比不同。统计阶段按 token pooling，训练阶段先对每条 response 平均，再对 batch 平均。这两个分母不能交换。

## 6. 为什么这个方法 work

正乘数减少了高离散领域的更新系数，给原本被淹没的领域更多空间。假如数学 $\sigma=0.5$、IF $\sigma=3$、pooled $\sigma=1$，则数学乘 2，IF 乘约 0.33。每个领域内什么 token 被鼓励、什么被压低都不变，变化的是领域之间的声音大小。

这是一种批次统计驱动的预调节。作者实验支持“控制强势 IF 信号有帮助”，并没有证明标准差可以衡量教师质量。

## 7. 为什么其他方法不行：它们忽略了什么

只改 assignment 忽略信号尺度；只增加数学权重可能仍留下过强 IF 信号。固定权重并非不能 work：论文固定 first-batch 权重或统一 $(2,1,0.25)$ 在 9B、4B 上与动态方法没有可检测差异。这里必须把“尺度校准有效”和“每批自适应必要”分开。

## 8. 这篇文章真正的 contribution

把多教师 OPD 拆为两个独立设计轴：**向谁学**与**各领域信号多大**。真正新增的位置主要是 RoadMap 的 W/N；多教师结构是它的背景，新增方法不混合老师分布。

## 9. 实验、消融和证据链

- Qwen3.5 9B、4B、2B；每个大小使用 math/code/IF 三个 RL 专家，训练 80 updates，response cap 8192，评估 8K/16K，六任务聚合。
- 对 Label 的 8K Total 增益分别为 +2.47、+3.08、+2.92 个百分点；16K 为 +1.17、+2.24、+2.36（Tables 1–2）。不是单一数学 benchmark 的分数。
- 三个 student seeds 的 16K 平均增益为 +1.12、+1.97、+2.34，作者的配对区间高于零。
- 只压低 IF 能恢复多数收益；只放大 math 较弱。固定权重在两个大尺寸上已很有竞争力；2B 动态估计相对 first-batch 固定权重高 1.10 点（Table 3、Appendix C.6）。
- Appendix 的更早 Qwen3 开发配置未呈现同样明确收益，说明诊断应针对具体专家池；不能假定所有多教师 OPD 都存在相同失衡。

以上是论文报告，本次没有重新训练或复现。

## 10. 局限和适用边界

标准差大可能反映难题、真实能力差距或噪声，缩小它可能同时削弱有效信号。结果主要基于一个模型家族、每个尺寸一个专家池；不是所有模型合并方法的最佳成绩。数学、代码与 IF 的域级校准仍无法分辨同域内什么知识值得学。

它与 CorrGRPO 都涉及标准差，但 CorrGRPO 改同题 outcome reward 的组统计，DN-MOPD 改跨领域 teacher log-ratio 尺度。粒度、来源和公式都不同。

## 11. 总结：完整故事线

领域标签解决老师分配，却留下教师信号尺度失衡；DN-MOPD 在既有优势上乘一个有界正数，让多个老师更公平地共同更新学生。有效经验主要是**压低过强领域**。它不是 TOP-D 式概率插值，也不是对关键动作的因果判断。

代码已克隆至 `/usr/data/DN-MOPD`，本目录 [main](main) 为符号链接。实现细节见 Appendix D.1 与仓库 `dn_mopd`、训练 hook。对照阅读：[OPD 信号变换对比](../../RoadMap/OPD信号变换对比.md)。
