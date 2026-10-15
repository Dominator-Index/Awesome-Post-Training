# On-Policy Distillation 论文与研究笔记

按研究问题组织OPD、OPSD及相关后训练方法。**当前45条记录，41篇有PDF，4篇待补PDF。** 不把下载归档、方法概览、完整精读和实验复现混为同一种完成状态。

- [RoadMap：数学形式、组件与方法](RoadMap/README.md) · [可编辑draw.io](RoadMap/roadmap.drawio) · [PDF图册](RoadMap/roadmap.pdf)
- [论文认知与关键公式](RoadMap/论文一句话认知.md) · [交互式浏览器](RoadMap/index.html)
- [结构化目录CSV](论文目录.csv) · `manifest.jsonl`（本地文件） · `download_summary.json`（本地文件）
- [研究idea](idea/) · [OPSD：老师可以看答案的理解](OPSD理解-老师可以看答案.md)

## 分类与数量

| 分类 | 条目数 | 研究问题 |
|---|---:|---|
| [训练机制与稳定性](训练机制与稳定性/README.md) | 24 | 轨迹分布、采样预算、教师可靠性、反馈变换、权重、归一化；AC2/CorrGRPO作为RL对照。 |
| [语言模型与推理](语言模型与推理/README.md) | 10 | 自蒸馏、参考解法、语言反馈、推理探索、数据来源与文本生成。 |
| [多模态与交互](多模态与交互/README.md) | 8 | 视觉证据、视觉token、GUI与多轮agent，及相关SFT修订对照。 |
| [其他应用](其他应用/README.md) | 3 | 投机解码、扩散模型与早期风险预测。 |

## 最近补充的论文如何归类

- 训练机制与稳定性：STOPD、AC2、CorrGRPO、Dr.OPD、AOPD、DN-MOPD、PowerOPD、DIAL-OPD、DuoOPD。
- 语言模型与推理：OPSD、RLCSD、DF-OPD。
- 多模态与交互：ReDraft、PIVOT。
- 其他应用：Draft-OPD。

AC2和CorrGRPO属于RL方法，ReDraft主要用修订后的文本做SFT；收录用于比较组件，不把它们统称为OPD算法。TOP-D、TrOPD、PowerOPD和DIAL-OPD的区别见[信号变换对比](RoadMap/OPD信号变换对比.md)。

## 完成状态说明

- 所有条目有README方法入口、研究问题分类、认知与方法卡片；正文未取得的条目不猜关键公式。
- 41篇canonical PDF已检查文件头、pdfinfo、页数、SHA256与前两页题名匹配；检查明细见manifest。
- 10篇含已有`analysis.md`；其余以README或专题笔记为主，不能表述为45篇全部精读完成。
- REOPOLD补齐同方法arXiv版本；VC-OPD补齐2026-10-10的arXiv版本。对应原OpenReview版本的逐字一致性尚未核实。
- 原目录中的NeurIPS标签沿用原始来源命名，不把arXiv版本或ICLR Under Review自动表述为会议录用。TrOPD仅有arXiv来源，已改用arXiv2026目录名。
- 实验结果与文献阅读分开：STOPD既有探索实验不能替代官方训练流程复现，具体状态以实验说明和实际日志为准。

### 待补PDF

| 方法 | 来源 | 当前状态 |
|---|---|---|
| [Better Rollouts](训练机制与稳定性/NeurIPS2026-Better-Rollouts/README.md) | [原始页面](https://csr16.github.io/) | OpenReview HTTP403；保留来源与待补状态 |
| [CoTrek](语言模型与推理/NeurIPS2026-CoTrek/README.md) | [原始页面](https://openreview.net/forum?id=DsDVXNgrHK) | OpenReview HTTP403；保留来源与待补状态 |
| [Fewer Tokens, Fewer Layers](多模态与交互/NeurIPS2026-Vision-Token-Pruning-VLM/README.md) | [原始页面](https://openreview.net/forum?id=T7WbWqjpqm) | OpenReview HTTP403；保留来源与待补状态 |
| [On-Policy Hindsight / Early Risk](其他应用/NeurIPS2026-On-Policy-Hindsight-Distillation/README.md) | [原始页面](https://openreview.net/forum?id=L26pEowauj) | OpenReview HTTP403；保留来源与待补状态 |

## 训练机制与稳定性

| 方法 | 一句话认知 | 材料 |
|---|---|---|
| [f-OPD](训练机制与稳定性/NeurIPS2026-f-OPD/README.md) | 旧 rollout 可以复用，但应按新鲜程度决定它还值得信任多少。 | [PDF](训练机制与稳定性/NeurIPS2026-f-OPD/paper.pdf) |
| [AsyncOPD](训练机制与稳定性/NeurIPS2026-AsyncOPD/README.md) | 异步训练能提高吞吐，但 loss 方向和打分缓存决定旧数据是否还能用。 | [PDF](训练机制与稳定性/NeurIPS2026-AsyncOPD/paper.pdf) |
| [Noisy Expert / NAIL](训练机制与稳定性/NeurIPS2026-Behavior-Cloning-Optimality/README.md) | 当专家带噪声时，学生访问自己的状态再问专家，可能比离线模仿更有样本效率。 | [PDF](训练机制与稳定性/NeurIPS2026-Behavior-Cloning-Optimality/paper.pdf) |
| [Rock Tokens](训练机制与稳定性/NeurIPS2026-Rock-Tokens/README.md) | 高 loss 不一定意味着值得学；持续难以纠正的风格残差可能消耗大量梯度。 | [PDF](训练机制与稳定性/NeurIPS2026-Rock-Tokens/paper.pdf) |
| [vOPD / KL Control Variate](训练机制与稳定性/NeurIPS2026-KL-Control-Variate/README.md) | 用一个与当前动作无关的 baseline，可让 OPD 梯度更稳定而保持其期望。 | [PDF](训练机制与稳定性/NeurIPS2026-KL-Control-Variate/paper.pdf) · [精读](训练机制与稳定性/NeurIPS2026-KL-Control-Variate/analysis.md) |
| [EffOPD / Learning to Foresee](训练机制与稳定性/NeurIPS2026-Learning-to-Foresee/README.md) | 如果早期更新方向已经接近最终方向，可以沿当前方向多走一点来加速。 | [PDF](训练机制与稳定性/NeurIPS2026-Learning-to-Foresee/paper.pdf) |
| [Better Rollouts](训练机制与稳定性/NeurIPS2026-Better-Rollouts/README.md) | 关注点是收集什么 rollout，而不只是在已有 rollout 上换 loss。 | 待补PDF |
| [REOPOLD / Relaxed OPD](训练机制与稳定性/NeurIPS2026-Relaxed-OPD/README.md) | 强烈负监督应适度，低信息 token 不必占据同样的学习预算。 | [PDF](训练机制与稳定性/NeurIPS2026-Relaxed-OPD/paper.pdf) |
| [SEAD](训练机制与稳定性/NeurIPS2026-SEAD/README.md) | 监督要匹配学生当前能力：太难的前缀和已掌握的 token 都可能浪费训练。 | [PDF](训练机制与稳定性/NeurIPS2026-SEAD/paper.pdf) |
| [SimCT](训练机制与稳定性/NeurIPS2026-SimCT/README.md) | 两模型 tokenizer 不同时，先找到可比较的文本单位，比直接比较 token id 更重要。 | [PDF](训练机制与稳定性/NeurIPS2026-SimCT/paper.pdf) |
| [Many Faces of OPD](训练机制与稳定性/NeurIPS2026-Many-Faces-OPD/README.md) | teacher 强不够，还要检查前缀兼容性、TopK 梯度与特权信息是否能迁移。 | [PDF](训练机制与稳定性/NeurIPS2026-Many-Faces-OPD/paper.pdf) |
| [TOP-D](训练机制与稳定性/NeurIPS2026-Trust-Region-Policy-Distillation/README.md) | teacher 太远时先构造较近的目标，再用受控更新逐步接近它。 | [PDF](训练机制与稳定性/NeurIPS2026-Trust-Region-Policy-Distillation/paper.pdf) |
| [PW-OPSD / Teacher Token Reliability](训练机制与稳定性/NeurIPS2026-Teacher-Token-Reliability/README.md) | 高 entropy 既可能是坏分支，也可能是多种好解法；可靠性不能只看 entropy。 | [PDF](训练机制与稳定性/NeurIPS2026-Teacher-Token-Reliability/paper.pdf) · [精读](训练机制与稳定性/NeurIPS2026-Teacher-Token-Reliability/analysis.md) |
| [LGR / Supervision Fidelity Decay](训练机制与稳定性/NeurIPS2026-Supervision-Fidelity-Decay/README.md) | 学生前缀越长，teacher 的纠正能力可能越弱，不能默认后段监督始终有效。 | [PDF](训练机制与稳定性/NeurIPS2026-Supervision-Fidelity-Decay/paper.pdf) |
| [TrOPD](训练机制与稳定性/arXiv2026-TrOPD/README.md) | teacher/student 分歧很大时，先判断哪些局部监督值得信任。 | [PDF](训练机制与稳定性/arXiv2026-TrOPD/paper.pdf) |
| [Dr.OPD](训练机制与稳定性/arXiv2026-Dr.OPD/README.md) | teacher 纠正信号是否值得加强，要看它与任务奖励提升方向是否一致。 | [PDF](训练机制与稳定性/arXiv2026-Dr.OPD/paper.pdf) |
| [CorrGRPO](训练机制与稳定性/arXiv2026-CorrGRPO/README.md) | 多奖励时，advantage 的分母也会决定学习强度；相关性与数值尺度应分开看。 | [PDF](训练机制与稳定性/arXiv2026-CorrGRPO/paper.pdf) |
| [AOPD](训练机制与稳定性/ICLR2027-AOPD/README.md) | teacher 喜欢的动作，应按当前 student 沿这一步能否成功来评价。 | [PDF](训练机制与稳定性/ICLR2027-AOPD/paper.pdf) · [精读](训练机制与稳定性/ICLR2027-AOPD/analysis.md) |
| [AC2](训练机制与稳定性/arXiv2026-AC2/README.md) | critic 足够可靠时，不需要完整 rollout 也能得到任务训练信号。 | [PDF](训练机制与稳定性/arXiv2026-AC2/paper.pdf) · [精读](训练机制与稳定性/arXiv2026-AC2/analysis.md) |
| [STOPD](训练机制与稳定性/2026-STOPD/README.md) | 局部 OPD 不必等回答结束；随机截断的后段缺失可以用逆概率加权补偿。 | [PDF](训练机制与稳定性/2026-STOPD/paper.pdf) · [精读](训练机制与稳定性/2026-STOPD/analysis.md) |
| [DN-MOPD](训练机制与稳定性/arXiv2026-DN-MOPD/README.md) | 老师分配正确还不够；跨领域反馈尺度失衡会让一个领域压过其他领域。 | [PDF](训练机制与稳定性/arXiv2026-DN-MOPD/paper.pdf) · [精读](训练机制与稳定性/arXiv2026-DN-MOPD/analysis.md) |
| [DIAL-OPD](训练机制与稳定性/ICLR2027-DIAL-OPD/README.md) | 大分歧未必值得学；两个模型都低概率的位置也会出现很大的 log-ratio。 | [PDF](训练机制与稳定性/ICLR2027-DIAL-OPD/paper.pdf) · [精读](训练机制与稳定性/ICLR2027-DIAL-OPD/analysis.md) |
| [PowerOPD](训练机制与稳定性/arXiv2026-PowerOPD/README.md) | 相同比值的低概率分歧不该同样强；直接设计教学系数的概率几何。 | [PDF](训练机制与稳定性/arXiv2026-PowerOPD/paper.pdf) · [精读](训练机制与稳定性/arXiv2026-PowerOPD/analysis.md) |
| [DuoOPD](训练机制与稳定性/arXiv2026-DuoOPD/README.md) | 学生已经做对时保护成功；学生做错且老师做对时，用老师已验证的答案支持纠错。 | [PDF](训练机制与稳定性/arXiv2026-DuoOPD/paper.pdf) · [精读](训练机制与稳定性/arXiv2026-DuoOPD/analysis.md) |

## 语言模型与推理

| 方法 | 一句话认知 | 材料 |
|---|---|---|
| [CoTrek](语言模型与推理/NeurIPS2026-CoTrek/README.md) | 它属于长 CoT 的 OPD 扩展问题，具体加速机制需要正文确认。 | 待补PDF |
| [CoPD / Co-Evolving Policy Distillation](语言模型与推理/NeurIPS2026-Co-Evolving-Policy-Distillation/README.md) | 专家还在训练时就互相教，可能比专家完全分化后再合并更容易吸收。 | [PDF](语言模型与推理/NeurIPS2026-Co-Evolving-Policy-Distillation/paper.pdf) |
| [VPD / Variational Policy Distillation](语言模型与推理/NeurIPS2026-Variational-Policy-Distillation/README.md) | 会阅读反馈的 teacher 也需要继续进步，固定的反馈解释能力可能成为瓶颈。 | [PDF](语言模型与推理/NeurIPS2026-Variational-Policy-Distillation/paper.pdf) |
| [Rethinking OPSD](语言模型与推理/NeurIPS2026-Rethinking-OPSD/README.md) | teacher 看答案后不再需要探索，但 student 仍需要；这种偏好未必适合长思考。 | [PDF](语言模型与推理/NeurIPS2026-Rethinking-OPSD/paper.pdf) · [精读](语言模型与推理/NeurIPS2026-Rethinking-OPSD/analysis.md) |
| [OPHSD / Harness Self-Distillation](语言模型与推理/NeurIPS2026-Harness-Self-Distillation/README.md) | 额外推理流程可以只在训练中当脚手架，再把它的帮助内化到模型里。 | [PDF](语言模型与推理/NeurIPS2026-Harness-Self-Distillation/paper.pdf) |
| [RWOPD / NL-to-SVA](语言模型与推理/NeurIPS2026-NL-to-SVA-OPD/README.md) | 语法像标准答案还不够，应让真正性质等价的 SVA 决定哪些监督值得用。 | [PDF](语言模型与推理/NeurIPS2026-NL-to-SVA-OPD/paper.pdf) |
| [d-OPSD / Self Future dLLMs](语言模型与推理/NeurIPS2026-Self-Future-dLLMs/README.md) | 扩散语言模型的监督单位应跟去噪步骤对齐，而不是硬套自回归前缀。 | [PDF](语言模型与推理/NeurIPS2026-Self-Future-dLLMs/paper.pdf) |
| [OPSD / Self-Distilled Reasoner](语言模型与推理/arXiv2026-OPSD/README.md) | 同一个模型看到参考解法后，可以给只看题目的自己提供更好的局部指导。 | [PDF](语言模型与推理/arXiv2026-OPSD/paper.pdf) |
| [RLCSD](语言模型与推理/arXiv2026-RLCSD/README.md) | 看参考后改变的很多只是风格；正确/错误参考作对照，更可能留下内容信号。 | [PDF](语言模型与推理/arXiv2026-RLCSD/paper.pdf) |
| [DF-OPD](语言模型与推理/arXiv2026-DF-OPD/README.md) | 题目很少也能产生很多不同前缀；teacher 可以先自己出题再教 student。 | [PDF](语言模型与推理/arXiv2026-DF-OPD/paper.pdf) |

## 多模态与交互

| 方法 | 一句话认知 | 材料 |
|---|---|---|
| [PRISM / Black-box Multimodal RL](多模态与交互/NeurIPS2026-Black-Box-Multimodal-RL/README.md) | 拿不到 teacher logits 时，也可以先对齐响应分布再做 RL。 | [PDF](多模态与交互/NeurIPS2026-Black-Box-Multimodal-RL/paper.pdf) |
| [Fewer Tokens, Fewer Layers](多模态与交互/NeurIPS2026-Vision-Token-Pruning-VLM/README.md) | 视觉输入 token 数与模型层数，也会影响蒸馏后的推理成本。 | 待补PDF |
| [VC-OPD](多模态与交互/NeurIPS2026-VC-OPD/README.md) | teacher偏好的token不一定依赖图像；应更多学习由真实视觉证据支撑的决策。 | [PDF](多模态与交互/NeurIPS2026-VC-OPD/paper.pdf) |
| [Vision-OPD](多模态与交互/NeurIPS2026-Vision-OPD/README.md) | 同一个模型放大关键区域后看得更清楚，这种感知优势可以教给看全图的 student。 | [PDF](多模态与交互/NeurIPS2026-Vision-OPD/paper.pdf) |
| [VA-OPD / Visual Advantage](多模态与交互/NeurIPS2026-Visual-Advantage-OPD/README.md) | 真正视觉相关的 token 很少，按所有语言 token 平均会稀释视觉监督。 | [PDF](多模态与交互/NeurIPS2026-Visual-Advantage-OPD/paper.pdf) |
| [GUI-SD](多模态与交互/NeurIPS2026-GUI-Grounding-Self-Distillation/README.md) | GUI 定位需要把监督集中到可靠且影响坐标的数字位置。 | [PDF](多模态与交互/NeurIPS2026-GUI-Grounding-Self-Distillation/paper.pdf) |
| [ReDraft](多模态与交互/ICLR2027-ReDraft/README.md) | 新任务几乎不会做时，先把学生回答修订成它能学的正确文本，可能更容易起步。 | [PDF](多模态与交互/ICLR2027-ReDraft/paper.pdf) |
| [PIVOT](多模态与交互/ICLR2027-PIVOT/README.md) | 全部失败时，最终奖励不给差异；失败诊断可以形成额外的 teacher 监督。 | [PDF](多模态与交互/ICLR2027-PIVOT/paper.pdf) |

## 其他应用

| 方法 | 一句话认知 | 材料 |
|---|---|---|
| [D-OPSD / Few-step Image Diffusion](其他应用/NeurIPS2026-D-OPSD/README.md) | 少步扩散模型继续学新概念时，应沿自己的少步轨迹监督，避免破坏原生成方式。 | [PDF](其他应用/NeurIPS2026-D-OPSD/paper.pdf) |
| [On-Policy Hindsight / Early Risk](其他应用/NeurIPS2026-On-Policy-Hindsight-Distillation/README.md) | 未来信息可以作为训练中的 hindsight，但部署时只能用当时可见信息。 | 待补PDF |
| [Draft-OPD](其他应用/arXiv2026-Draft-OPD/README.md) | 投机解码的 draft 错误不能只被最终正确文本覆盖，必须保留被拒绝的提议来学。 | [PDF](其他应用/arXiv2026-Draft-OPD/paper.pdf) |

## 本地代码与实验入口

论文代码仓库均保留在`/usr/data`；各论文目录中的`main`与实验软链接是本机入口，不随GitHub文献仓库提交。官方公开代码URL保留在论文README中，本机链接映射见仓库根目录的`local-links.json`。历史来源清单保留在`_archive/`；当前文件状态以本目录manifest和检查报告为准。
