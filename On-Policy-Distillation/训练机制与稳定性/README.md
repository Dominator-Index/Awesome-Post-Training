# 训练机制与稳定性

轨迹分布、采样预算、教师可靠性、反馈变换、权重、归一化；AC2/CorrGRPO作为RL对照。

[返回总目录](../README.md) · [RoadMap](../RoadMap/README.md)

| 方法 | 一句话认知 | 材料 |
|---|---|---|
| [f-OPD](NeurIPS2026-f-OPD/README.md) | 旧 rollout 可以复用，但应按新鲜程度决定它还值得信任多少。 | [PDF](NeurIPS2026-f-OPD/paper.pdf) |
| [AsyncOPD](NeurIPS2026-AsyncOPD/README.md) | 异步训练能提高吞吐，但 loss 方向和打分缓存决定旧数据是否还能用。 | [PDF](NeurIPS2026-AsyncOPD/paper.pdf) |
| [Noisy Expert / NAIL](NeurIPS2026-Behavior-Cloning-Optimality/README.md) | 当专家带噪声时，学生访问自己的状态再问专家，可能比离线模仿更有样本效率。 | [PDF](NeurIPS2026-Behavior-Cloning-Optimality/paper.pdf) |
| [Rock Tokens](NeurIPS2026-Rock-Tokens/README.md) | 高 loss 不一定意味着值得学；持续难以纠正的风格残差可能消耗大量梯度。 | [PDF](NeurIPS2026-Rock-Tokens/paper.pdf) |
| [vOPD / KL Control Variate](NeurIPS2026-KL-Control-Variate/README.md) | 用一个与当前动作无关的 baseline，可让 OPD 梯度更稳定而保持其期望。 | [PDF](NeurIPS2026-KL-Control-Variate/paper.pdf) · [精读](NeurIPS2026-KL-Control-Variate/analysis.md) |
| [EffOPD / Learning to Foresee](NeurIPS2026-Learning-to-Foresee/README.md) | 如果早期更新方向已经接近最终方向，可以沿当前方向多走一点来加速。 | [PDF](NeurIPS2026-Learning-to-Foresee/paper.pdf) |
| [Better Rollouts](NeurIPS2026-Better-Rollouts/README.md) | 关注点是收集什么 rollout，而不只是在已有 rollout 上换 loss。 | 待补PDF |
| [REOPOLD / Relaxed OPD](NeurIPS2026-Relaxed-OPD/README.md) | 强烈负监督应适度，低信息 token 不必占据同样的学习预算。 | [PDF](NeurIPS2026-Relaxed-OPD/paper.pdf) |
| [SEAD](NeurIPS2026-SEAD/README.md) | 监督要匹配学生当前能力：太难的前缀和已掌握的 token 都可能浪费训练。 | [PDF](NeurIPS2026-SEAD/paper.pdf) |
| [SimCT](NeurIPS2026-SimCT/README.md) | 两模型 tokenizer 不同时，先找到可比较的文本单位，比直接比较 token id 更重要。 | [PDF](NeurIPS2026-SimCT/paper.pdf) |
| [Many Faces of OPD](NeurIPS2026-Many-Faces-OPD/README.md) | teacher 强不够，还要检查前缀兼容性、TopK 梯度与特权信息是否能迁移。 | [PDF](NeurIPS2026-Many-Faces-OPD/paper.pdf) |
| [TOP-D](NeurIPS2026-Trust-Region-Policy-Distillation/README.md) | teacher 太远时先构造较近的目标，再用受控更新逐步接近它。 | [PDF](NeurIPS2026-Trust-Region-Policy-Distillation/paper.pdf) |
| [PW-OPSD / Teacher Token Reliability](NeurIPS2026-Teacher-Token-Reliability/README.md) | 高 entropy 既可能是坏分支，也可能是多种好解法；可靠性不能只看 entropy。 | [PDF](NeurIPS2026-Teacher-Token-Reliability/paper.pdf) · [精读](NeurIPS2026-Teacher-Token-Reliability/analysis.md) |
| [LGR / Supervision Fidelity Decay](NeurIPS2026-Supervision-Fidelity-Decay/README.md) | 学生前缀越长，teacher 的纠正能力可能越弱，不能默认后段监督始终有效。 | [PDF](NeurIPS2026-Supervision-Fidelity-Decay/paper.pdf) |
| [TrOPD](arXiv2026-TrOPD/README.md) | teacher/student 分歧很大时，先判断哪些局部监督值得信任。 | [PDF](arXiv2026-TrOPD/paper.pdf) |
| [Dr.OPD](arXiv2026-Dr.OPD/README.md) | teacher 纠正信号是否值得加强，要看它与任务奖励提升方向是否一致。 | [PDF](arXiv2026-Dr.OPD/paper.pdf) |
| [CorrGRPO](arXiv2026-CorrGRPO/README.md) | 多奖励时，advantage 的分母也会决定学习强度；相关性与数值尺度应分开看。 | [PDF](arXiv2026-CorrGRPO/paper.pdf) |
| [AOPD](ICLR2027-AOPD/README.md) | teacher 喜欢的动作，应按当前 student 沿这一步能否成功来评价。 | [PDF](ICLR2027-AOPD/paper.pdf) · [精读](ICLR2027-AOPD/analysis.md) |
| [AC2](arXiv2026-AC2/README.md) | critic 足够可靠时，不需要完整 rollout 也能得到任务训练信号。 | [PDF](arXiv2026-AC2/paper.pdf) · [精读](arXiv2026-AC2/analysis.md) |
| [STOPD](2026-STOPD/README.md) | 局部 OPD 不必等回答结束；随机截断的后段缺失可以用逆概率加权补偿。 | [PDF](2026-STOPD/paper.pdf) · [精读](2026-STOPD/analysis.md) |
| [DN-MOPD](arXiv2026-DN-MOPD/README.md) | 老师分配正确还不够；跨领域反馈尺度失衡会让一个领域压过其他领域。 | [PDF](arXiv2026-DN-MOPD/paper.pdf) · [精读](arXiv2026-DN-MOPD/analysis.md) |
| [DIAL-OPD](ICLR2027-DIAL-OPD/README.md) | 大分歧未必值得学；两个模型都低概率的位置也会出现很大的 log-ratio。 | [PDF](ICLR2027-DIAL-OPD/paper.pdf) · [精读](ICLR2027-DIAL-OPD/analysis.md) |
| [PowerOPD](arXiv2026-PowerOPD/README.md) | 相同比值的低概率分歧不该同样强；直接设计教学系数的概率几何。 | [PDF](arXiv2026-PowerOPD/paper.pdf) · [精读](arXiv2026-PowerOPD/analysis.md) |
| [DuoOPD](arXiv2026-DuoOPD/README.md) | 学生已经做对时保护成功；学生做错且老师做对时，用老师已验证的答案支持纠错。 | [PDF](arXiv2026-DuoOPD/paper.pdf) · [精读](arXiv2026-DuoOPD/analysis.md) |
