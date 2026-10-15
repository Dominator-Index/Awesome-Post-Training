# PIVOT

**PIVOT: Pivot-Aware On Policy Self Distillation for Multi-Turn VLM Agents**

- **作者**：Jiazhou Zhou、Hu Zhou、Yucheng Chen、Jinyuan Qu、Ying-Cong Chen、Lei Zhang。
- **来源**：[arXiv:2609.35303](https://arxiv.org/abs/2609.35303)，v1，2026-09-28。
- **会议状态**：按用户提供的信息标注 ICLR 2027 Under Review；下载的 arXiv 版首页为 Preprint，尚未独立核实投稿状态，不表示录用。
- **本地 PDF**：[paper.pdf](paper.pdf)，34 页。
- **主题**：多轮 VLM agent、失败诊断、信用分配、GRPO 与 OPD 联合训练。

## 核心问题

多轮 agent 的 GRPO 训练存在两个问题：同一组轨迹都失败时缺少相对奖励梯度；最终回报难以精确定位造成失败的动作。作者通过反事实环境回滚实验研究关键失败步骤，并认为在所测环境中，回到 pivot 附近重新行动对恢复任务有重要作用。但真实环境未必支持回滚，训练时回滚也会增加成本。

Pivot 指在剩余步数预算约束下首次导致不可恢复失败的动作；它不能简单等同于轨迹中的任意错误。

## 方法概览

PIVOT（Pivot-Aware Internalized Visual On-Policy Training）在同一架构内使用三个功能角色：

1. **Analyzer**：根据视觉轨迹拼图和动作日志定位 pivot，并诊断失败原因。
2. **Teacher**：利用这些特权诊断信息，对失败轨迹上的 token 重新打分，teacher 分支不参与该学生更新的梯度传播。
3. **Student**：联合优化 GRPO 和带置信度门控的 OPD 目标。

训练阶段用诊断条件下的 token 信号提供纠正指导，避免实际环境回滚；测试时只保留 student，不附加 Analyzer、Teacher 或诊断提示。

这里“内化状态恢复”是作者对训练效果的表述，不表示参数更新能把正在运行的真实环境物理状态直接恢复。

## 与 ReDraft 的区别

ReDraft 构造验证通过的完整修订回答，再用于 SFT；PIVOT 在失败轨迹上用诊断信息构造 teacher 信号，并联合进行 GRPO 与 OPD 更新。两者都利用失败经验，但监督形式和训练目标不同。

## 摘要报告的结果

作者在五项多轮 VLM agent 任务上评估，覆盖网格谜题、3D 控制与导航、生成式推理。摘要报告 Qwen2.5-VL-3B 的总体准确率为 0.90，相对 SFT+GRPO 基线高约 8 个百分点；Qwen3-VL-2B 达到 0.92，相对该基线高约 12 个百分点。这些是作者报告的实验结果，尚未独立复现。

本文档是归档与方法概览，不是完整精读。来源、版本及文件哈希见 [SOURCE.txt](SOURCE.txt)，结构化记录见 [manifest.jsonl](manifest.jsonl)。
