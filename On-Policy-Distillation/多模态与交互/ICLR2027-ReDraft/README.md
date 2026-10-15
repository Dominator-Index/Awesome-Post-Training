# ReDraft

**ReDraft, Don’t Just Distill: Reference-Driven Revision for Continual MLLM Post-Training**

- **状态**：ICLR 2027 Under Review（PDF 首页明确标注，尚未录用）。
- **作者**：匿名，双盲审稿中。
- **主题分类**：多模态与交互；持续后训练、能力获取与遗忘。
- **本地论文**：[paper.pdf](paper.pdf)，39 页。
- **OpenReview**：[论文页面](https://openreview.net/forum?id=Et89lO3Lc2) · [原始 PDF 链接](https://openreview.net/pdf?id=Et89lO3Lc2)。
- **文件来源**：用户提供的 PDF。按用户要求，目录命名为 `ICLR2027-ReDraft`，放在 `On-Policy-Distillation` 根目录。

## 核心问题

多模态模型的持续后训练需要同时学习新能力、保留旧能力。作者认为，SFT 的明确专家目标有利于从接近零准确率开始学习，但可能造成较大策略变化和遗忘；RLVR 与自蒸馏更贴近当前策略，却在模型尚不会完成任务时缺乏足够有效的学习信号。

## 方法概览

1. 模型先生成自己的回答。
2. 对错误回答，让模型参考专家答案进行修改。
3. 仅保留通过验证器检查的修改结果。
4. 使用这些结果进行监督微调。

目标是让训练答案既明确、正确，又接近模型现有的表达与行为。最终训练使用经过验证的修订文本作为监督目标；不能直接将其等同于普通 OPSD 的逐 token teacher 分布匹配。

## 摘要报告的结果

作者在 Qwen2.5-VL-3B/7B 的 Counting、Clock Reading、Jigsaw 上评估，其中两个任务初始准确率接近零。摘要报告跨实验汇总结果：

| 方法 | 新任务提升（百分点） | 原有任务下降（百分点） |
|---|---:|---:|
| ReDraft | 56.9 | 1.5 |
| SFT | 52.9 | 16.6 |
| OPSD | 19.3 | 6.2 |

这些是论文作者报告的结果，尚未独立复现。本文档是基于摘要的归档说明，不是完整精读或审稿。

## 文件记录

原始上传文件名：`9629_ReDraft_Don_t_Just_Distil.pdf`，归档后统一命名为 `paper.pdf`。来源和 SHA-256 见 [SOURCE.txt](SOURCE.txt)，结构化记录见 `manifest.jsonl`（本地文件）。PDF 已通过文件头及 pdfinfo 检查，并人工核对首页题名。
