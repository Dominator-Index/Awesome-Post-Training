# Draft-OPD：给投机解码的 draft model 做 OPD

## 论文与归档

- **完整题名**：Draft-OPD: On-Policy Distillation for Speculative Draft Models。
- **项目页**：https://www.haodilei.top/draft-opd/
- **论文**：[arXiv:2605.29343](https://arxiv.org/abs/2605.29343)，本目录保存 v3，2026-09-18，16 页。
- **官方代码**：[Simplified-Reasoning/Draft-OPD](https://github.com/Simplified-Reasoning/Draft-OPD)。项目页旧仓库链接会重定向到这里。
- 本地代码位于 `/usr/data/Draft-OPD`，本目录 `main` 为软链接。未确认会议状态，按 arXiv 预印本归档。

## 1. 要解决什么问题？

投机解码中，小 draft model 先提出一段 token，大 target model 并行验证。draft 与 target 越一致，平均接受长度越长，加速越明显。

传统 SFT 使用 target 生成的固定轨迹，训练前缀来自 target；实际推理却要验证 draft 自己提出的块。因此，继续 SFT 可能很快遇到接受长度瓶颈。

直接搬常规 OPD 也有障碍：drafter 独立生成整段序列不可靠；如果只保存经过 target 验证的最终文本，被拒绝的 draft token 又消失了，无法学习真正限制接受长度的错误。

## 2. 方法：稳定的主轨迹 + draft 错误重放

1. 使用 target 辅助的投机解码生成稳定的主轨迹。
2. 记录各 draft block 的起点作为 anchor，以及验证暴露的错误位置。
3. 从这些 anchor 重放 draft 提议，使 target 和 draft 在相同的 draft 前缀上计算概率。
4. 同时保留 accepted 与 rejected proposals，分别构造蒸馏目标。

设 anchor 上下文为 $c_m$，draft block 为 $d_m$，两模型都在 $(c_m,d_{m,<k})$ 上打分。这意味着 target 也要看到被拒绝的 draft 前缀，而不仅仅是最终经过验证的文本。

## 3. 为什么分两种 KL？

记 target 分布为 $p$，draft 分布为 $q$：

- accepted token 用 forward KL：$D_{\mathrm{KL}}(p\|q)$，鼓励 draft 覆盖 target 分布。
- rejected token 用 reverse KL：$D_{\mathrm{KL}}(q\|p)$，抑制 draft 偏好、但 target 不支持的模式。

$$
L_{\mathrm{acc}}=\frac{1}{|I_{\mathrm{acc}}|}\sum_{(m,k)\in I_{\mathrm{acc}}}D_{\mathrm{KL}}(p_{m,k}\|q_{m,k}),
$$
$$
L_{\mathrm{rej}}=\frac{1}{Z}\sum_{(m,k)\in I_{\mathrm{rej}}}\gamma^{k-1}D_{\mathrm{KL}}(q_{m,k}\|p_{m,k}),\qquad
Z=\sum_{(m,k)\in I_{\mathrm{rej}}}\gamma^{k-1}.
$$

最终损失为两者的加权平均；实验使用相等的两项权重。位置衰减按 block 内位置 $k$ 计算：较早的拒绝会使后续整段无法被接受，所以应优先纠正早期错误。

这里的 rejected 集合包括首次拒绝之后的后缀，它们受到错误前缀影响，不能都当成同等可靠的错误标签。

## 4. 和 ReDraft、普通 OPD 的区别

| 方法 | 核心处理 | 主要目标 |
|---|---|---|
| ReDraft | student 回答修订成验证通过的文本，再做 SFT | 改善学生回答能力 |
| 普通 OPD | student 生成完整回答，teacher 在学生前缀上给分布反馈 | 能力蒸馏 |
| Draft-OPD | target 辅助主轨迹，重放 draft proposals，包括被拒绝的部分，再做 KL 蒸馏 | 提高投机接受长度及推理速度 |

因此 Draft-OPD 不是“先把错误答案改对，再把正确答案做 SFT”。它确实在 draft 引起的状态上使用 OPD 信号，训练对象和评估目标也与独立解题 student 不同。

## 5. 实验怎么理解？

论文在 Qwen3 系列、数学、代码及其他任务上测试，报告部分设置超过 5 倍无损加速。项目页 thinking、temperature=0 的两个模型平均速度为约 4.88 倍，相比 EAGLE-3 和 DFlash 提升约 23% 和 13%。这些加速数字依赖模型、任务、硬件、温度和服务配置，不能视为所有场景的固定收益。

“无损”来自正确执行投机解码的验证与采样流程：它保留 target 的输出分布。并不是说 draft model 本身达到了 target 的回答质量。

## 6. 对我们关心的错误前缀有什么启发？

这篇明确保留了被拒绝的提议，并区分前缀可靠程度：错误不能只被最终正确轨迹覆盖掉；后续错误也不应与首个错误同权。

一个可迁移的思路是让普通 OPD 同时利用可靠主轨迹与学生错误分支，对早期偏离加强反馈、对后续被污染的位置降权。但普通数学 OPD 没有投机验证器直接提供 accepted/rejected 标记，需要另行定义诊断机制。这是延伸设想，本文没有验证。

## 阅读定位

§3–4 解释 rollout 矛盾、anchor replay 和两种 KL；§5 给出实验；Appendix B 解释 KL 方向选择。来源和代码快照见 `SOURCE.txt` / `manifest.jsonl`。

代码已克隆，未安装依赖或运行训练。
