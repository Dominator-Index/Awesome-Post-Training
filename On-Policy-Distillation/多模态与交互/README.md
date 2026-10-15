# 多模态与交互

视觉证据、视觉token、GUI与多轮agent，及相关SFT修订对照。

[返回总目录](../README.md) · [RoadMap](../RoadMap/README.md)

| 方法 | 一句话认知 | 材料 |
|---|---|---|
| [PRISM / Black-box Multimodal RL](NeurIPS2026-Black-Box-Multimodal-RL/README.md) | 拿不到 teacher logits 时，也可以先对齐响应分布再做 RL。 | [PDF](NeurIPS2026-Black-Box-Multimodal-RL/paper.pdf) |
| [Fewer Tokens, Fewer Layers](NeurIPS2026-Vision-Token-Pruning-VLM/README.md) | 视觉输入 token 数与模型层数，也会影响蒸馏后的推理成本。 | 待补PDF |
| [VC-OPD](NeurIPS2026-VC-OPD/README.md) | teacher偏好的token不一定依赖图像；应更多学习由真实视觉证据支撑的决策。 | [PDF](NeurIPS2026-VC-OPD/paper.pdf) |
| [Vision-OPD](NeurIPS2026-Vision-OPD/README.md) | 同一个模型放大关键区域后看得更清楚，这种感知优势可以教给看全图的 student。 | [PDF](NeurIPS2026-Vision-OPD/paper.pdf) |
| [VA-OPD / Visual Advantage](NeurIPS2026-Visual-Advantage-OPD/README.md) | 真正视觉相关的 token 很少，按所有语言 token 平均会稀释视觉监督。 | [PDF](NeurIPS2026-Visual-Advantage-OPD/paper.pdf) |
| [GUI-SD](NeurIPS2026-GUI-Grounding-Self-Distillation/README.md) | GUI 定位需要把监督集中到可靠且影响坐标的数字位置。 | [PDF](NeurIPS2026-GUI-Grounding-Self-Distillation/paper.pdf) |
| [ReDraft](ICLR2027-ReDraft/README.md) | 新任务几乎不会做时，先把学生回答修订成它能学的正确文本，可能更容易起步。 | [PDF](ICLR2027-ReDraft/paper.pdf) |
| [PIVOT](ICLR2027-PIVOT/README.md) | 全部失败时，最终奖励不给差异；失败诊断可以形成额外的 teacher 监督。 | [PDF](ICLR2027-PIVOT/paper.pdf) |
