# 语言模型与推理

自蒸馏、参考解法、语言反馈、推理探索、数据来源与文本生成。

[返回总目录](../README.md) · [RoadMap](../RoadMap/README.md)

| 方法 | 一句话认知 | 材料 |
|---|---|---|
| [CoTrek](NeurIPS2026-CoTrek/README.md) | 它属于长 CoT 的 OPD 扩展问题，具体加速机制需要正文确认。 | 待补PDF |
| [CoPD / Co-Evolving Policy Distillation](NeurIPS2026-Co-Evolving-Policy-Distillation/README.md) | 专家还在训练时就互相教，可能比专家完全分化后再合并更容易吸收。 | [PDF](NeurIPS2026-Co-Evolving-Policy-Distillation/paper.pdf) |
| [VPD / Variational Policy Distillation](NeurIPS2026-Variational-Policy-Distillation/README.md) | 会阅读反馈的 teacher 也需要继续进步，固定的反馈解释能力可能成为瓶颈。 | [PDF](NeurIPS2026-Variational-Policy-Distillation/paper.pdf) |
| [Rethinking OPSD](NeurIPS2026-Rethinking-OPSD/README.md) | teacher 看答案后不再需要探索，但 student 仍需要；这种偏好未必适合长思考。 | [PDF](NeurIPS2026-Rethinking-OPSD/paper.pdf) · [精读](NeurIPS2026-Rethinking-OPSD/analysis.md) |
| [OPHSD / Harness Self-Distillation](NeurIPS2026-Harness-Self-Distillation/README.md) | 额外推理流程可以只在训练中当脚手架，再把它的帮助内化到模型里。 | [PDF](NeurIPS2026-Harness-Self-Distillation/paper.pdf) |
| [RWOPD / NL-to-SVA](NeurIPS2026-NL-to-SVA-OPD/README.md) | 语法像标准答案还不够，应让真正性质等价的 SVA 决定哪些监督值得用。 | [PDF](NeurIPS2026-NL-to-SVA-OPD/paper.pdf) |
| [d-OPSD / Self Future dLLMs](NeurIPS2026-Self-Future-dLLMs/README.md) | 扩散语言模型的监督单位应跟去噪步骤对齐，而不是硬套自回归前缀。 | [PDF](NeurIPS2026-Self-Future-dLLMs/paper.pdf) |
| [OPSD / Self-Distilled Reasoner](arXiv2026-OPSD/README.md) | 同一个模型看到参考解法后，可以给只看题目的自己提供更好的局部指导。 | [PDF](arXiv2026-OPSD/paper.pdf) |
| [RLCSD](arXiv2026-RLCSD/README.md) | 看参考后改变的很多只是风格；正确/错误参考作对照，更可能留下内容信号。 | [PDF](arXiv2026-RLCSD/paper.pdf) |
| [DF-OPD](arXiv2026-DF-OPD/README.md) | 题目很少也能产生很多不同前缀；teacher 可以先自己出题再教 student。 | [PDF](arXiv2026-DF-OPD/paper.pdf) |
