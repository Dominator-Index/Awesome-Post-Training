# OPSD — Self-Distilled Reasoner

- **完整题名**：Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
- **来源**：[arXiv:2601.18734](https://arxiv.org/abs/2601.18734)
- **本地 PDF**：[paper.pdf](paper.pdf)，v3。
- **归档名称**：arXiv2026-OPSD；年份表示 arXiv 首次发表年份。
- **官方代码**：[siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD)
- **本地代码**：[main](main) 是指向 `/usr/data/OPSD` 的软链接。

## 核心理解：老师可以看参考解法

Teacher 可以获得特权信息，例如经过验证的推理轨迹（verified reasoning traces）；student 的输入只有题目，不包含这段参考解法。

Student 先按自己的策略生成推理和回答。训练时，teacher 在 student 生成的前缀上，结合参考推理过程预测下一个 token，再通过逐 token 的分布监督训练 student。这里的“只有题目”不排除自回归生成时使用 student 自己已经生成的前缀。

因此，可以理解为：**老师拿着参考解题过程，指导学生独立作答。**Teacher 不一定具有更强的独立解题能力，但仍需能够理解参考解法，并把它转化成适合学生当前前缀的可靠指导。

更完整的讨论见：[OPSD 理解：老师可以看答案](../../OPSD理解-老师可以看答案.md)。
