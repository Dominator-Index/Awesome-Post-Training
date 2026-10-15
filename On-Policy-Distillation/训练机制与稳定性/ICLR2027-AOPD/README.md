# ICLR2027 AOPD

**When to Follow the Teacher: Advantage-Based On-Policy Distillation**

- 状态：ICLR 2027 **Under Review**，PDF 首页明确标注；作者匿名。
- 原始文件：[原始上传文件](../../_sources/53047_When_to_Follow_the_Teach.pdf)，保留原件；本目录归档为 `paper.pdf`，19 页。
- [完整中文解读](analysis.md)：问题、目标函数、无偏估计、截断机制、理论、实验及与 Dr.OPD / RLCSD 的比较。
- 来源及哈希：`SOURCE.txt`、`manifest.jsonl`。
- 未识别到官方代码仓库，因此没有创建指向未知仓库的 `main`。

核心思路：teacher 推荐某个 token 不代表当前 student 能沿这条路径成功。AOPD 用 student 的任务 advantage 重新分配 teacher 监督，并通过已有学生 rollout 的终局奖励和概率比估计目标，而不额外生成反事实续写。
