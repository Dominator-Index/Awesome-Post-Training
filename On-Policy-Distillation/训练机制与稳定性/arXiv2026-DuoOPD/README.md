# DuoOPD

**认知：student 已经答对时，要保护它的成功；teacher 有已验证成功时，用这个成功支撑纠错。**

**方法：verifier 决定 token 系数的正负号；四种师生联合结果决定使用原始评分、参考条件评分或同任务共享正权重。**

- [PDF](paper.pdf)
- [完整数学、四种情况、消融与局限](analysis.md)
- [论文](https://arxiv.org/abs/2609.33711v1)
- [官方代码](https://github.com/YongYuanDeAo/DuoOPD)
- [本地源码](main) → `/usr/data/DuoOPD`。git 拉取失败后取得官方 archive 快照，非完整 git checkout；来源见 `source.json`。
- 目录按 arXiv2026 归档。

关键边界：正确答案不保证中间推理正确；该方法改变原始 OPD 目标，也需要完整结果进行 verification。
