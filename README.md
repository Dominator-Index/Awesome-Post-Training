# Awesome Post-Training

按训练信号、组件和研究问题整理大模型后训练论文、数学推导与研究笔记。

## 阅读入口

- [On-Policy Distillation / Self-Distillation](On-Policy-Distillation/README.md)：45条论文记录，按四个研究问题类别整理。
- [理论与机制分析](理论与机制分析/README.md)：1条论文记录，研究后训练的理论解释、学习动力学与参数更新机制。
- [方法RoadMap](On-Policy-Distillation/RoadMap/README.md)：SFT、PPO、GRPO、OPD、OPSD的数学组件与论文改动位置。
- [论文认知与关键公式](On-Policy-Distillation/RoadMap/论文一句话认知.md) · [交互浏览器](On-Policy-Distillation/RoadMap/index.html)。
- [研究idea](On-Policy-Distillation/idea/)：区分已有理论、待验证假设和实验观察。

## 归档状态

2026-10-16新增Fisher参数更新稀疏性论文，主分类为“理论与机制分析”，并在OPD目录提供交叉入口。全库共46条独立论文记录，42篇有PDF，4篇待补；OPD原有45条记录不变，交叉链接不重复计数。新增论文的版本、页数与SHA256见其README。

2026-10-15整理：45条记录，41篇有canonical PDF，4篇受OpenReview访问限制仍待补。每条有分类与方法入口；完整精读文件数量、PDF页数与SHA256见`On-Policy-Distillation/download_summary.json`（本地文件）。不将预印本、Under Review、方法概览或探索实验等同于论文录用、完整精读或成功复现。

## 本地布局

```text
/usr/data/Awesome-Post-Training/                    # 本仓库
├── 理论与机制分析/
│   └── arXiv2026-Fisher-Update-Sparsity/
└── On-Policy-Distillation/
    ├── 训练机制与稳定性/
    ├── 语言模型与推理/
    ├── 多模态与交互/
    ├── 其他应用/
    ├── RoadMap/
    └── idea/
```

工作区`awesome-exploration-rl/Awesome-Post-Training`软链接指向整个仓库。原工作区顶层`On-Policy-Distillation`已迁移，不再保留该路径。

本地代码checkout与实验输出保留在`/usr/data`，通过论文目录的`main`或实验软链接访问；这些本机入口不随本仓库提交。映射记录见`local-links.json`（本地文件），迁移记录见`migration-20261015.json`（本地文件）。PDF、论文笔记、draw.io源图与导出图册随仓库归档；论文版权归原作者或出版方。
