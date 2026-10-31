# OPSD Profiling Slides

OPSD实验诊断用的Beamer slides：训练时能算哪些统计量、要付出什么成本、各自回答什么问题。直接看PDF即可，`.tex`是源码。

| 文件 | 语言 | 页数 | 内容 |
|---|---|---|---|
| [idea003-stats.pdf](idea003-stats.pdf) | 中文 | 37 | 只讲[idea 003](../idea/003-PMI-OPSD-特权信号分解与前缀兼容度诊断.md)用到的统计量，按假设H1–H3组织：记号、统计量地图、E0–E4、混杂与对策、预先写下的判据、执行计划 |
| [opsd-profiling-zh.pdf](opsd-profiling-zh.pdf) | 中文 | 93 | OPSD实验的全部统计量（信号、梯度、校准、表示、分支续写等）；开头6页是通俗导读，附录有缩写与符号索引 |
| [opsd-profiling-en.pdf](opsd-profiling-en.pdf) | English | 86 | 上一份的英文原版，没有导读 |

每份slides都先在记号页定义符号再使用。idea 003这份slides里，救回率与背书率的阈值（0.1、0.3）是补充的默认值，可调；H3那张柱状图是预测方向的示意，不是数据。

## 编译

```bash
latexmk -xelatex idea003-stats.tex
```

需要xelatex。中文版用xeCJK和PingFang SC、Songti SC字体（macOS自带）；主题为metropolis。`opsd-profiling-*-part2.tex`由同名主文件`\input`，不单独编译。编译中间文件已在`.gitignore`中忽略。
