# VC-OPD

**认知：** teacher偏好的token不一定依赖图像；应更多学习由真实视觉证据支撑的决策。

**方法：** 在相同学生前缀上用原图和损坏图各评分一次，图像条件下的teacher决定方向，反事实概率差经sigmoid与权重和归一化后调节力度。

## 关键公式

$$
\Delta_t=\log p_T^+(y_t)-\log p_T^-(y_t),\quad w_t=\sigma((\Delta_t-\tau_v)/T_v),\quad\bar w_t=\frac{Nw_t}{\sum_jw_j+\epsilon},\quad A_t^{VC}=\bar w_t\widetilde A_t^{dir}
$$

$p_T^+$ 与 $p_T^-$ 分别是原图和反事实图像下、在同一个student前缀上的teacher概率。权重只改变力度；方向仍来自原图teacher与student的分歧，$\widetilde A_t^{dir}$是原文归一化后的方向信号。

**边界：** 扰动差是视觉依赖的代理，不是严格因果效应；原OpenReview版本未核对。

[PDF](paper.pdf) · [arXiv论文](https://arxiv.org/abs/2610.14939) · [RoadMap](../../RoadMap/README.md)

本次核对摘要和方法，不将其标记为完整精读，也不以目录中的NeurIPS标签推断arXiv与原OpenReview版本完全相同。
