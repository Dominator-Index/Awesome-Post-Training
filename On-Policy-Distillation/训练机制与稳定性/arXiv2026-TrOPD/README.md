# Trust Region On-Policy Distillation (TrOPD)

- Authors: Xingrun Xing, Haoqing Wang, Boyan Gao, Ziheng Li, Yehui Tang
- arXiv: [2606.01249](https://arxiv.org/abs/2606.01249), v3
- PDF: [paper.pdf](paper.pdf)

## Core idea

TrOPD classifies student-generated tokens by whether teacher supervision is reliable, using teacher–student decoding agreement to define a trust region. It applies on-policy updates in reliable regions, handles outliers with clipping, masking, or forward-KL estimation, and adds teacher-prefix imitation as off-policy guidance.

## Distinction from Trust Region Policy Distillation (TOP-D)

This is a separate paper from [Trust Region Policy Distillation](../NeurIPS2026-Trust-Region-Policy-Distillation/). TrOPD uses teacher–student decoding agreement to identify reliable token regions, changes the estimator for outlier regions, and adds teacher-prefix guidance. TOP-D's core mechanism is to interpolate between the teacher and student to construct a proximal teacher, then distill against that interpolated target. The two papers therefore differ in how they form or select the supervision target.
