# Coding Agent 方法知识库的选择准确率

```
date                 2026-10-16
git_commit           2687e26
worktree_dirty       True
source_sha256        3cdf597b385bbba0a7612ee5cdf7167d2b4e887ef6b81c3a6365ceabcd90541f
dataset              method_notes_selection.jsonl (75 papers)
notes                11
max_notes_per_paper  2
labels               Claude Code, after seeing selector output; pending owner review
```

## 1. 总体

- 精确率 **0.925**（选中的“论文-类别”对中，正确的比例）
- 召回率 **0.860**（应选的“论文-类别”对中，被选中的比例）
- 完全一致 **0.867**（选中的集合与标注完全相同，含“都不选”）

## 2. 按类别

| 类别 | 选中 | 应选 | 正确 | 精确率 | 召回率 |
|---|---|---|---|---|---|
| attention | 7 | 6 | 6 | 0.857 | 1.000 |
| contrastive | 5 | 5 | 5 | 1.000 | 1.000 |
| diffusion | 5 | 7 | 5 | 1.000 | 0.714 |
| distillation | 2 | 4 | 2 | 1.000 | 0.500 |
| graph_neural_networks | 5 | 6 | 5 | 1.000 | 0.833 |
| lora | 5 | 6 | 5 | 1.000 | 0.833 |
| mixture_of_experts | 5 | 4 | 4 | 0.800 | 1.000 |
| policy_gradient | 5 | 6 | 4 | 0.800 | 0.667 |
| preference_optimization | 4 | 3 | 3 | 0.750 | 1.000 |
| quantization | 6 | 6 | 6 | 1.000 | 1.000 |
| vae | 4 | 4 | 4 | 1.000 | 1.000 |

## 3. 不一致的论文

| arXiv | 标题 | 多选 | 漏选 |
|---|---|---|---|
| 2506.05814v1 | Positional Encoding meets Persistent Homology on Graphs | attention | graph_neural_networks |
| 2610.03112 | Building Interpretable Feature Representations for Resume-Vacancy Matc | — | distillation, lora |
| 2610.07446 | CogAdapt: Cognition-informed Sparse Adaptation of Code LLMs | mixture_of_experts | — |
| 2610.12877 | TAME-Nav: Temporal Asymmetric Mixture-of-Experts Distillation for Agil | — | distillation |
| 2602.02313v2 | Interpreting and Controlling LLM Reasoning through Integrated Policy G | policy_gradient | — |
| 2512.22796v1 | Parallel Diffusion Solver via Residual Dirichlet Policy Optimization | — | diffusion |
| 2509.04897v2 | PLaMo 2 Technical Report | preference_optimization | — |
| 2508.05613v1 | Cooper: Co-Optimizing Policy and Reward Models in Reinforcement Learni | — | policy_gradient |
| 2601.05432v1 | Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geoloca | — | policy_gradient |
| 2601.10714v1 | Alterbute: Editing Intrinsic Attributes of Objects in Images | — | diffusion |

