# 画像推荐 vs 旧推荐：盲评 A/B

```
date            2026-10-17
git_commit      2687e26
worktree_dirty  True
source_sha256   068b10063559be6d49b2da5353bd0ab2ba485fbb0f0e2b888dca3012fffb0108
personas        10
feed_size       5
evaluator       gpt-4o-mini, both list orders
lists           results/2026-10-17_feed_persona_ab_lists.json
```

**胜负（两种顺序结论一致才算胜）**：画像推荐 8，旧推荐 0，平局 2

**评审认为值得读的论文数（平均，满分 5）**：旧推荐 2.95，画像推荐 5.00

| 画像 | 胜者 | 旧推荐值得读 | 画像推荐值得读 | 两份重合篇数 |
|---|---|---|---|---|
| rag_engineer | profile | 1.5/5 | 5.0/5 | 1 |
| agent_builder | tie | 5.0/5 | 5.0/5 | 2 |
| peft_student | profile | 2.5/5 | 5.0/5 | 1 |
| diffusion_researcher | profile | 3.5/5 | 5.0/5 | 2 |
| vlm_engineer | tie | 4.0/5 | 5.0/5 | 0 |
| alignment_researcher | profile | 3.0/5 | 5.0/5 | 2 |
| gnn_student | profile | 3.5/5 | 5.0/5 | 2 |
| speech_engineer | profile | 0.0/5 | 5.0/5 | 0 |
| eval_researcher | profile | 3.5/5 | 5.0/5 | 1 |
| inference_engineer | profile | 3.0/5 | 5.0/5 | 0 |

旧推荐拿到的兴趣文本与画像推荐相同（目标、主题、钉选论文标题），差别只在召回和排序方法。
评审模型与画像推荐内部的逐篇判断是同一个模型（gpt-4o-mini），可能偏向画像推荐的选择；10 个画像是小样本。
