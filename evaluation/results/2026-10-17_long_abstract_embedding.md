# 长摘要分段平均嵌入的检索效果

```
date                 2026-10-17
git_commit           2687e26
worktree_dirty       True
source_sha256        38cd3e3d4fac52c1968e5566e599c213848e61709ffbccfbfa8630c7f78e39a5
embedding_model      sentence-transformers/all-MiniLM-L6-v2
index_papers         3000
questions            150
long_abstract_share  0.69 of indexed papers exceed 256 tokens
questions_file       datasets/long_abstract_questions.jsonl (gpt-4o-mini paraphrases)
search               vector-only cosine
```

| 向量方式 | 问题来自 | n | Recall@1 | Recall@10 | MRR |
|---|---|---|---|---|---|
| truncated | tail | 150 | 0.380 | 0.647 | 0.462 |
| pooled | tail | 150 | 0.560 | 0.827 | 0.661 |
| truncated | head | 150 | 0.627 | 0.880 | 0.717 |
| pooled | head | 150 | 0.620 | 0.847 | 0.690 |

tail：问题改写自摘要中超过 256 token 的部分（截断时完全看不到）；head：改写自摘要第一句（两种方式都看得到）。
