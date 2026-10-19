# 全文切块弱监督对照

```
date            2026-09-26
git_commit      0d66ecb
worktree_dirty  True
source_sha256   88b2f6bec5ac746fd6943c3909b71dad36e1dbfd20ccbea4541fc519b0edd7d5
dataset         same Chroma full-text snapshot, 8 units, 15 excerpt-derived queries
embedding_model SentenceTransformer
limitation      snapshot has one full-text paper, split into sections; excerpt-derived queries; no human labels or table comparison
```

| mode | Recall@5 | MRR@5 | target evidence@5 | records | text bytes | search ms |
|---|---|---|---|---|---|---|
| fixed | 0.933 | 0.783 | 0.733 | 87 | 85915 | 0.08 |
| semantic_child | 1.000 | 0.797 | 0.733 | 97 | 85823 | 0.01 |
| parent_child | 1.000 | 0.797 | 0.733 | 97 | 206166 | 0.01 |