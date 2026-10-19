# 语义缓存阈值压力测试

```
date            2026-09-26
git_commit      0d66ecb
worktree_dirty  True
source_sha256   88b2f6bec5ac746fd6943c3909b71dad36e1dbfd20ccbea4541fc519b0edd7d5
dataset         cache_pairs.jsonl (12 hand-labelled pairs)
embedding_model sentence-transformers/all-MiniLM-L6-v2
scope           in-memory Redis simulation; no production hit-rate or dollar savings
```

| threshold | equivalent hit | wrong reuse | mean lookup ms |
|---|---|---|---|
| 0.980 | 0.167 | 0.167 | 22.2 |
| 0.997 | 0.167 | 0.000 | 12.5 |