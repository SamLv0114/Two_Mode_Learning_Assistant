# 检索评估 baseline（四方案对照）

```
date            2026-09-19
git_commit      237fd51
corpus_snapshot prod 2026-09-19
corpus_size     29386 vectors (type=paper)
embedding_model sentence-transformers/all-MiniLM-L6-v2
reranker        cross-encoder/ms-marco-MiniLM-L-6-v2
dataset         retrieval.jsonl (120 queries)
top_k           10
```

## 总体

| 配置 | Recall@1 | Recall@5 | Recall@10 | MRR | 延迟P50(ms) | 延迟P95(ms) |
|---|---|---|---|---|---|---|
| vector_only | 0.275 | 0.443 | 0.515 | 0.353 | 893 | 1025 |
| bm25_only | 0.371 | 0.520 | 0.578 | 0.460 | 45 | 165 |
| hybrid_rrf | 0.287 | 0.602 | 0.653 | 0.445 | 503 | 947 |
| hybrid_rrf_rerank | 0.473 | 0.687 | 0.727 | 0.586 | 1918 | 2427 |

## 分档：semantic

| 配置 | Recall@1 | Recall@5 | Recall@10 | MRR | query数 |
|---|---|---|---|---|---|
| vector_only | 0.360 | 0.620 | 0.740 | 0.486 | 50 |
| bm25_only | 0.180 | 0.360 | 0.460 | 0.264 | 50 |
| hybrid_rrf | 0.280 | 0.640 | 0.700 | 0.433 | 50 |
| hybrid_rrf_rerank | 0.420 | 0.740 | 0.800 | 0.533 | 50 |

## 分档：exact

| 配置 | Recall@1 | Recall@5 | Recall@10 | MRR | query数 |
|---|---|---|---|---|---|
| vector_only | 0.375 | 0.525 | 0.550 | 0.424 | 40 |
| bm25_only | 0.850 | 0.925 | 0.925 | 0.880 | 40 |
| hybrid_rrf | 0.475 | 0.900 | 0.900 | 0.672 | 40 |
| hybrid_rrf_rerank | 0.825 | 0.925 | 0.925 | 0.871 | 40 |

## 分档：topic

| 配置 | Recall@1 | Recall@5 | Recall@10 | MRR | query数 |
|---|---|---|---|---|---|
| vector_only | 0.000 | 0.038 | 0.093 | 0.038 | 30 |
| bm25_only | 0.049 | 0.248 | 0.313 | 0.225 | 30 |
| hybrid_rrf | 0.050 | 0.142 | 0.246 | 0.162 | 30 |
| hybrid_rrf_rerank | 0.091 | 0.281 | 0.340 | 0.295 | 30 |