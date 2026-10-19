# 论文向量模型对比

```
date                2026-10-19
git_commit          1eb6efe
worktree_dirty      True
source_sha256       2fc147ac623d20544c20ddfcb9b56dac8fd4e0dc22b621882751196c412bb522
index_papers        3000
questions           150 papers x (head, tail)
questions_file      datasets/long_abstract_questions.jsonl (gpt-4o-mini paraphrases, cached)
papers_embedded_as  title + abstract, overlapping pieces averaged (embed_papers)
search              vector-only cosine
hardware            local CPU (Apple Silicon), not the server
```

## 检索效果

| 模型 | tail Recall@10 | tail MRR | head Recall@10 | head MRR | MRR 比现用模型（全部问题，95% 区间） |
|---|---|---|---|---|---|
| MiniLM-L6 (current) | 0.820 | 0.660 | 0.840 | 0.690 | — |
| bge-small-en-v1.5 | 0.793 | 0.653 | 0.893 | 0.786 | +0.044 [+0.009, +0.081] |
| bge-small-en-v1.5 + query instruction | 0.800 | 0.679 | 0.893 | 0.800 | +0.064 [+0.027, +0.101] |
| e5-small-v2 | 0.820 | 0.698 | 0.880 | 0.732 | +0.040 [-0.001, +0.083] |

最后一列：同一道题上两个模型倒数名次之差的均值，配对 bootstrap 2000 次；区间不含 0 才算可信的差别。
tail：问题改写自摘要中超过 256 token 的部分；head：改写自摘要第一句。
问题是从“摘要超过 256 token”的论文里选的，长摘要的比例高于整个语料库。

## 代价

| 模型 | 最多读入 token | 维度 | 切出的段数 | 权重 MB | 每个查询 ms | 建索引 s |
|---|---|---|---|---|---|---|
| MiniLM-L6 (current) | 256 | 384 | 5095 | 91 | 10.3 | 49 |
| bge-small-en-v1.5 | 512 | 384 | 3008 | 133 | 20.7 | 305 |
| bge-small-en-v1.5 + query instruction | 512 | 384 | 3008 | 133 | 16.5 | 123 |
| e5-small-v2 | 512 | 384 | 3008 | 133 | 20.4 | 120 |

查询耗时是本机 CPU 上单条问题编码的中位数；服务器是 2 核 2GB 的云主机，绝对值会更慢，相对快慢可以参考。
