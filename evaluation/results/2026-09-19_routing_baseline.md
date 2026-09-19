# 路由评估 baseline

```
date            2026-09-19
git_commit      237fd51
llm_model       gpt-4o-mini
embedding_model sentence-transformers/all-MiniLM-L6-v2
dataset         routing.jsonl (100 queries)
live_thresholds keyword=0.6, embedding=0.65
```

## 1. 线上阈值下的表现 (keyword=0.6, embedding=0.65)

- 意图准确率 **94.0%** (94/100)
- 对抗样本准确率 **50.0%**
- 三阶段解决占比：关键词 66（66.0%）、
  向量 5（5.0%）、
  LLM 兜底 29（29.0%）
- **免 LLM 调用完成路由的比例：71.0%**

## 2. 判错的 query

| id | query | gold | 预测 | 命中阶段 |
|---|---|---|---|---|
| rt058 | what are people working on in multimodal learning right now | recommendation | research_qa | keyword |
| rt070 | did I already upload that PDF about diffusion models | document_management | research_qa | embedding |
| rt093 | how does a feed-forward network work | research_qa | recommendation | keyword |
| rt096 | explain what a knowledge base is in RAG systems | research_qa | document_management | keyword |
| rt098 | what is trending topic detection in time series analysis | research_qa | recommendation | keyword |
| rt099 | explain how citation networks are used to find papers that i | research_qa | recommendation | keyword |

## 3. research_qa 二次路由准确率

39/52 = **75.0%**

混淆情况（gold -> 实际）：

| gold_route | 实际路由 | 条数 |
|---|---|---|
| analytical | analytical | 12 |
| complex | analytical | 2 |
| complex | complex | 3 |
| complex | simple | 3 |
| simple | analytical | 1 |
| simple | complex | 7 |
| simple | simple | 24 |

## 4. 双阈值扫描

| kw阈值 | emb阈值 | 意图准确率 | LLM调用率 | 对抗准确率 | 关键词解决 | 向量解决 | LLM解决 |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.55 | 0.940 | 0.270 | 0.500 | 66 | 7 | 27 |
| 0.5 | 0.6 | 0.940 | 0.280 | 0.500 | 66 | 6 | 28 |
| 0.5 | 0.65 | 0.940 | 0.290 | 0.500 | 66 | 5 | 29 |
| 0.5 | 0.7 | 0.940 | 0.300 | 0.500 | 66 | 4 | 30 |
| 0.5 | 0.75 | 0.950 | 0.310 | 0.500 | 66 | 3 | 31 |
| 0.55 | 0.55 | 0.940 | 0.270 | 0.500 | 66 | 7 | 27 |
| 0.55 | 0.6 | 0.940 | 0.280 | 0.500 | 66 | 6 | 28 |
| 0.55 | 0.65 | 0.940 | 0.290 | 0.500 | 66 | 5 | 29 |
| 0.55 | 0.7 | 0.940 | 0.300 | 0.500 | 66 | 4 | 30 |
| 0.55 | 0.75 | 0.950 | 0.310 | 0.500 | 66 | 3 | 31 |
| 0.6 | 0.55 | 0.940 | 0.270 | 0.500 | 66 | 7 | 27 |
| 0.6 | 0.6 | 0.940 | 0.280 | 0.500 | 66 | 6 | 28 |
| 0.6 | 0.65 | 0.940 | 0.290 | 0.500 | 66 | 5 | 29 |
| 0.6 | 0.7 | 0.940 | 0.300 | 0.500 | 66 | 4 | 30 |
| 0.6 | 0.75 | 0.950 | 0.310 | 0.500 | 66 | 3 | 31 |
| 0.65 | 0.55 | 0.940 | 0.270 | 0.500 | 66 | 7 | 27 |
| 0.65 | 0.6 | 0.940 | 0.280 | 0.500 | 66 | 6 | 28 |
| 0.65 | 0.65 | 0.940 | 0.290 | 0.500 | 66 | 5 | 29 |
| 0.65 | 0.7 | 0.940 | 0.300 | 0.500 | 66 | 4 | 30 |
| 0.65 | 0.75 | 0.950 | 0.310 | 0.500 | 66 | 3 | 31 |
| 0.7 | 0.55 | 0.960 | 0.700 | 0.875 | 1 | 29 | 70 |
| 0.7 | 0.6 | 0.970 | 0.720 | 1.000 | 1 | 27 | 72 |
| 0.7 | 0.65 | 0.970 | 0.770 | 1.000 | 1 | 22 | 77 |
| 0.7 | 0.7 | 0.970 | 0.790 | 1.000 | 1 | 20 | 79 |
| 0.7 | 0.75 | 0.980 | 0.810 | 1.000 | 1 | 18 | 81 |
| 0.75 | 0.55 | 0.960 | 0.700 | 0.875 | 1 | 29 | 70 |
| 0.75 | 0.6 | 0.970 | 0.720 | 1.000 | 1 | 27 | 72 |
| 0.75 | 0.65 | 0.970 | 0.770 | 1.000 | 1 | 22 | 77 |
| 0.75 | 0.7 | 0.970 | 0.790 | 1.000 | 1 | 20 | 79 |
| 0.75 | 0.75 | 0.980 | 0.810 | 1.000 | 1 | 18 | 81 |

准确率最高的组合：keyword=0.7, embedding=0.75，
准确率 0.980，LLM 调用率 0.810。
