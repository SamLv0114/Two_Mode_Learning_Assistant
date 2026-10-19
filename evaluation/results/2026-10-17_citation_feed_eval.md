# 推荐各阶段的引用离线评测

```
date            2026-10-17
git_commit      2687e26
worktree_dirty  True
source_sha256   068b10063559be6d49b2da5353bd0ab2ba485fbb0f0e2b888dca3012fffb0108
cases           30
gold_per_case   median 6 references from the 9 months before the test paper
profile         goal = test paper title, 2 of its older references pinned
judge_model     gpt-4o-mini (feed_judge)
judged_share    900/900
cases_file      datasets/citation_feed_cases.jsonl
```

| 系统 | Top10 命中数 | Precision@10 | nDCG@10 | 至少命中 1 篇 | 候选池覆盖 gold 比例 |
|---|---|---|---|---|---|
| content：画像检索语料库 + 预排序 | 0.07 | 0.007 | 0.010 | 2/30 | 0.01 |
| graph：Semantic Scholar 推荐原始顺序 | 0.03 | 0.003 | 0.016 | 1/30 | 0.04 |
| channels：两路配额合并，预排序 | 0.13 | 0.013 | 0.015 | 4/30 | 0.03 |
| judged：模型逐篇判断后的最终排序 | 0.13 | 0.013 | 0.021 | 4/30 | 0.03 |

gold 是测试论文在发表前 9 个月内引用的新论文（作者写作期间找到并用上的新工作）；钉选的是它更早的参考文献。引用是“有用”的代理指标，不等于读者的判断。
Semantic Scholar 的推荐模型基于引用图训练，图里可能已经包含测试论文本身的引用，对 graph 一路有利。
