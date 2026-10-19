# 长期记忆压缩率评估（n>1，多风格对话）

```
date            2026-09-20
git_commit      0d66ecb
llm_model       gpt-4o-mini
n_conversations 30
profiles        refining / diverse / mixed_qa, each at short(6-8) and long(18-22) turns, 5 topic-seed variants
```

## 总体分布（30段对话，不是单一样本）

- 均值 **76.0%**
- 最小 43.5%，最大 95.1%
- P25 68.6%，中位数 77.7%，P75 90.1%

## 按对话风格分组（这才是方差的真实来源）

| profile | n | 均值 | 最小 | 最大 |
|---|---|---|---|---|
| refining | 10 | 79.5% | 63.9% | 92.2% |
| diverse | 10 | 61.6% | 43.5% | 77.8% |
| mixed_qa | 10 | 87.0% | 75.1% | 95.1% |

## 逐条结果

| id | profile | length | turns | raw_tokens | ctx_tokens | facts | reduction |
|---|---|---|---|---|---|---|---|
| mc001 | refining | short | 7 | 277 | 87 | 6 | 68.6% |
| mc002 | refining | short | 6 | 235 | 75 | 5 | 68.1% |
| mc003 | refining | short | 7 | 255 | 80 | 6 | 68.6% |
| mc004 | refining | short | 8 | 321 | 86 | 6 | 73.2% |
| mc005 | refining | short | 6 | 219 | 79 | 6 | 63.9% |
| mc006 | refining | long | 18 | 779 | 87 | 6 | 88.8% |
| mc007 | refining | long | 22 | 957 | 75 | 5 | 92.2% |
| mc008 | refining | long | 18 | 731 | 78 | 6 | 89.3% |
| mc009 | refining | long | 20 | 865 | 86 | 6 | 90.1% |
| mc010 | refining | long | 22 | 897 | 71 | 5 | 92.1% |
| mc011 | diverse | short | 6 | 221 | 109 | 9 | 50.7% |
| mc012 | diverse | short | 8 | 290 | 141 | 12 | 51.4% |
| mc013 | diverse | short | 6 | 207 | 112 | 10 | 45.9% |
| mc014 | diverse | short | 6 | 219 | 117 | 10 | 46.6% |
| mc015 | diverse | short | 6 | 209 | 118 | 10 | 43.5% |
| mc016 | diverse | long | 21 | 775 | 173 | 15 | 77.7% |
| mc017 | diverse | long | 21 | 775 | 172 | 15 | 77.8% |
| mc018 | diverse | long | 18 | 632 | 167 | 15 | 73.6% |
| mc019 | diverse | long | 19 | 693 | 168 | 15 | 75.8% |
| mc020 | diverse | long | 18 | 634 | 172 | 15 | 72.9% |
| mc021 | mixed_qa | short | 8 | 351 | 56 | 5 | 84.0% |
| mc022 | mixed_qa | short | 6 | 269 | 67 | 5 | 75.1% |
| mc023 | mixed_qa | short | 8 | 332 | 72 | 7 | 78.3% |
| mc024 | mixed_qa | short | 8 | 347 | 57 | 5 | 83.6% |
| mc025 | mixed_qa | short | 6 | 249 | 37 | 3 | 85.1% |
| mc026 | mixed_qa | long | 19 | 829 | 41 | 3 | 95.1% |
| mc027 | mixed_qa | long | 19 | 834 | 58 | 4 | 93.0% |
| mc028 | mixed_qa | long | 18 | 747 | 72 | 7 | 90.4% |
| mc029 | mixed_qa | long | 20 | 863 | 70 | 6 | 91.9% |
| mc030 | mixed_qa | long | 19 | 777 | 49 | 4 | 93.7% |