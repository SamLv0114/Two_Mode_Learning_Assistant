# 离线生成质量评测

```
date                       2026-09-26
git_commit                 0d66ecb
worktree_dirty             True
source_sha256              aabb1b6cf9b319823f5dc09340c1d5e4cee6a8320c80e967c184cec5b0da52a3
dataset                    generation_golden.jsonl (50 human-verified cases)
dataset_sha256             bfd0f2d60a2e54de18d98fd3c04bfa15dd57c7d8c1e33604d122d7ae182b3961
predictions_sha256         05e39afd11e151fc2614c22581cc41bbb0264f82a580b38654f0ba63e4fde9d7
corpus_path                /private/tmp/learning_assistant_eval_chroma
generator_model            gpt-4o-mini
judge_model                gpt-4o
rubric_sha256              fb94911f1a296cba3e43d043d55687a60c171594a2506591257c6a343f182b1d
correctness_rubric_sha256  9f2c58d9e6d0cfe4eed1e9394f73a7e068025091f94b677e872aae1ec90efa79
same_model_bias            False
gold_draft_model           gpt-4o-mini (human-confirmed; see generation_golden.jsonl)
```

| 指标 | 均值 | 样本数 |
|---|---|---|
| faithfulness | 0.736 | 50 |
| answer_relevancy | 0.954 | 50 |
| answer_correctness | 0.500 | 47 |
| unsupported_claims | 2.96 | 50 |
| context_precision | 0.209 | 50 |

Answer correctness excludes 3 evidence-insufficient cases; faithfulness and relevancy include them. The reference answers began as gpt-4o-mini drafts and were confirmed by the project owner, so shared-model drafting bias remains possible.


| 诊断指标 | 结果 |
|---|---|
| gold source present in returned citations | 22/50 |
| correctness when gold source present | 0.705 (n=22) |
| correctness when gold source absent | 0.320 (n=25) |
| generator LLM calls | 370 |
| generator estimated cost USD (judge excluded) | 0.266508 |

The questions do not identify their target papers; alternate valid answers may therefore score below the paper-specific reference. Treat these as exploratory results, not production answer accuracy. Source presence uses exact citation IDs. Costs are SDK usage estimates from evaluation/pricing.json, not billed amounts or cache savings.


Raw answers and per-case judgments are stored in the accompanying JSONL.


Judge–human calibration: unavailable; no human-scored subset supplied.
