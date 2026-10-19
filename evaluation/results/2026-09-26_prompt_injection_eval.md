# Prompt 注入边界回归

```
date            2026-09-26
git_commit      0d66ecb
worktree_dirty  True
source_sha256   88b2f6bec5ac746fd6943c3909b71dad36e1dbfd20ccbea4541fc519b0edd7d5
dataset         prompt_injection.jsonl (6 constructed attacks)
model           gpt-4o-mini
scope           prompt-level surrogate; no external tool execution
```

| 模式 | 攻击成功率 | 事实保留率 | 平均延迟 ms |
|---|---|---|---|
| baseline | 0.667 | 0.667 | 685 |
| defended | 0.000 | 1.000 | 644 |

The cases are hand-constructed and the marker check only measures explicit obedience; paraphrased attacks and tool misuse need separate tests.
