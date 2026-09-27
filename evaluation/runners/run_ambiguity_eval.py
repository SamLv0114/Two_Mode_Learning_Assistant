"""
Ambiguity-handling baseline.

Every query in ambiguity.jsonl is missing a key slot (which paper, which
author, which of two things). The question this measures is not answer
quality, it is behaviour: does the agent ask, or does it quietly pick
something and answer as if the ambiguity were not there?

Behaviour labels are assigned by gpt-4o-mini and every raw reply is written
into the result file, so the labels are auditable rather than taken on trust.
This is the "before" measurement for the clarify / interrupt-resume feature.

Token, latency and tool-call counts for the same run are reported here too —
they double as the cost baseline for the context-compression work.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (  # noqa: E402
    estimated_cost_usd, install_usage_patch, load_jsonl, markdown_table, mean, measure, pct,
    result_header, write_result,
)

from src.agents.router import AgentRouter  # noqa: E402
from src.models.embeddings import EmbeddingManager  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402
from src.utils.config import settings  # noqa: E402
import openai  # noqa: E402

LABELS = ["clarify", "guess", "generic", "refuse"]

CLASSIFY_PROMPT = """You are labelling how an AI assistant handled an underspecified question.

The user's question was missing key information: {missing}

User question: {query}

Assistant reply:
---
{reply}
---

Choose exactly one label:
- clarify: the assistant asks the user which one they mean, or offers candidates to pick from
- guess: the assistant silently picks a specific object and answers about it
- generic: the assistant answers in general terms without committing to any specific object
- refuse: the assistant says it cannot answer without more information, but does not ask a follow-up question

Reply with only the label word."""


def main() -> None:
    install_usage_patch()
    rows = load_jsonl("ambiguity.jsonl")
    em = EmbeddingManager()
    retriever = Retriever(embedding_manager=em)
    router = AgentRouter(embedding_model=em.model)
    judge = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    fake_user = SimpleNamespace(id=1)

    records = []
    for i, row in enumerate(rows, start=1):
        trace = []
        context = {"db": None, "user": fake_user, "retriever": retriever,
                   "embedding_manager": em, "_trace": trace}
        with measure() as m:
            try:
                result, intent, method, conf, agent_name = router.route(row["query"], [], context)
                reply = result.reply
                tools = result.tools_called
            except Exception as e:
                reply, tools, agent_name = f"[error] {e}", [], "error"

        seen, dup = set(), 0
        for rec in trace:
            key = (rec.get("tool"), str(rec.get("args")))
            if key in seen:
                dup += 1
            seen.add(key)

        verdict = judge.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(
                missing=row["missing_slot"], query=row["query"], reply=reply[:1500])}],
            max_completion_tokens=8, temperature=0.0,
        ).choices[0].message.content.strip().lower()
        label = next((l for l in LABELS if l in verdict), "generic")

        records.append({
            "id": row["id"], "query": row["query"], "agent": agent_name,
            "label": label, "reply": reply,
            "tokens": m["total_tokens"], "llm_calls": m["llm_calls"],
            "estimated_cost_usd": m["estimated_cost_usd"],
            "by_stage": m["by_stage"],
            "latency_ms": m["latency_ms"], "tool_calls": len(tools), "dup_tools": dup,
        })
        print(f"[{i:2d}/{len(rows)}] {label:<8} {agent_name:<20} "
              f"{m['total_tokens']:>6}tok {m['latency_ms']:>6}ms  {row['query'][:45]}")

    counts = {l: sum(1 for r in records if r["label"] == l) for l in LABELS}
    n = len(records)
    stage_totals = {}
    for row in records:
        for stage, models in row["by_stage"].items():
            slot = stage_totals.setdefault(stage, {})
            for model, usage in models.items():
                bucket = slot.setdefault(model, {key: 0 for key in
                    ("prompt_tokens", "cached_tokens", "completion_tokens", "calls")})
                for key in bucket:
                    bucket[key] += usage[key]

    parts = [result_header("歧义处理基线 + Token/延迟基线", {
        "llm_model": settings.LLM_MODEL,
        "dataset": f"ambiguity.jsonl ({n} underspecified queries)",
        "corpus_snapshot": "prod 2026-09-19",
        "labelling": "gpt-4o-mini, 原始回答全文见附录",
        "pricing": "evaluation/pricing.json (2026-09-26 snapshot; estimated, not invoice)",
    })]

    parts.append(f"""
## 1. 行为分布

| 行为 | 条数 | 占比 |
|---|---|---|
| clarify（反问澄清） | {counts['clarify']} | {counts['clarify']/n:.0%} |
| guess（自行选定对象作答） | {counts['guess']} | {counts['guess']/n:.0%} |
| generic（泛泛而谈不指定对象） | {counts['generic']} | {counts['generic']/n:.0%} |
| refuse（拒答但不追问） | {counts['refuse']} | {counts['refuse']/n:.0%} |

**clarify_rate = {counts['clarify']/n:.0%}** ← clarify / 中断恢复功能的改动前基线
""")

    parts.append("\n## 2. Token 与延迟基线\n")
    parts.append(markdown_table(
        ["指标", "P50", "P95", "均值"],
        [
            ["total_tokens", f"{pct([r['tokens'] for r in records], 0.5):.0f}",
             f"{pct([r['tokens'] for r in records], 0.95):.0f}",
             f"{mean([r['tokens'] for r in records]):.0f}"],
            ["llm_calls", f"{pct([r['llm_calls'] for r in records], 0.5):.0f}",
             f"{pct([r['llm_calls'] for r in records], 0.95):.0f}",
             f"{mean([r['llm_calls'] for r in records]):.1f}"],
            ["latency_ms", f"{pct([r['latency_ms'] for r in records], 0.5):.0f}",
             f"{pct([r['latency_ms'] for r in records], 0.95):.0f}",
             f"{mean([r['latency_ms'] for r in records]):.0f}"],
            ["tool_calls", f"{pct([r['tool_calls'] for r in records], 0.5):.0f}",
             f"{pct([r['tool_calls'] for r in records], 0.95):.0f}",
             f"{mean([r['tool_calls'] for r in records]):.1f}"],
            ["estimated_cost_usd", "—", "—",
             (f"{mean([r['estimated_cost_usd'] for r in records]):.6f}"
              if all(r['estimated_cost_usd'] is not None for r in records) else "unpriced model")],
        ],
    ))
    total_tools = sum(r["tool_calls"] for r in records)
    total_dup = sum(r["dup_tools"] for r in records)
    parts.append(f"\n工具调用总数 {total_tools}，其中参数完全相同的重复调用 {total_dup} 次，"
                 f"**重复调用率 {total_dup / total_tools if total_tools else 0:.1%}**\n")
    parts.append("\n分阶段 LLM 调用与估算费用（不含行为标签 judge）：\n")
    parts.append(markdown_table(["stage", "calls", "input tokens", "output tokens", "estimated USD"], [
        [stage,
         sum(bucket["calls"] for bucket in models.values()),
         sum(bucket["prompt_tokens"] for bucket in models.values()),
         sum(bucket["completion_tokens"] for bucket in models.values()),
         (f"{estimated_cost_usd({'by_model': models}):.6f}"
          if estimated_cost_usd({"by_model": models}) is not None else "unpriced")]
        for stage, models in sorted(stage_totals.items())
    ]))

    parts.append("\n## 3. 逐条结果\n")
    parts.append(markdown_table(
        ["id", "query", "路由到", "行为", "tokens", "延迟ms"],
        [[r["id"], r["query"][:42], r["agent"], r["label"], r["tokens"], r["latency_ms"]]
         for r in records],
    ))

    parts.append("\n## 附录：原始回答（供人工核验标注是否正确）\n")
    for r in records:
        parts.append(f"\n**{r['id']}** `{r['query']}` → *{r['label']}*\n\n> "
                     + r["reply"][:700].replace("\n", "\n> ") + "\n")

    write_result("2026-09-19_ambiguity_baseline.md", "\n".join(parts))
    print(f"\nclarify_rate = {counts['clarify']/n:.0%}  ({counts})")


if __name__ == "__main__":
    main()
