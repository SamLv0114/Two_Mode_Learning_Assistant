"""Compare old last-8 truncation with rolling-summary prompt assembly.

The constructed code/URL retention task is a narrow memory probe, not a
general answer-quality benchmark. --dry-run checks token assembly offline.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import load_jsonl, markdown_table, mean, result_header, write_result  # noqa: E402
from src.agents.context_budget import ContextBudget  # noqa: E402
from src.utils.config import settings  # noqa: E402


SYSTEM = "Answer with the exact code and source URL the user gave earlier. Do not invent either item."
PADDING = "This intermediate turn discusses general retrieval and summarization without changing the original code or URL. " * 25


def case_history(row):
    history = [{"role": "user", "content": f"Remember my code {row['fact']} and source {row['source']}. " + PADDING},
               {"role": "assistant", "content": "I will remember those exact details."}]
    for i in range(14):
        history.extend([{"role": "user", "content": f"Side discussion {i}: " + PADDING},
                        {"role": "assistant", "content": "Here is a generic explanation. " + PADDING}])
    return history


def run_case(row, client, dry_run=False):
    budget = ContextBudget(settings.LLM_MODEL)
    history = case_history(row)
    question = "What exact code and source URL did I give at the start?"
    old = [{"role": "system", "content": SYSTEM}, *history[-8:], {"role": "user", "content": question}]
    if dry_run:
        # The fake response exercises summary assembly only. Quality scores
        # are deliberately omitted because a fake summary proves nothing.
        summarizer = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Dry-run summary"))])
        )))
    else:
        summarizer = client
    started = time.perf_counter()
    new = budget.build(SYSTEM, "", history[-8:], history, question, [],
                       {"user": SimpleNamespace(id=-1), "session_id": row["id"]},
                       summarizer, settings.LLM_MODEL)
    summary_ms = int((time.perf_counter() - started) * 1000)
    result = {"id": row["id"], "old_tokens": budget.count_messages(old),
              "new_tokens": budget.count_messages(new), "summary_ms": summary_ms}
    if not dry_run:
        for name, messages in (("old", old), ("new", new)):
            started = time.perf_counter()
            response = client.chat.completions.create(model=settings.LLM_MODEL, messages=messages,
                                                      temperature=0, max_tokens=120)
            answer = response.choices[0].message.content or ""
            result[name] = {"answer": answer,
                            "fact_correct": row["fact"] in answer,
                            "source_correct": row["source"] in answer,
                            "latency_ms": int((time.perf_counter() - started) * 1000)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl("context_long_sessions.jsonl")
    if args.dry_run:
        client = None
    else:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=20, max_retries=0)
    scored = [run_case(row, client, args.dry_run) for row in rows]
    source = Path(__file__).resolve().parents[1] / "datasets/context_long_sessions.jsonl"
    report = result_header("上下文预算与摘要对照", {
        "dataset": f"context_long_sessions.jsonl ({len(rows)} constructed cases)",
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": settings.LLM_MODEL,
        "mode": "offline token assembly only" if args.dry_run else "model answers",
    })
    metric_rows = [["old input tokens", f"{mean([r['old_tokens'] for r in scored]):.1f}"],
                   ["new input tokens", f"{mean([r['new_tokens'] for r in scored]):.1f}"],
                   ["summary latency ms", f"{mean([r['summary_ms'] for r in scored]):.1f}"]]
    if not args.dry_run:
        for arm in ("old", "new"):
            metric_rows.extend([
                [f"{arm} exact code", f"{mean([r[arm]['fact_correct'] for r in scored]):.3f}"],
                [f"{arm} source URL", f"{mean([r[arm]['source_correct'] for r in scored]):.3f}"],
                [f"{arm} answer latency ms", f"{mean([r[arm]['latency_ms'] for r in scored]):.1f}"],
            ])
    report += "\n\n" + markdown_table(["metric", "value"], metric_rows)
    report += "\n\nConstructed exact-code memory probe; general answer correctness needs the human-verified generation set.\n"
    write_result(f"{date.today()}_context_budget_{'dry' if args.dry_run else 'live'}.md", report)
    print(report)


if __name__ == "__main__":
    main()
