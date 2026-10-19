"""Manual, paid Coding Agent baseline on curator supplied cases.

Set CODE_LAB_EVAL_CASES to a JSONL file of
{"kind":"paper","source":"2608.14407","objective":"..."} or
{"kind":"repo","source":"https://github.com/owner/repo"}.
Never part of the default or deterministic CI suites.
"""
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    cases_name = os.getenv("CODE_LAB_EVAL_CASES")
    if not cases_name:
        sys.exit("Set CODE_LAB_EVAL_CASES to a curated JSONL file; no model calls were made")
    cases_path = Path(cases_name).expanduser().resolve()
    if not cases_path.is_file():
        sys.exit(f"Case file does not exist: {cases_path}")
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    if not cases or any(item.get("kind") not in ("paper", "repo") or not item.get("source") for item in cases):
        sys.exit("Each case needs kind=paper|repo and a source")

    from langgraph.checkpoint.memory import InMemorySaver
    from src.agents.coding_practice import build_paper_graph, build_repo_graph
    from src.agents.coding_services import CodingServices, WorkerClient
    from src.api.deps import get_embedding_manager
    from src.database.models import SessionLocal
    from src.utils.config import settings

    if not settings.OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is required; no model calls were made")
    execution_enabled = bool(settings.PRACTICE_WORKER_URL
                             and settings.PRACTICE_WORKER_URL.startswith("https://")
                             and settings.PRACTICE_WORKER_TOKEN)
    if execution_enabled and any(item["kind"] == "paper" for item in cases):
        WorkerClient()  # Validate configured worker before any model request.

    results = []
    for number, case in enumerate(cases, start=1):
        db = SessionLocal()
        started = time.monotonic()
        kind = case["kind"]
        try:
            services = CodingServices(db, embedding_manager=get_embedding_manager() if kind == "paper" else None)
            graph = (build_paper_graph(services, InMemorySaver()) if kind == "paper"
                     else build_repo_graph(services, InMemorySaver()))
            config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 35}
            payload = ({"paper_id": case["source"], "objective": case.get("objective", ""), "attempts": 0,
                        "execution_enabled": execution_enabled}
                       if kind == "paper" else {"repo_url": case["source"]})
            graph.invoke(payload, config)
            state = graph.get_state(config).values or {}
            manifest = state.get("manifest") or {}
            faithfulness = state.get("faithfulness") or {}
            result = {"kind": kind, "source": case["source"], "stop_reason": state.get("stop_reason"),
                      "title": state.get("title") if kind == "paper" else None,
                      "evidence": state.get("evidence") if kind == "paper" else None,
                      "manifest": manifest if kind == "paper" else None,
                      "generated_code": state.get("code") if kind == "paper" else None,
                      "explanation": state.get("explanation") if kind == "paper" else None,
                      "questions": state.get("questions") if kind == "paper" else None,
                      "execution": state.get("execution") if kind == "paper" else None,
                      "timeline": state.get("timeline") if kind == "paper" else None,
                      "faithfulness_review": faithfulness if kind == "paper" else None,
                      "repo_url": state.get("repo_url") if kind == "repo" else None,
                      "commit_sha": state.get("commit_sha") if kind == "repo" else None,
                      "architecture": state.get("architecture") if kind == "repo" else None,
                      "entry_points": state.get("entry_points") if kind == "repo" else None,
                      "file_roles": state.get("file_roles") if kind == "repo" else None,
                      "limitations": state.get("limitations") if kind == "repo" else None,
                      "method": manifest.get("method") if kind == "paper" else None,
                      "preflight_feasible": manifest.get("feasible") if kind == "paper" else None,
                      "preflight_reason": manifest.get("reason") if kind == "paper" else None,
                      "unknown_detail_count": len(manifest.get("unknown_details", [])) if kind == "paper" else None,
                      "run_pass": bool(state.get("execution", {}).get("passed")) if kind == "paper" else None,
                      "faithfulness_model_review": faithfulness.get("status") if kind == "paper" else None,
                      "repair_rounds": max(0, len(state.get("timeline", [])) - 1) if kind == "paper" else None,
                      "paper_linked_repo": bool(state.get("linked_repo")) if kind == "paper" else None,
                      "repo_linked_paper": bool(state.get("linked_paper")) if kind == "repo" else None,
                      "paper_executions": state.get("attempts") if kind == "paper" else None,
                      "execution_enabled": execution_enabled if kind == "paper" else None,
                      "paper_questions": len(state.get("questions", [])) if kind == "paper" else None,
                      "repo_files_inspected": len(state.get("file_tree", [])) if kind == "repo" else None,
                      "ready_for_user": bool(kind == "repo" and state.get("architecture")),
                      "input_tokens": services.input_tokens, "output_tokens": services.output_tokens}
        except Exception as exc:
            result = {"kind": kind, "source": case["source"], "error": str(exc)[:300]}
        finally:
            db.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
        print(f"{number}/{len(cases)} {kind}: {result.get('stop_reason') or result.get('error') or 'ready'}", flush=True)

    output_name = os.getenv("CODING_AGENT_EVAL_OUTPUT", f"{date.today().isoformat()}_coding_agent_eval.jsonl")
    output = Path(output_name).expanduser()
    if not output.is_absolute():
        output = ROOT / "evaluation" / "results" / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results))
    paper_cases = [item for item in results if item["kind"] == "paper"]
    repo_cases = [item for item in results if item["kind"] == "repo"]
    summary = {"paper_cases": len(paper_cases),
               "paper_success": sum(item.get("stop_reason") == "success" for item in paper_cases),
               "paper_draft_generated": sum(item.get("stop_reason") == "generated_unexecuted" for item in paper_cases),
               "paper_preflight_refused": sum(item.get("preflight_feasible") is False for item in paper_cases),
               "paper_run_pass": sum(item.get("run_pass") is True for item in paper_cases),
               "paper_model_review_pass": sum(item.get("faithfulness_model_review") == "PASS" for item in paper_cases),
               "paper_with_linked_repo": sum(item.get("paper_linked_repo") is True for item in paper_cases),
               "repo_cases": len(repo_cases),
               "repo_ready_for_user": sum(bool(item.get("ready_for_user")) for item in repo_cases),
               "repo_with_linked_paper": sum(item.get("repo_linked_paper") is True for item in repo_cases),
               "note": "RUN:PASS is a synthetic self-check; faithfulness is same-model review, not human verification",
               "stop_reasons": dict(Counter(item.get("stop_reason") or item.get("error", "ready") for item in results)),
               "input_tokens": sum(item.get("input_tokens", 0) for item in results),
               "output_tokens": sum(item.get("output_tokens", 0) for item in results)}
    print(json.dumps(summary, indent=2))
    print(f"Case results: {output}")


if __name__ == "__main__":
    main()
