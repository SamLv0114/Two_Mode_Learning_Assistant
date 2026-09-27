"""Compare the isolated supervisor experiment against the existing router.

Requires the same 50–100 human-verified generation golden set. It never changes
the production router and refuses to report unverified candidate scores.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import install_usage_patch, load_jsonl, markdown_table, mean, measure, result_header, write_result  # noqa: E402
from evaluation.runners.run_generation_eval import JUDGE_MODEL, judge_answer, load_verified_golden  # noqa: E402
from src.agents.plan_and_solve_agent import PlanAndSolveAgent  # noqa: E402
from src.agents.router import AgentRouter  # noqa: E402
from src.agents.supervisor_experiment import SupervisorExperiment  # noqa: E402
from src.database.models import SessionLocal  # noqa: E402
from src.models.embeddings import EmbeddingManager  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402
from src.utils.config import settings  # noqa: E402


def main():
    rows = load_verified_golden()
    install_usage_patch()
    from openai import OpenAI
    judge = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30, max_retries=0)
    rubric = (Path(__file__).resolve().parents[1] / "rubrics/generation_v1.md").read_text()
    embedding = EmbeddingManager()
    router = AgentRouter(embedding_model=embedding.model)
    retriever = Retriever(embedding)
    secondary = PlanAndSolveAgent()
    results = []
    for row in rows:
        arms = {}
        for arm in ("router", "supervisor"):
            with SessionLocal() as db:
                context = {"db": db, "user": SimpleNamespace(id=-1),
                           "retriever": retriever, "embedding_manager": embedding, "_trace": []}
                started = time.perf_counter()
                with measure() as usage:
                    if arm == "router":
                        answer, *_ = router.route(row["question"], [], context)
                    else:
                        primary = router._get_research_agent()
                        answer = SupervisorExperiment(primary, secondary).run(row["question"], [], context)
                judgment = judge_answer(judge, row["question"], answer.reply, answer.citations, rubric)
                arms[arm] = {"faithfulness": judgment["faithfulness"],
                             "answer_relevancy": judgment["answer_relevancy"],
                             "unsupported_claims": len(judgment["unsupported_claims"]),
                             "tool_calls": len(answer.tools_called),
                             "latency_ms": int((time.perf_counter() - started) * 1000),
                             "usage": usage}
        results.append({"id": row["id"], **arms})
    report = result_header("Supervisor 对照实验", {
        "dataset": f"generation_golden.jsonl ({len(rows)} human-verified cases)",
        "generator_model": settings.LLM_MODEL, "judge_model": JUDGE_MODEL,
        "status": "offline experiment; production router unchanged",
    })
    report += "\n\n" + markdown_table(
        ["arm", "faithfulness", "relevancy", "unsupported claims", "tools", "latency ms"],
        [[arm] + [f"{mean([r[arm][key] for r in results]):.3f}" for key in
                  ("faithfulness", "answer_relevancy", "unsupported_claims", "tool_calls", "latency_ms")]
         for arm in ("router", "supervisor")],
    )
    write_result(f"{date.today()}_supervisor_eval.md", report)
    print(report)
    out = Path(__file__).resolve().parents[1] / "results" / f"{date.today()}_supervisor_eval.jsonl"
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
