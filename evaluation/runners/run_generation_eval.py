"""Offline generation evaluation over human-verified research Q&A.

With no arguments, collect fresh answers and judge them. `--predictions` judges
previously collected answers without rerunning the agents. The runner refuses
to publish scores from pending or fewer than 50 human-verified examples.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (  # noqa: E402
    RESULTS_DIR, install_usage_patch, load_jsonl, markdown_table, mean,
    measure, result_header, write_result,
)

RUBRIC_PATH = Path(__file__).resolve().parents[1] / "rubrics" / "generation_v1.md"
CORRECTNESS_RUBRIC_PATH = Path(__file__).resolve().parents[1] / "rubrics" / "generation_correctness_v1.md"
GOLDEN_PATH = Path(__file__).resolve().parents[1] / "datasets" / "generation_golden.jsonl"
PREDICTIONS_PATH = RESULTS_DIR / "generation_predictions.jsonl"
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "datasets" / "judge_calibration.jsonl"
JUDGMENTS_PATH = RESULTS_DIR / "generation_judgments.partial.jsonl"
JUDGMENTS_META_PATH = RESULTS_DIR / "generation_judgments.meta.json"
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")


class PacedCompletions:
    """Bound judge request rate and retry transient token-per-minute limits."""

    def __init__(self, completions, min_interval_seconds=5.0):
        self.completions = completions
        self.min_interval_seconds = min_interval_seconds
        self.last_call_at = None

    def create(self, **kwargs):
        from openai import RateLimitError

        for attempt in range(6):
            if self.last_call_at is not None:
                wait = self.min_interval_seconds - (time.monotonic() - self.last_call_at)
                if wait > 0:
                    time.sleep(wait)
            self.last_call_at = time.monotonic()
            try:
                return self.completions.create(**kwargs)
            except RateLimitError:
                if attempt == 5:
                    raise
                time.sleep(10 * (attempt + 1))


def judgment_signature(prediction_path, rubric, correctness_rubric):
    digest = hashlib.sha256()
    for payload in (GOLDEN_PATH.read_bytes(), prediction_path.read_bytes(),
                    rubric.encode(), correctness_rubric.encode(), JUDGE_MODEL.encode()):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_judgment_checkpoint(signature, rows):
    if not JUDGMENTS_PATH.exists() or not JUDGMENTS_META_PATH.exists():
        return []
    meta = json.loads(JUDGMENTS_META_PATH.read_text(encoding="utf-8"))
    if meta.get("signature") != signature:
        return []
    scored = [json.loads(line) for line in JUDGMENTS_PATH.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if [row["id"] for row in scored] != [row["id"] for row in rows[:len(scored)]]:
        raise ValueError("Judgment checkpoint case order does not match golden set")
    return scored


def validate_golden(rows):
    if not 50 <= len(rows) <= 100:
        raise ValueError(f"Expected 50–100 human-verified cases, got {len(rows)}")
    ids = set()
    for row in rows:
        case_id = row.get("id")
        if not case_id or case_id in ids:
            raise ValueError(f"Missing or duplicate case id: {case_id}")
        ids.add(case_id)
        if row.get("review_status") != "human_verified" or not row.get("reviewer"):
            raise ValueError(f"{case_id}: human review is required before publishing metrics")
        if not row.get("question") or not row.get("reference_answer") or not row.get("sources"):
            raise ValueError(f"{case_id}: question, reference answer and sources are required")
        if not row.get("expected_behavior") or not row.get("annotation_notes"):
            raise ValueError(f"{case_id}: expected behavior and annotation notes are required")
        if not all(source.get("id") and source.get("text") for source in row["sources"]):
            raise ValueError(f"{case_id}: each source needs an id and evidence text")


def load_verified_golden():
    if not GOLDEN_PATH.exists():
        raise ValueError("Human-verified golden set is missing. Run evaluation/review_golden.py --reviewer YOUR_ID first.")
    rows = load_jsonl("generation_golden.jsonl")
    validate_golden(rows)
    return rows


def context_precision(returned_ids, relevant_ids):
    """Mean precision at returned relevant hits; not recall normalized AP.

    With one relevant source per case this reduces to its reciprocal rank.
    None means no gold labels.
    """
    if relevant_ids is None:
        return None
    relevant = set(relevant_ids)
    hits = 0
    precision_sum = 0.0
    for rank, source_id in enumerate(returned_ids, start=1):
        if source_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / hits if hits else 0.0


def calibration_stats(scored, labels):
    by_id = {row["id"]: row["judgment"] for row in scored}
    if not labels:
        return None
    pairs = [(by_id[row["id"]], row) for row in labels if row["id"] in by_id]
    if not pairs:
        return None
    return {metric: {
        "mae": mean([abs(float(judge[metric]) - float(human["human_" + metric])) for judge, human in pairs]),
        "pass_agreement": mean([(float(judge[metric]) >= 0.7) ==
                                (float(human["human_" + metric]) >= 0.7) for judge, human in pairs]),
        "n": len(pairs),
    } for metric in ("faithfulness", "answer_relevancy")}


def collect_answers(rows):
    from src.agents.router import AgentRouter
    from src.database.models import SessionLocal
    from src.models.embeddings import EmbeddingManager
    from src.rag.retriever import Retriever

    install_usage_patch()
    embedding = EmbeddingManager()
    retriever = Retriever(embedding_manager=embedding)
    router = AgentRouter(embedding_model=embedding.model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = []
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            db = SessionLocal()
            trace = []
            context = {
                "db": db, "user": SimpleNamespace(id=-1), "retriever": retriever,
                "embedding_manager": embedding, "_trace": trace,
            }
            try:
                with measure() as usage:
                    result, intent, method, confidence, agent = router.route(row["question"], [], context)
                prediction = {
                    "id": row["id"], "question": row["question"], "answer": result.reply,
                    "citations": result.citations, "tools": result.tools_called,
                    "trace": trace, "agent": agent, "intent": intent,
                    "usage": usage,
                }
                predictions.append(prediction)
                out.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{index}/{len(rows)}] generated {row['id']} ({agent})", flush=True)
            finally:
                db.close()
    return predictions


def judge_answer(client, question, answer, citations, rubric):
    evidence = [
        {"title": c.get("title"), "url": c.get("url"), "text": (c.get("evidence") or "")[:800]}
        for c in citations if c.get("evidence")
    ][:8]
    payload = {"question": question, "answer": answer, "retrieved_evidence": evidence}
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": rubric + "\nReturn only the specified JSON object."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    score = json.loads(response.choices[0].message.content)
    for key in ("faithfulness", "answer_relevancy"):
        score[key] = max(0.0, min(1.0, float(score[key])))
    if not isinstance(score.get("unsupported_claims"), list):
        raise ValueError("Judge omitted unsupported_claims list")
    return score


def judge_correctness(client, row, answer, rubric):
    """Compare an answer with reviewed gold, separate from source faithfulness."""
    payload = {
        "question": row["question"], "reference_answer": row["reference_answer"],
        "assistant_answer": answer,
        "gold_evidence": [{"id": source["id"], "text": source["text"][:1200]}
                          for source in row["sources"][:3]],
    }
    response = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": rubric + "\nReturn only the specified JSON object."},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    score = json.loads(response.choices[0].message.content)
    score["answer_correctness"] = max(0.0, min(1.0, float(score["answer_correctness"])))
    for key in ("missing_key_points", "contradictions"):
        if not isinstance(score.get(key), list):
            raise ValueError(f"Correctness judge omitted {key} list")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, help="Existing JSONL answers to judge")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH,
                        help="Human-scored subset with human_faithfulness and human_answer_relevancy")
    args = parser.parse_args()
    rows = load_verified_golden()

    if args.predictions:
        predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        predictions = collect_answers(rows)
    by_id = {row["id"]: row for row in predictions}
    if set(by_id) != {row["id"] for row in rows} or len(by_id) != len(predictions):
        raise ValueError("Prediction IDs must match the verified dataset exactly")

    from openai import OpenAI
    from src.utils.config import settings
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required to run the offline judge")
    raw_client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=30, max_retries=0)
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=PacedCompletions(raw_client.chat.completions)))
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    correctness_rubric = CORRECTNESS_RUBRIC_PATH.read_text(encoding="utf-8")
    prediction_path = args.predictions or PREDICTIONS_PATH
    signature = judgment_signature(prediction_path, rubric, correctness_rubric)
    scored = load_judgment_checkpoint(signature, rows)
    if scored:
        print(f"Resuming {len(scored)}/{len(rows)} completed judgments", flush=True)
    else:
        JUDGMENTS_META_PATH.write_text(json.dumps({"signature": signature,
                                                     "judge_model": JUDGE_MODEL}), encoding="utf-8")
        JUDGMENTS_PATH.write_text("", encoding="utf-8")
    with JUDGMENTS_PATH.open("a", encoding="utf-8") as checkpoint:
        for row in rows[len(scored):]:
            pred = by_id[row["id"]]
            judgment = judge_answer(client, row["question"], pred["answer"], pred.get("citations", []), rubric)
            correctness = (judge_correctness(client, row, pred["answer"], correctness_rubric)
                           if row.get("answerable_from_excerpt", True) else None)
            returned_ids = [c.get("url") or c.get("id") for c in pred.get("citations", [])]
            result = {
                "id": row["id"], "judgment": judgment, "correctness": correctness,
                "context_precision": context_precision(returned_ids, row.get("relevant_source_ids")),
            }
            scored.append(result)
            checkpoint.write(json.dumps(result, ensure_ascii=False) + "\n")
            checkpoint.flush()
            print(f"[{len(scored)}/{len(rows)}] judged {row['id']}", flush=True)
    faithfulness = [r["judgment"]["faithfulness"] for r in scored]
    relevancy = [r["judgment"]["answer_relevancy"] for r in scored]
    precision = [r["context_precision"] for r in scored if r["context_precision"] is not None]
    unsupported = [len(r["judgment"]["unsupported_claims"]) for r in scored]
    correctness_scores = [r["correctness"]["answer_correctness"] for r in scored
                          if r["correctness"] is not None]
    source_hits = [r for r in scored if r["context_precision"] is not None
                   and r["context_precision"] > 0]
    source_misses = [r for r in scored if r["context_precision"] == 0]
    hit_correctness = [r["correctness"]["answer_correctness"] for r in source_hits
                       if r["correctness"] is not None]
    miss_correctness = [r["correctness"]["answer_correctness"] for r in source_misses
                        if r["correctness"] is not None]
    generator_usage = [pred.get("usage", {}) for pred in predictions]
    generator_costs = [item.get("estimated_cost_usd") for item in generator_usage]
    report = result_header("离线生成质量评测", {
        "dataset": f"generation_golden.jsonl ({len(rows)} human-verified cases)",
        "dataset_sha256": hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "corpus_path": str(settings.VECTOR_DB_DIR),
        "generator_model": settings.LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
        "correctness_rubric_sha256": hashlib.sha256(correctness_rubric.encode()).hexdigest(),
        "same_model_bias": JUDGE_MODEL == settings.LLM_MODEL,
        "gold_draft_model": "gpt-4o-mini (human-confirmed; see generation_golden.jsonl)",
    })
    report += "\n\n" + markdown_table(
        ["指标", "均值", "样本数"],
        [
            ["faithfulness", f"{mean(faithfulness):.3f}", len(faithfulness)],
            ["answer_relevancy", f"{mean(relevancy):.3f}", len(relevancy)],
            ["answer_correctness", f"{mean(correctness_scores):.3f}", len(correctness_scores)],
            ["unsupported_claims", f"{mean(unsupported):.2f}", len(unsupported)],
            ["returned_context_precision", f"{mean(precision):.3f}" if precision else "unlabelled", len(precision)],
        ],
    )
    report += (f"\n\nAnswer correctness excludes {len(rows) - len(correctness_scores)} "
               "evidence-insufficient cases; faithfulness and relevancy include them. "
               "The reference answers began as gpt-4o-mini drafts and were confirmed "
               "by the project owner, so shared-model drafting bias remains possible.\n")
    report += "\n\n" + markdown_table(["诊断指标", "结果"], [
        ["gold source present in returned citations", f"{len(source_hits)}/{len(rows)}"],
        ["correctness when gold source present", f"{mean(hit_correctness):.3f} (n={len(hit_correctness)})"],
        ["correctness when gold source absent", f"{mean(miss_correctness):.3f} (n={len(miss_correctness)})"],
        ["generator LLM calls", sum(item.get("llm_calls", 0) for item in generator_usage)],
        ["generator estimated cost USD (judge excluded)",
         f"{sum(generator_costs):.6f}" if all(cost is not None for cost in generator_costs)
         else "unpriced model"],
    ])
    report += ("\n\nThe questions do not identify their target papers; alternate valid answers "
               "may therefore score below the paper-specific reference. Treat these as "
               "exploratory results, not production answer accuracy. Source presence uses "
               "exact citation IDs. Costs are SDK usage estimates from evaluation/pricing.json, "
               "not billed amounts or cache savings.\n")
    report += "\n\nRaw answers and per-case judgments are stored in the accompanying JSONL.\n"
    labels = [json.loads(line) for line in args.calibration.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if args.calibration.exists() else []
    agreement = calibration_stats(scored, labels)
    if agreement:
        report += "\n\nJudge–human calibration (pass threshold 0.7):\n\n"
        report += markdown_table(["metric", "n", "MAE", "pass agreement"], [
            [name, values["n"], f"{values['mae']:.3f}", f"{values['pass_agreement']:.3f}"]
            for name, values in agreement.items()
        ])
    else:
        report += "\n\nJudge–human calibration: unavailable; no human-scored subset supplied.\n"
    write_result(f"{date.today()}_generation_eval.md", report)
    detail_path = RESULTS_DIR / f"{date.today()}_generation_eval.jsonl"
    with detail_path.open("w", encoding="utf-8") as out:
        for row in scored:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
