"""
Routing evaluation: intent classification accuracy, three-stage resolution
distribution, and a threshold sweep.

The three stages are evaluated offline from cached per-query signals rather
than by re-running recognize() once per threshold combination: keyword score
and embedding score are deterministic and threshold-independent, so they are
computed once, and the LLM fallback is called only for the queries that could
ever fall through (those below the strictest threshold in the sweep grid).
That keeps a 30-combination sweep down to one pass of local computation plus a
handful of LLM calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (  # noqa: E402
    install_usage_patch, load_jsonl, markdown_table, mean,
    result_header, write_result,
)

from src.agents.intent_recognizer import IntentRecognizer  # noqa: E402
from src.agents.router import _is_analytical_query, _is_complex_query  # noqa: E402

KW_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
EMB_GRID = [0.55, 0.60, 0.65, 0.70, 0.75]
LIVE_KW, LIVE_EMB = 0.60, 0.65   # what production currently uses


def norm_intent(value) -> str:
    """recognize() returns an Intent enum from stages 1-2 and a plain str from
    stage 3; normalise both to the bare value ('research_qa', ...)."""
    text = str(value)
    return text.split(".")[-1].lower() if text.startswith("Intent.") else text.lower()


def expected_route(query: str) -> str:
    """Replicate AgentRouter._select_research_agent's decision order."""
    if _is_analytical_query(query):
        return "analytical"
    if _is_complex_query(query):
        return "complex"
    return "simple"


def main() -> None:
    install_usage_patch()
    rows = load_jsonl("routing.jsonl")
    rec = IntentRecognizer()

    # ── Pass 1: cache threshold-independent signals ──────────────────────────
    signals = []
    max_kw, max_emb = max(KW_GRID), max(EMB_GRID)
    for row in rows:
        q = row["query"]
        kw_intent, kw_conf = rec._keyword_match(q)
        emb_intent, emb_score = rec._embedding_match(q)
        llm_intent = None
        if kw_conf < max_kw and emb_score < max_emb:
            llm_intent, _ = rec._llm_classify(q)
        signals.append({
            "row": row,
            "kw": (norm_intent(kw_intent) if kw_intent else None, kw_conf),
            "emb": (norm_intent(emb_intent), emb_score),
            "llm": norm_intent(llm_intent) if llm_intent else None,
        })
    print(f"cached signals for {len(signals)} queries "
          f"({sum(1 for s in signals if s['llm'])} needed an LLM call)")

    def decide(sig, kw_t, emb_t):
        kw_intent, kw_conf = sig["kw"]
        if kw_intent and kw_conf >= kw_t:
            return kw_intent, "keyword"
        emb_intent, emb_score = sig["emb"]
        if emb_score >= emb_t:
            return emb_intent, "embedding"
        return sig["llm"], "llm"

    def score(kw_t, emb_t):
        stage = {"keyword": 0, "embedding": 0, "llm": 0}
        correct = 0
        adv_total = adv_correct = 0
        wrong = []
        for sig in signals:
            row = sig["row"]
            pred, used = decide(sig, kw_t, emb_t)
            stage[used] += 1
            ok = pred == row["gold_intent"]
            correct += ok
            if row["slice"] == "adversarial":
                adv_total += 1
                adv_correct += ok
            if not ok:
                wrong.append((row["id"], row["query"], row["gold_intent"], pred, used))
        n = len(signals)
        return {
            "accuracy": correct / n,
            "stage": stage,
            "llm_rate": stage["llm"] / n,
            "adv_accuracy": adv_correct / adv_total if adv_total else 0.0,
            "wrong": wrong,
        }

    live = score(LIVE_KW, LIVE_EMB)

    # ── Route selection accuracy (research_qa second pass) ───────────────────
    route_rows = [r for r in rows if r.get("gold_route")]
    route_correct = sum(expected_route(r["query"]) == r["gold_route"] for r in route_rows)
    route_conf = {}
    for r in route_rows:
        key = (r["gold_route"], expected_route(r["query"]))
        route_conf[key] = route_conf.get(key, 0) + 1

    # ── Sweep ────────────────────────────────────────────────────────────────
    sweep = []
    for kw_t in KW_GRID:
        for emb_t in EMB_GRID:
            s = score(kw_t, emb_t)
            sweep.append([kw_t, emb_t, f"{s['accuracy']:.3f}", f"{s['llm_rate']:.3f}",
                          f"{s['adv_accuracy']:.3f}", s["stage"]["keyword"],
                          s["stage"]["embedding"], s["stage"]["llm"]])

    # ── Report ───────────────────────────────────────────────────────────────
    from src.utils.config import settings
    parts = [result_header("路由评估 baseline", {
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "dataset": f"routing.jsonl ({len(rows)} queries)",
        "live_thresholds": f"keyword={LIVE_KW}, embedding={LIVE_EMB}",
    })]

    parts.append(f"""
## 1. 线上阈值下的表现 (keyword={LIVE_KW}, embedding={LIVE_EMB})

- 意图准确率 **{live['accuracy']:.1%}** ({int(live['accuracy'] * len(rows))}/{len(rows)})
- 对抗样本准确率 **{live['adv_accuracy']:.1%}**
- 三阶段解决占比：关键词 {live['stage']['keyword']}（{live['stage']['keyword']/len(rows):.1%}）、
  向量 {live['stage']['embedding']}（{live['stage']['embedding']/len(rows):.1%}）、
  LLM 兜底 {live['stage']['llm']}（{live['llm_rate']:.1%}）
- **免 LLM 调用完成路由的比例：{1 - live['llm_rate']:.1%}**
""")

    parts.append("## 2. 判错的 query\n")
    parts.append(markdown_table(
        ["id", "query", "gold", "预测", "命中阶段"],
        [[w[0], w[1][:60], w[2], w[3], w[4]] for w in live["wrong"]] or [["-", "无", "-", "-", "-"]],
    ))

    parts.append(f"""
## 3. research_qa 二次路由准确率

{route_correct}/{len(route_rows)} = **{route_correct / len(route_rows):.1%}**

混淆情况（gold -> 实际）：
""")
    parts.append(markdown_table(
        ["gold_route", "实际路由", "条数"],
        [[k[0], k[1], v] for k, v in sorted(route_conf.items())],
    ))

    parts.append("\n## 4. 双阈值扫描\n")
    parts.append(markdown_table(
        ["kw阈值", "emb阈值", "意图准确率", "LLM调用率", "对抗准确率", "关键词解决", "向量解决", "LLM解决"],
        sweep,
    ))

    best = max(sweep, key=lambda r: (float(r[2]), -float(r[3])))
    parts.append(f"""
准确率最高的组合：keyword={best[0]}, embedding={best[1]}，
准确率 {best[2]}，LLM 调用率 {best[3]}。
""")

    write_result("2026-09-19_routing_baseline.md", "\n".join(parts))
    print(f"\nlive accuracy {live['accuracy']:.3f} | llm_rate {live['llm_rate']:.3f} "
          f"| adversarial {live['adv_accuracy']:.3f} | route {route_correct}/{len(route_rows)}")


if __name__ == "__main__":
    main()
