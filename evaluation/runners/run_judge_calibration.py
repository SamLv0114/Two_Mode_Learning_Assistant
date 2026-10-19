"""
Does the LLM judge agree with a deterministic ground truth?

The judge is CriticAgent's groundedness score: ReflectionAgent re-answers when it is
low, and LLMJudge reports it as answer quality. Its labels can only be trusted if it
agrees with cases whose answer is known without asking a model.

Cases are built from real abstracts, so the gold label holds by construction:

  supported   a sentence copied from the evidence the judge is shown
  number      that sentence with one number changed             (unsupported)
  negation    that sentence with "not" after its first auxiliary (unsupported)
  unrelated   a sentence from a different paper's abstract       (unsupported)

CriticAgent shows the judge only the first EVIDENCE_CHARS characters of each excerpt,
so sentences are taken from that window; a sentence past it would be "supported" in
the abstract but invisible to the judge.

Reported: agreement with the gold labels (accuracy, Cohen's kappa), how often an
unsupported answer is passed (by perturbation), how often a supported one is failed,
a threshold sweep, and test-retest agreement of the judge with itself across repeats.

    python evaluation/runners/run_judge_calibration.py --papers 20 --repeats 2
"""
import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

from _harness import (RESULTS_DIR, USAGE, estimated_cost_usd, install_usage_patch,  # noqa: E402
                      markdown_table, result_header, write_result)

EVIDENCE_CHARS = 700        # CriticAgent cuts each excerpt to this many characters
THRESHOLD = 0.5             # groundedness at or above this counts as "supported"
SWEEP = (0.3, 0.5, 0.7, 0.9)
MIN_SENTENCE, MAX_SENTENCE = 60, 300
SEED = 7

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_AUXILIARY = re.compile(r"\b(is|are|was|were|can|does|do|will|has|have|could|should)\b")


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join((text or "").split())) if s.strip()]


def visible_sentences(abstract: str) -> List[str]:
    """
    Whole sentences of the abstract that lie inside the part the judge is shown. The
    window's last sentence is usually cut short, so a candidate must also be a complete
    sentence of the full abstract.
    """
    complete = set(sentences(abstract))
    window = " ".join(abstract.split())[:EVIDENCE_CHARS]
    return [s for s in sentences(window) if s in complete and MIN_SENTENCE <= len(s) <= MAX_SENTENCE]


def change_number(sentence: str) -> Optional[str]:
    match = _NUMBER.search(sentence)
    if not match:
        return None
    text = match.group(1)
    decimals = len(text.split(".")[1]) if "." in text else 0
    changed = f"{float(text) * 3 + 7:.{decimals}f}"
    return sentence[:match.start(1)] + changed + sentence[match.end(1):]


def negate(sentence: str) -> Optional[str]:
    match = _AUXILIARY.search(sentence)
    if not match:
        return None
    return sentence[:match.end()] + " not" + sentence[match.end():]


def build_cases(papers: Sequence[Dict], seed: int = SEED) -> List[Dict]:
    """One supported case per usable paper plus every perturbation that applies to it."""
    usable = [(p, visible_sentences(p.get("abstract") or "")) for p in papers]
    usable = [(p, options) for p, options in usable if options]
    rng = random.Random(seed)
    cases = []
    for index, (paper, options) in enumerate(usable):
        with_number = [s for s in options if _NUMBER.search(s)]
        chosen = rng.choice(with_number or options)
        other_options = usable[(index + 1) % len(usable)][1] if len(usable) > 1 else []
        variants = {
            "supported": chosen,
            "number": change_number(chosen),
            "negation": negate(chosen),
            "unrelated": rng.choice(other_options) if other_options else None,
        }
        for kind, answer in variants.items():
            if answer and (kind == "supported" or answer != chosen):
                cases.append({
                    "id": f"{paper.get('arxiv_id', index)}:{kind}", "kind": kind,
                    "gold": kind == "supported", "title": paper.get("title", ""),
                    "evidence": paper["abstract"], "answer": answer,
                })
    return cases


def cohen_kappa(gold: Sequence[bool], predicted: Sequence[bool]) -> Optional[float]:
    n = len(gold)
    if not n:
        return None
    observed = sum(g == p for g, p in zip(gold, predicted)) / n
    gold_yes, pred_yes = sum(gold) / n, sum(predicted) / n
    expected = gold_yes * pred_yes + (1 - gold_yes) * (1 - pred_yes)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def summarise(results: List[Dict], threshold: float = THRESHOLD) -> Dict:
    """results: one row per case, with the judge's scores across repeats (None = judge failed)."""
    scored = [r for r in results if r["scores"] and all(s is not None for s in r["scores"])]
    gold = [r["gold"] for r in scored]
    first = [r["scores"][0] >= threshold for r in scored]
    by_kind: Dict[str, List[Dict]] = defaultdict(list)
    for row, verdict in zip(scored, first):
        by_kind[row["kind"]].append({"verdict": verdict, "score": row["scores"][0]})
    repeated = [r for r in scored if len(r["scores"]) > 1]
    stable = [len({s >= threshold for s in r["scores"]}) == 1 for r in repeated]
    drift = [max(r["scores"]) - min(r["scores"]) for r in repeated]
    unsupported = [v for row, v in zip(scored, first) if not row["gold"]]
    supported = [v for row, v in zip(scored, first) if row["gold"]]
    return {
        "cases": len(results), "scored": len(scored), "judge_failures": len(results) - len(scored),
        "accuracy": sum(g == p for g, p in zip(gold, first)) / len(scored) if scored else None,
        "kappa": cohen_kappa(gold, first),
        "false_pass": sum(unsupported) / len(unsupported) if unsupported else None,
        "false_fail": (len(supported) - sum(supported)) / len(supported) if supported else None,
        "by_kind": {kind: {"n": len(rows), "passed": sum(r["verdict"] for r in rows),
                           "mean_score": sum(r["score"] for r in rows) / len(rows)}
                    for kind, rows in sorted(by_kind.items())},
        "sweep": {t: {"accuracy": sum(g == (r["scores"][0] >= t) for g, r in zip(gold, scored)) / len(scored),
                      "kappa": cohen_kappa(gold, [r["scores"][0] >= t for r in scored])}
                  for t in SWEEP} if scored else {},
        "retest_agreement": sum(stable) / len(stable) if stable else None,
        "retest_max_drift": max(drift) if drift else None,
        "retest_mean_drift": sum(drift) / len(drift) if drift else None,
    }


def judge_cases(cases: List[Dict], judge: Callable[[Dict], Optional[float]], repeats: int) -> List[Dict]:
    return [{**case, "scores": [judge(case) for _ in range(repeats)]} for case in cases]


def critic_judge() -> Callable[[Dict], Optional[float]]:
    from src.agents.critic_agent import CriticAgent
    critic = CriticAgent()

    def judge(case: Dict) -> Optional[float]:
        result = critic.evaluate(f'What does the paper "{case["title"]}" report?', case["answer"],
                                 citations=[{"title": case["title"], "evidence": case["evidence"]}])
        return None if result.failed else result.groundedness
    return judge


def load_papers(count: int, seed: int = SEED) -> List[Dict]:
    from src.database.models import Paper, SessionLocal
    db = SessionLocal()
    try:
        rows = (db.query(Paper.arxiv_id, Paper.title, Paper.abstract)
                .filter(Paper.abstract.isnot(None)).order_by(Paper.id).limit(5000).all())
    finally:
        db.close()
    rows = [{"arxiv_id": a, "title": t, "abstract": ab} for a, t, ab in rows if ab and len(ab) > 400]
    return random.Random(seed).sample(rows, min(count, len(rows)))


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def report(summary: Dict, config: Dict, cost: Optional[float]) -> str:
    kinds = {"supported": "有依据（原句）", "number": "改数字", "negation": "加否定", "unrelated": "他文句子"}
    lines = [result_header("LLM 裁判与确定性基准的一致性（groundedness）", config), "",
             "## 1. 总体", "",
             f"- 用例 {summary['cases']} 个，裁判成功打分 {summary['scored']} 个，失败 {summary['judge_failures']} 个",
             f"- 阈值 {THRESHOLD} 下准确率 **{fmt(summary['accuracy'])}**，Cohen's kappa **{fmt(summary['kappa'])}**",
             f"- 误放率（无依据被判为有依据）**{fmt(summary['false_pass'])}**，"
             f"误杀率（有依据被判为无依据）**{fmt(summary['false_fail'])}**",
             f"- 估算花费 {'—' if cost is None else f'${cost:.4f}'}", "",
             "## 2. 按构造方式", "",
             markdown_table(["构造方式", "金标准", "n", "判为有依据", "平均分"],
                            [[kinds.get(k, k), "有依据" if k == "supported" else "无依据", v["n"], v["passed"],
                              fmt(v["mean_score"], 2)] for k, v in summary["by_kind"].items()]), "",
             "## 3. 阈值扫描", "",
             markdown_table(["阈值", "准确率", "kappa"],
                            [[t, fmt(v["accuracy"]), fmt(v["kappa"])] for t, v in summary["sweep"].items()]), "",
             "## 4. 裁判自身的稳定性（同一用例重复打分）", "",
             f"- 判定结论前后一致的比例 **{fmt(summary['retest_agreement'])}**",
             f"- 分数漂移：平均 {fmt(summary['retest_mean_drift'])}，最大 {fmt(summary['retest_max_drift'])}", "",
             "## 说明", "",
             "金标准由构造方式决定，不依赖任何模型判断。句子只取自裁判能看到的证据窗口"
             f"（前 {EVIDENCE_CHARS} 字符）。“改数字”和“加否定”是表层改动，衡量的是裁判能否发现明确矛盾，"
             "不代表它能发现措辞改写后的细微错误。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--papers", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    install_usage_patch()
    USAGE.reset()
    cases = build_cases(load_papers(args.papers))
    results = judge_cases(cases, critic_judge(), args.repeats)
    summary = summarise(results)
    from src.agents.critic_agent import CRITIC_MODEL
    from src.utils.llm_usage import model_for
    config = {"judge": "CriticAgent.groundedness", "judge_model": model_for("critic", CRITIC_MODEL),
              "papers": args.papers, "repeats": args.repeats, "threshold": THRESHOLD, "seed": SEED,
              "evidence_window": f"{EVIDENCE_CHARS} chars"}
    from datetime import date
    stem = f"{date.today().isoformat()}_judge_calibration"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stem}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    write_result(f"{stem}.md", report(summary, config, estimated_cost_usd(USAGE.snapshot())))


if __name__ == "__main__":
    main()
