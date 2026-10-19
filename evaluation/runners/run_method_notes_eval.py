"""
Does the Coding Agent load the right implementation notes for a paper?

method_notes.select() picks method families from a paper's title and abstract by
keyword. A wrong pick puts misleading background into the generation and review
prompts, a missed one loses the help, so both directions are measured against
labelled papers (datasets/method_notes_selection.jsonl):

  precision   of the (paper, family) pairs selected, the share that are right
  recall      of the labelled (paper, family) pairs, the share that are selected
  exact       papers whose selected set equals the labelled set, including "none"

The set is 55 papers drawn 5 per family from what the selector matched in a
18.7k-paper corpus, plus 20 it matched to nothing (seeded sample, 2026-10-16).
Labels were written by Claude Code after reading each abstract, and after seeing the
selector's output, so they are not independent: review them before relying on them.
No model is called; this runs offline in under a second.

    python evaluation/runners/run_method_notes_eval.py
"""
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import load_jsonl, markdown_table, result_header, write_result  # noqa: E402

from src.agents import method_notes  # noqa: E402

DATASET = "method_notes_selection.jsonl"


def evaluate(rows: List[Dict]) -> Dict:
    selected_pairs = gold_pairs = correct_pairs = exact = 0
    errors = []
    per_family: Dict[str, Counter] = {}
    for row in rows:
        selected = {note.id for note in method_notes.select(row["title"], row.get("abstract", ""))}
        gold = set(row["gold"])
        selected_pairs += len(selected)
        gold_pairs += len(gold)
        correct_pairs += len(selected & gold)
        exact += selected == gold
        for family in selected | gold:
            stats = per_family.setdefault(family, Counter())
            stats["selected"] += family in selected
            stats["gold"] += family in gold
            stats["correct"] += family in selected and family in gold
        if selected != gold:
            errors.append({"arxiv_id": row["arxiv_id"], "title": row["title"],
                           "extra": sorted(selected - gold), "missed": sorted(gold - selected)})
    return {
        "papers": len(rows),
        "precision": correct_pairs / selected_pairs if selected_pairs else None,
        "recall": correct_pairs / gold_pairs if gold_pairs else None,
        "exact": exact / len(rows) if rows else None,
        "per_family": per_family,
        "errors": errors,
    }


def fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def report(result: Dict) -> str:
    families = sorted(result["per_family"])
    rows = [[f, s["selected"], s["gold"], s["correct"],
             fmt(s["correct"] / s["selected"] if s["selected"] else None),
             fmt(s["correct"] / s["gold"] if s["gold"] else None)]
            for f in families for s in [result["per_family"][f]]]
    return "\n".join([
        result_header("Coding Agent 方法知识库的选择准确率", {
            "dataset": f"{DATASET} ({result['papers']} papers)",
            "notes": len(method_notes.all_notes()), "max_notes_per_paper": method_notes.MAX_NOTES,
            "labels": "Claude Code, after seeing selector output; pending owner review"}),
        "", "## 1. 总体", "",
        f"- 精确率 **{fmt(result['precision'])}**（选中的“论文-类别”对中，正确的比例）",
        f"- 召回率 **{fmt(result['recall'])}**（应选的“论文-类别”对中，被选中的比例）",
        f"- 完全一致 **{fmt(result['exact'])}**（选中的集合与标注完全相同，含“都不选”）", "",
        "## 2. 按类别", "",
        markdown_table(["类别", "选中", "应选", "正确", "精确率", "召回率"], rows), "",
        "## 3. 不一致的论文", "",
        markdown_table(["arXiv", "标题", "多选", "漏选"],
                       [[e["arxiv_id"], e["title"][:70], ", ".join(e["extra"]) or "—",
                         ", ".join(e["missed"]) or "—"] for e in result["errors"]]), "",
    ]) + "\n"


def main() -> None:
    result = evaluate(load_jsonl(DATASET))
    write_result(f"{date.today().isoformat()}_method_notes_selection.md", report(result))
    print(f"precision={fmt(result['precision'])} recall={fmt(result['recall'])} exact={fmt(result['exact'])}")


if __name__ == "__main__":
    main()
