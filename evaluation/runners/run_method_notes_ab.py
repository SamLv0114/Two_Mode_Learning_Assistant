"""
With and without method notes, on real papers: what changes in the Coding Agent's output?

For each paper and configuration this runs the method check, generation, the review,
one revision when the review finds issues, and a second review — the same steps as the
paper graph, without execution. It saves the programs and a summary for reading; it does
not score them, because whether a program is right needs a person (or the sandbox) to
judge. One run per configuration is anecdote: run it several times before concluding.

    python evaluation/runners/run_method_notes_ab.py            # ~$0.03 with gpt-4o-mini
"""
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import RESULTS_DIR  # noqa: E402

from src.agents import method_notes  # noqa: E402

PAPERS = {"2106.09685": "Implement the LoRA update of a linear layer and train only the adapter",
          "2006.11239": "Implement DDPM training and sampling on 2-D toy data"}


def run_paper(services, paper_id: str, objective: str, with_notes: bool) -> dict:
    evidence = services.paper_evidence(paper_id, objective)
    select = method_notes.select if with_notes else (lambda *args, **kwargs: [])
    with patch.object(method_notes, "select", select):
        manifest = services.assess_paper_method(evidence, objective)
        notes = [n.id for n in method_notes.select(evidence["title"], evidence["abstract"], manifest["method"], objective)]
    state = {**evidence, "manifest": manifest, "objective": objective, "method_notes": notes, "plan": {}}
    state["code"] = services.generate_paper(state).get("code", "")
    review = services.review_draft(state) if state["code"] else {"checks": [], "issue_count": None}
    static_before = review.get("static_issues", [])
    if review.get("issue_count"):
        fixed = services.fix_draft(state, review)
        if fixed.get("code"):
            state["code"] = fixed["code"]
            review = services.review_draft(state)
    return {"notes": notes, "feasible": manifest["feasible"],
            "open_choices": [c["question"] for c in manifest.get("open_choices", [])],
            "review_after_fix": dict(Counter(c["status"] for c in review["checks"])),
            "issues_after_fix": review.get("issue_count"),
            "code_checks_before_fix": static_before, "code_checks_after_fix": review.get("static_issues", []),
            "code": state["code"]}


def main() -> None:
    from src.agents.coding_services import CodingServices
    from src.database.models import SessionLocal

    db = SessionLocal()
    results = {}
    try:
        for paper_id, objective in PAPERS.items():
            for with_notes in (True, False):
                services = CodingServices(db)
                key = f"{paper_id}/{'with' if with_notes else 'without'}_notes"
                results[key] = run_paper(services, paper_id, objective, with_notes)
                results[key]["tokens"] = services.input_tokens + services.output_tokens
                print(key, json.dumps({k: v for k, v in results[key].items() if k != "code"}, ensure_ascii=False))
    finally:
        db.close()
    path = RESULTS_DIR / f"{date.today().isoformat()}_method_notes_ab.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", path)


if __name__ == "__main__":
    main()
