"""Deterministic tool-trajectory evaluation against partial-order labels."""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import RESULTS_DIR, load_jsonl, markdown_table, mean, result_header, write_result  # noqa: E402
from evaluation.runners.run_generation_eval import PREDICTIONS_PATH, load_verified_golden  # noqa: E402


def score_trajectory(actual, required, forbidden, before):
    names = [item["tool"] if isinstance(item, dict) else item for item in actual]
    missing = sorted(set(required) - set(names))
    forbidden_used = sorted(set(forbidden) & set(names))
    order_violations = [
        [first, second] for first, second in before
        if first in names and second in names and names.index(first) >= names.index(second)
    ]
    return {
        "passed": not (missing or forbidden_used or order_violations),
        "missing": missing,
        "forbidden_used": forbidden_used,
        "order_violations": order_violations,
        "tool_calls": len(names),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    args = parser.parse_args()
    golden = load_verified_golden()
    labelled = [r for r in golden if "expected_tools" in r or "forbidden_tools" in r or "before" in r]
    if not labelled:
        raise ValueError("No human-labelled tool expectations in generation_golden.jsonl")
    predictions = {
        row["id"]: row for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip() for row in [json.loads(line)]
    }
    scored = []
    for row in labelled:
        if row["id"] not in predictions:
            raise ValueError(f"Missing prediction for {row['id']}")
        pred = predictions[row["id"]]
        score = score_trajectory(
            pred.get("trace") or pred.get("tools", []), row.get("expected_tools", []),
            row.get("forbidden_tools", []), row.get("before", []),
        )
        scored.append({"id": row["id"], **score})
    report = result_header("Agent 工具轨迹评测", {
        "dataset": f"generation_golden.jsonl ({len(labelled)} labelled trajectories)",
        "predictions": str(args.predictions),
    })
    report += "\n\n" + markdown_table(
        ["指标", "值"],
        [["task_pass_rate", f"{mean([x['passed'] for x in scored]):.3f}"],
         ["mean_tool_calls", f"{mean([x['tool_calls'] for x in scored]):.2f}"],
         ["missing_required_tasks", sum(bool(x["missing"]) for x in scored)],
         ["forbidden_tool_tasks", sum(bool(x["forbidden_used"]) for x in scored)],
         ["order_violation_tasks", sum(bool(x["order_violations"]) for x in scored)]],
    )
    write_result(f"{date.today()}_trajectory_eval.md", report)
    with (RESULTS_DIR / f"{date.today()}_trajectory_eval.jsonl").open("w", encoding="utf-8") as out:
        for row in scored:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
