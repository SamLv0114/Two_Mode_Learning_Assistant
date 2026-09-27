"""Interactive human-review step for the 50 pending generation candidates.

Requires a real reviewer to inspect source evidence and type an answer. It
never auto-promotes LLM-generated or extractive candidates to verified gold.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "datasets/generation_candidates.jsonl"
GOLDEN = ROOT / "datasets/generation_golden.jsonl"
DRAFTS = ROOT / "datasets/generation_answer_drafts.jsonl"


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def source_digest(row):
    payload = {"question": row["question"], "sources": row["sources"]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def review(reviewer, input_fn=None):
    if input_fn is None:
        input_fn = input
    candidates = read_rows(CANDIDATES)
    verified = {row["id"]: row for row in read_rows(GOLDEN)}
    drafts = {row["id"]: row for row in read_rows(DRAFTS)}
    for row in candidates:
        if row["id"] in verified:
            continue
        print(f"\n[{row['id']}] {row['question']}")
        for source in row["sources"]:
            print(f"\nSOURCE {source['id']} — {source.get('title', '')}\n{source['text']}\n")
        draft = drafts.get(row["id"])
        if draft and draft.get("source_digest") != source_digest(row):
            print("A draft exists, but its question/evidence changed; ignoring the stale draft.")
            draft = None
        if draft:
            print(f"AI draft (unverified): {draft.get('draft_answer') or '[marked unanswerable]'}")
            if draft.get("limitation"):
                print(f"Draft limitation: {draft['limitation']}")
        print("Inspect the source and answer yourself. Type q to stop or s to skip.")
        answer = input_fn("Verified reference answer (Enter accepts the displayed draft): ").strip()
        if answer.lower() == "q":
            break
        if answer.lower() == "s":
            continue
        if not answer and draft and draft.get("answerable"):
            answer = draft.get("draft_answer", "").strip()
        if not answer:
            print("No verified answer; case remains pending.")
            continue
        behavior = input_fn("Expected behavior / uncertainty to preserve: ").strip()
        notes = input_fn("Annotation notes (how you checked the evidence and answer): ").strip()
        confirmed = input_fn("Did you personally verify the source supports this answer? [yes/no] ").strip().lower()
        if not behavior or not notes or confirmed != "yes":
            print("Case not promoted; all fields and explicit source verification are required.")
            continue
        print(f"Draft required tools: {row.get('expected_tools', [])}; forbidden tools: {row.get('forbidden_tools', [])}; order: {row.get('before', [])}")
        tools_confirmed = input_fn("Did you also verify these tool expectations? [yes/no] ").strip().lower() == "yes"
        golden_row = {
            **row, "reference_answer": answer, "expected_behavior": behavior,
            "annotation_notes": notes, "review_status": "human_verified", "reviewer": reviewer,
        }
        if not tools_confirmed:
            for field in ("expected_tools", "forbidden_tools", "before"):
                golden_row.pop(field, None)
        verified[row["id"]] = golden_row
        write_rows(GOLDEN, list(verified.values()))
        print(f"Saved {len(verified)}/{len(candidates)} reviewed cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True, help="Name or stable reviewer ID")
    args = parser.parse_args()
    review(args.reviewer)
