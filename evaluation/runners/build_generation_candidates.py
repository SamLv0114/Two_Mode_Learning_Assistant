"""Prepare 50 source-backed draft cases for human golden-set review.

The output is deliberately named `generation_candidates.jsonl`, never
`generation_golden.jsonl`: selecting a paper and copying its abstract is not
human verification of the question, answer, or evidence.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import DATASETS_DIR, load_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-sqlite", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if not args.chroma_sqlite.is_file():
        raise FileNotFoundError(args.chroma_sqlite)
    conn = sqlite3.connect(f"file:{args.chroma_sqlite}?mode=ro", uri=True)
    candidates = []
    try:
        for row in load_jsonl("retrieval.jsonl"):
            if row.get("slice") != "semantic" or len(row.get("gold", [])) != 1:
                continue
            arxiv_id = row["gold"][0]
            found = conn.execute(
                "SELECT e.id FROM embeddings e JOIN embedding_metadata m ON e.id=m.id "
                "WHERE m.key='paper_id' AND m.string_value=? LIMIT 1", (arxiv_id,),
            ).fetchone()
            if not found:
                continue
            metadata = dict(conn.execute(
                "SELECT key,string_value FROM embedding_metadata WHERE id=?", found,
            ))
            title = metadata.get("title") or row.get("gold_title", "")
            document = metadata.get("chroma:document", "")
            excerpt = document.split("Abstract:", 1)[-1].strip()[:1200]
            if not excerpt:
                continue
            source_id = metadata.get("url") or f"https://arxiv.org/abs/{arxiv_id}"
            candidates.append({
                "id": f"gq{len(candidates):03d}",
                "question": row["query"],
                "reference_answer": "",
                "sources": [{"id": source_id, "title": title, "text": excerpt}],
                "relevant_source_ids": [source_id],
                "expected_tools": ["search_knowledge_base"],
                "forbidden_tools": [],
                "before": [],
                "review_status": "pending",
                "reviewer": "",
                "review_notes": "Check that the question is answerable from this excerpt; write a reference answer and verify the source.",
            })
            if len(candidates) >= args.limit:
                break
    finally:
        conn.close()
    if len(candidates) < args.limit:
        raise ValueError(f"Only {len(candidates)} candidate source pairs found; requested {args.limit}")
    out = DATASETS_DIR / "generation_candidates.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates)} unverified candidates to {out}")


if __name__ == "__main__":
    main()
