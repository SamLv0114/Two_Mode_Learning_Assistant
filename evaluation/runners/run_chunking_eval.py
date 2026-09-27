"""Weakly supervised chunk comparison on one immutable full-text snapshot.

Use a writable copy of VECTOR_DB_DIR. The snapshot has already been PDF-parsed,
so this measures chunk boundaries and parent evidence, not table extraction.
"""
import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import markdown_table, mean, result_header, write_result  # noqa: E402
from src.models.embeddings import EmbeddingManager  # noqa: E402
from src.utils.preprocessing import chunk_text, semantic_parent_child_chunks  # noqa: E402


def reconstruct(chunks):
    text = ""
    for part in chunks:
        overlap = next((n for n in range(min(100, len(text), len(part)), 15, -1)
                        if text.endswith(part[:n])), 0)
        text += part[overlap:]
    return text


def make_corpus(collection, max_papers):
    rows = collection.get(where={"type": "paper_fulltext"}, limit=6000,
                          include=["documents", "metadatas"])
    grouped = {}
    for content, meta in zip(rows["documents"], rows["metadatas"]):
        grouped.setdefault(meta["arxiv_id"], []).append((meta.get("chunk_index", 0), content))
    selected = list(grouped.items())[:max_papers]
    if not selected:
        raise ValueError("Snapshot has no paper_fulltext entries")
    corpus = {}
    for paper_id, chunks in selected:
        corpus[paper_id] = reconstruct([text for _, text in sorted(chunks)[:80]])
    if len(corpus) == 1 and max_papers > 1:
        paper_id, text = next(iter(corpus.items()))
        segments = [text[i:i + 5000] for i in range(0, len(text), 5000)]
        corpus = {f"{paper_id}:section_{i}": segment for i, segment in enumerate(segments[:max_papers])}
    return corpus


def weak_queries(corpus):
    rows = []
    for paper_id, text in corpus.items():
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                     if 75 <= len(s.strip()) <= 250 and "http" not in s]
        for sentence in sentences[::max(1, len(sentences) // 2)][:2]:
            rows.append({"paper_id": paper_id, "query": " ".join(sentence.split()[:18]),
                         "target": sentence})
    return rows


def index_records(corpus, mode):
    records = []
    for paper_id, text in corpus.items():
        if mode == "fixed":
            records.extend({"paper_id": paper_id, "child": chunk, "evidence": chunk}
                           for chunk in chunk_text(text, 500, 50))
        else:
            records.extend({"paper_id": paper_id, "child": item["child"],
                            "evidence": item["parent"] if mode == "parent_child" else item["child"]}
                           for item in semantic_parent_child_chunks(text))
    return records


def evaluate(model, corpus, queries, mode):
    records = index_records(corpus, mode)
    vectors = np.asarray(model.encode([r["child"] for r in records], batch_size=64,
                                      show_progress_bar=False), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    query_vectors = np.asarray(model.encode([r["query"] for r in queries], batch_size=64,
                                            show_progress_bar=False), dtype=np.float32)
    query_vectors /= np.maximum(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-8)
    outcomes = []
    for row, query_vector in zip(queries, query_vectors):
        start = time.perf_counter()
        ranks = np.argsort(-(vectors @ query_vector))[:5]
        elapsed = (time.perf_counter() - start) * 1000
        ranked = [records[i] for i in ranks]
        matching = [i + 1 for i, hit in enumerate(ranked) if hit["paper_id"] == row["paper_id"]]
        outcomes.append({"recall5": bool(matching), "rr": 1 / matching[0] if matching else 0,
                         "evidence_hit": any(row["target"] in hit["evidence"] for hit in ranked),
                         "search_ms": elapsed})
    return {"recall5": mean([r["recall5"] for r in outcomes]),
            "mrr5": mean([r["rr"] for r in outcomes]),
            "evidence_hit5": mean([r["evidence_hit"] for r in outcomes]),
            "search_ms": mean([r["search_ms"] for r in outcomes]),
            "index_records": len(records),
            "index_text_bytes": sum(len(json.dumps(r).encode()) for r in records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=12)
    args = parser.parse_args()
    manager = EmbeddingManager()
    corpus = make_corpus(manager.collection, args.max_papers)
    queries = weak_queries(corpus)
    results = {mode: evaluate(manager.model, corpus, queries, mode)
               for mode in ("fixed", "semantic_child", "parent_child")}
    report = result_header("全文切块弱监督对照", {
        "dataset": f"same Chroma full-text snapshot, {len(corpus)} units, {len(queries)} excerpt-derived queries",
        "embedding_model": manager.model.__class__.__name__,
        "limitation": "snapshot has one full-text paper, split into sections; excerpt-derived queries; no human labels or table comparison",
    })
    report += "\n\n" + markdown_table(
        ["mode", "Recall@5", "MRR@5", "target evidence@5", "records", "text bytes", "search ms"],
        [[mode, f"{r['recall5']:.3f}", f"{r['mrr5']:.3f}", f"{r['evidence_hit5']:.3f}",
          r["index_records"], r["index_text_bytes"], f"{r['search_ms']:.2f}"]
         for mode, r in results.items()],
    )
    write_result(f"{date.today()}_chunking_eval.md", report)
    print(report)


if __name__ == "__main__":
    main()
