"""
Retrieval evaluation: four configurations compared on the same query set.

  vector_only        dense vector search only
  bm25_only          lexical BM25 only
  hybrid_rrf         both, fused by Reciprocal Rank Fusion
  hybrid_rrf_rerank  RRF shortlist re-scored by a Cross-Encoder

Metric definitions live in evaluation/README.md. Results are reported per
slice (semantic / exact / topic) as well as overall, because a single blended
number hides which retrieval mode each configuration actually wins on.
"""
import os
import sys
from pathlib import Path

os.environ["VECTOR_DB_DIR"] = "/Users/samlv/researchmate_eval/vector_db_prod"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (  # noqa: E402
    load_jsonl, markdown_table, mean, pct, result_header, write_result,
)

from src.models.embeddings import EmbeddingManager  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402
from src.utils.config import settings  # noqa: E402
import time  # noqa: E402

TOP_K = 10
SLICES = ["semantic", "exact", "topic"]


def arxiv_ids(results) -> list:
    out = []
    for r in results:
        meta = r.get("metadata") or {}
        aid = meta.get("arxiv_id") or meta.get("paper_id")
        if aid:
            out.append(aid)
    return out


def recall_at_k(ranked: list, gold: list, k: int) -> float:
    hit = len(set(ranked[:k]) & set(gold))
    return hit / len(gold)


def rr(ranked: list, gold: list) -> float:
    for i, aid in enumerate(ranked[:TOP_K], start=1):
        if aid in gold:
            return 1.0 / i
    return 0.0


def make_configs(retriever: Retriever):
    def vector_only(q):
        settings.RAG_RERANK_ENABLED = False
        return retriever.retrieve(q, n_results=TOP_K, filter_type="paper", use_hybrid=False)

    def bm25_only(q):
        return retriever._bm25_search(q, TOP_K, "paper", user_id=None)

    def hybrid_rrf(q):
        settings.RAG_RERANK_ENABLED = False
        return retriever.retrieve(q, n_results=TOP_K, filter_type="paper", use_hybrid=True)

    def hybrid_rrf_rerank(q):
        settings.RAG_RERANK_ENABLED = True
        return retriever.retrieve(q, n_results=TOP_K, filter_type="paper", use_hybrid=True)

    return [
        ("vector_only", vector_only),
        ("bm25_only", bm25_only),
        ("hybrid_rrf", hybrid_rrf),
        ("hybrid_rrf_rerank", hybrid_rrf_rerank),
    ]


def main() -> None:
    rows = load_jsonl("retrieval.jsonl")
    em = EmbeddingManager()
    corpus_size = em.collection.count()
    retriever = Retriever(embedding_manager=em)
    print(f"corpus {corpus_size} vectors | {len(rows)} queries")

    per_config = {}
    for name, fn in make_configs(retriever):
        print(f"\n== {name} ==")
        records = []
        for i, row in enumerate(rows):
            t0 = time.time()
            try:
                results = fn(row["query"])
            except Exception as e:
                print(f"  ! {row['id']}: {e}")
                results = []
            elapsed = (time.time() - t0) * 1000
            ranked = arxiv_ids(results)
            gold = row["gold"]
            records.append({
                "slice": row["slice"],
                "r1": recall_at_k(ranked, gold, 1),
                "r5": recall_at_k(ranked, gold, 5),
                "r10": recall_at_k(ranked, gold, 10),
                "mrr": rr(ranked, gold),
                "ms": elapsed,
            })
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)}")
        per_config[name] = records

    # ── report ───────────────────────────────────────────────────────────────
    parts = [result_header("检索评估 baseline（四方案对照）", {
        "corpus_snapshot": "prod 2026-09-19",
        "corpus_size": f"{corpus_size} vectors (type=paper)",
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "dataset": f"retrieval.jsonl ({len(rows)} queries)",
        "top_k": TOP_K,
    })]

    overall_rows = []
    for name, recs in per_config.items():
        overall_rows.append([
            name,
            f"{mean([r['r1'] for r in recs]):.3f}",
            f"{mean([r['r5'] for r in recs]):.3f}",
            f"{mean([r['r10'] for r in recs]):.3f}",
            f"{mean([r['mrr'] for r in recs]):.3f}",
            f"{pct([r['ms'] for r in recs], 0.5):.0f}",
            f"{pct([r['ms'] for r in recs], 0.95):.0f}",
        ])
    parts.append("\n## 总体\n")
    parts.append(markdown_table(
        ["配置", "Recall@1", "Recall@5", "Recall@10", "MRR", "延迟P50(ms)", "延迟P95(ms)"],
        overall_rows,
    ))

    for sl in SLICES:
        sl_rows = []
        for name, recs in per_config.items():
            sub = [r for r in recs if r["slice"] == sl]
            if not sub:
                continue
            sl_rows.append([
                name,
                f"{mean([r['r1'] for r in sub]):.3f}",
                f"{mean([r['r5'] for r in sub]):.3f}",
                f"{mean([r['r10'] for r in sub]):.3f}",
                f"{mean([r['mrr'] for r in sub]):.3f}",
                len(sub),
            ])
        if sl_rows:
            parts.append(f"\n## 分档：{sl}\n")
            parts.append(markdown_table(
                ["配置", "Recall@1", "Recall@5", "Recall@10", "MRR", "query数"], sl_rows))

    write_result("2026-09-19_retrieval_baseline.md", "\n".join(parts))
    for name, recs in per_config.items():
        print(f"{name:<20} R@5={mean([r['r5'] for r in recs]):.3f} "
              f"MRR={mean([r['mrr'] for r in recs]):.3f} "
              f"P50={pct([r['ms'] for r in recs], 0.5):.0f}ms")


if __name__ == "__main__":
    main()
