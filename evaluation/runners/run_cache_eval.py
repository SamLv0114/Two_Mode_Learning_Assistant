"""Small labelled query-pair stress test for proposed semantic cache thresholds.

This checks near-duplicate reuse risk, not production traffic hit rate or cost.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import load_jsonl, markdown_table, mean, result_header, write_result  # noqa: E402
from src.rag.semantic_cache import SemanticRetrievalCache  # noqa: E402
from src.utils.config import settings  # noqa: E402


class MemoryRedis:
    def __init__(self):
        self.values = {}
        self.lists = {}

    def get(self, key):
        return self.values.get(key)

    def lrange(self, key, start, stop):
        return self.lists.get(key, [])[start:] if stop == -1 else self.lists.get(key, [])[start:stop + 1]

    def pipeline(self, transaction=True):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def setex(self, key, ttl, value):
        self.values[key] = value

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key, start, stop):
        self.lists[key] = self.lrange(key, start, stop)

    def expire(self, *_args):
        pass

    def execute(self):
        pass


def main():
    from src.models.embeddings import EmbeddingManager
    rows = load_jsonl("cache_pairs.jsonl")
    model = EmbeddingManager().model
    results = []
    for threshold in (0.98, 0.997):
        scored = []
        for row in rows:
            cache = SemanticRetrievalCache(MemoryRedis(), lambda text: model.encode(text).tolist(), threshold)
            scope = {"user_id": 1, "corpus_revision": "1", "model": settings.EMBEDDING_MODEL}
            cache.put(row["source"], scope, [{"id": row["id"]}])
            started = time.perf_counter()
            hit = cache.get(row["probe"], scope) is not None
            scored.append({"id": row["id"], "hit": hit, "equivalent": row["equivalent"],
                           "lookup_ms": (time.perf_counter() - started) * 1000})
        results.append((threshold, scored))
    report = result_header("语义缓存阈值压力测试", {
        "dataset": f"cache_pairs.jsonl ({len(rows)} hand-labelled pairs)",
        "embedding_model": settings.EMBEDDING_MODEL,
        "scope": "in-memory Redis simulation; no production hit-rate or dollar savings",
    })
    report += "\n\n" + markdown_table(["threshold", "equivalent hit", "wrong reuse", "mean lookup ms"], [
        [f"{threshold:.3f}",
         f"{mean([r['hit'] for r in scored if r['equivalent']]):.3f}",
         f"{mean([r['hit'] for r in scored if not r['equivalent']]):.3f}",
         f"{mean([r['lookup_ms'] for r in scored]):.1f}"]
        for threshold, scored in results
    ])
    write_result(f"{date.today()}_cache_eval.md", report)
    print(report)
    print(json.dumps({str(threshold): scored for threshold, scored in results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
