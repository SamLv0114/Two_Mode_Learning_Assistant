"""
Which sentence-embedding model should index the papers?

Uses the long-abstract evaluation's cached questions (datasets/long_abstract_questions.jsonl,
gpt-4o-mini paraphrases of one sentence from the start of an abstract and one from past
MiniLM's 256-token cut) and the same 3000-paper index, so no model API is called. Each
candidate embeds every paper the way production does (title + abstract in overlapping
pieces that fit the model's input, averaged; EmbeddingManager.embed_papers), embeds the
questions with the model's own query convention, and is scored on where the right paper ranks.

Also reported: weight memory (parameters x 4 bytes) and query latency on this machine's
CPU, since the server has 2 GB of RAM and no GPU.

    python evaluation/runners/run_embedding_model_eval.py
"""
import json
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import DATASETS_DIR, markdown_table, result_header, write_result  # noqa: E402
from run_long_abstract_eval import QUESTIONS_FILE, SEED, rank_of  # noqa: E402

# (label, model id, prefix for questions, prefix for papers)
CANDIDATES = [
    ("MiniLM-L6 (current)", "sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "", ""),
    ("bge-small-en-v1.5 + query instruction", "BAAI/bge-small-en-v1.5",
     "Represent this sentence for searching relevant passages: ", ""),
    ("e5-small-v2", "intfloat/e5-small-v2", "query: ", "passage: "),
]
PAPERS = 3000
BOOTSTRAP_ROUNDS = 2000


def corpus_and_questions():
    """The long-abstract evaluation's index and questions, rebuilt the same way."""
    from src.database.models import Paper, SessionLocal

    db = SessionLocal()
    rows = [(a, t, ab) for a, t, ab in db.query(Paper.arxiv_id, Paper.title, Paper.abstract).all() if ab]
    db.close()
    corpus = random.Random(SEED).sample(rows, min(PAPERS, len(rows)))
    questions = [json.loads(line) for line in (DATASETS_DIR / QUESTIONS_FILE).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    wanted = {q["arxiv_id"] for q in questions}
    corpus = [r for r in corpus if r[0] not in wanted] + [r for r in rows if r[0] in wanted]
    return corpus, questions


def pooled_vectors(em, corpus, passage_prefix: str) -> np.ndarray:
    """EmbeddingManager.embed_papers, with the model's passage prefix on every piece."""
    pieces, owners = [], []
    for i, (_, title, abstract) in enumerate(corpus):
        for piece in em.paper_pieces(title, abstract):
            pieces.append(passage_prefix + piece)
            owners.append(i)
    vectors = np.asarray(em.model.encode(pieces, batch_size=64, normalize_embeddings=True, show_progress_bar=False))
    pooled = np.zeros((len(corpus), vectors.shape[1]))
    np.add.at(pooled, owners, vectors)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True), len(pieces)


def evaluate(label: str, model_id: str, query_prefix: str, passage_prefix: str, corpus, questions,
             cache: Dict[str, object]) -> Dict:
    from sentence_transformers import SentenceTransformer
    from src.models.embeddings import EmbeddingManager

    if model_id not in cache:
        cache[model_id] = SentenceTransformer(model_id, device="cpu")
    model = cache[model_id]
    em = EmbeddingManager.__new__(EmbeddingManager)
    em.model = model
    start = time.perf_counter()
    index, n_pieces = pooled_vectors(em, corpus, passage_prefix)
    index_seconds = time.perf_counter() - start

    position = {a: i for i, (a, _, _) in enumerate(corpus)}
    ranks: Dict[str, List[int]] = {"head": [], "tail": []}
    query_times = []
    for q in questions:
        if q["arxiv_id"] not in position:
            continue
        for kind in ("head", "tail"):
            start = time.perf_counter()
            vec = np.asarray(model.encode(query_prefix + q[f"{kind}_q"], normalize_embeddings=True))
            query_times.append(time.perf_counter() - start)
            ranks[kind].append(rank_of(vec, index, position[q["arxiv_id"]]))

    def metrics(r: List[int]) -> Dict[str, float]:
        return {"r1": sum(x <= 1 for x in r) / len(r), "r10": sum(x <= 10 for x in r) / len(r),
                "mrr": sum(1 / x for x in r) / len(r), "n": len(r)}

    params = sum(p.numel() for p in model.parameters())
    return {"label": label, "model": model_id, "head": metrics(ranks["head"]), "tail": metrics(ranks["tail"]),
            "reciprocal": [1 / x for x in ranks["head"] + ranks["tail"]],
            "max_tokens": model.max_seq_length, "dims": index.shape[1], "pieces": n_pieces,
            "weights_mb": params * 4 / 1e6, "query_ms": 1000 * float(np.median(query_times)),
            "index_s": index_seconds}


def paired_interval(candidate: List[float], baseline: List[float]) -> str:
    """Mean MRR difference over the same questions, with a 95% paired-bootstrap interval."""
    diff = np.asarray(candidate) - np.asarray(baseline)
    rng = np.random.default_rng(SEED)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOTSTRAP_ROUNDS)]
    low, high = np.percentile(means, [2.5, 97.5])
    return f"{diff.mean():+.3f} [{low:+.3f}, {high:+.3f}]"


def main() -> None:
    corpus, questions = corpus_and_questions()
    cache: Dict[str, object] = {}
    results = []
    for label, model_id, query_prefix, passage_prefix in CANDIDATES:
        result = evaluate(label, model_id, query_prefix, passage_prefix, corpus, questions, cache)
        results.append(result)
        print(f"{label}: tail R@10={result['tail']['r10']:.3f} MRR={result['tail']['mrr']:.3f} | "
              f"head R@10={result['head']['r10']:.3f} MRR={result['head']['mrr']:.3f} | "
              f"{result['query_ms']:.1f} ms/query, {result['weights_mb']:.0f} MB")

    baseline = results[0]["reciprocal"]
    quality = [[r["label"], f"{r['tail']['r10']:.3f}", f"{r['tail']['mrr']:.3f}",
                f"{r['head']['r10']:.3f}", f"{r['head']['mrr']:.3f}",
                "—" if i == 0 else paired_interval(r["reciprocal"], baseline)] for i, r in enumerate(results)]
    cost = [[r["label"], r["max_tokens"], r["dims"], r["pieces"], f"{r['weights_mb']:.0f}",
             f"{r['query_ms']:.1f}", f"{r['index_s']:.0f}"] for r in results]
    n = results[0]["tail"]["n"]
    write_result(f"{date.today().isoformat()}_embedding_model_eval.md", "\n".join([
        result_header("论文向量模型对比", {
            "index_papers": len(corpus), "questions": f"{n} papers x (head, tail)",
            "questions_file": f"datasets/{QUESTIONS_FILE} (gpt-4o-mini paraphrases, cached)",
            "papers_embedded_as": "title + abstract, overlapping pieces averaged (embed_papers)",
            "search": "vector-only cosine", "hardware": "local CPU (Apple Silicon), not the server"}),
        "", "## 检索效果", "",
        markdown_table(["模型", "tail Recall@10", "tail MRR", "head Recall@10", "head MRR",
                        "MRR 比现用模型（全部问题，95% 区间）"], quality), "",
        "最后一列：同一道题上两个模型倒数名次之差的均值，配对 bootstrap 2000 次；区间不含 0 才算可信的差别。",
        "tail：问题改写自摘要中超过 256 token 的部分；head：改写自摘要第一句。",
        "问题是从“摘要超过 256 token”的论文里选的，长摘要的比例高于整个语料库。", "",
        "## 代价", "",
        markdown_table(["模型", "最多读入 token", "维度", "切出的段数", "权重 MB", "每个查询 ms", "建索引 s"], cost), "",
        "查询耗时是本机 CPU 上单条问题编码的中位数；服务器是 2 核 2GB 的云主机，绝对值会更慢，相对快慢可以参考。", ""]))


if __name__ == "__main__":
    main()
