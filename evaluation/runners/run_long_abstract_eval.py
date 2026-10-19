"""
Does pooling the pieces of a long abstract (EmbeddingManager.embed_papers) help retrieval?

all-MiniLM-L6-v2 reads 256 tokens and drops the rest, so a paper's vector used to miss
the end of a long abstract. For papers whose "title + abstract" is longer than that,
gpt-4o-mini rewrites one sentence from the start of the abstract and one from the part
past the 256-token cut into search questions (paraphrased, so exact words do not decide
the match). Both questions are then searched, vector-only, against an index of papers
embedded the old way (truncated) and the new way (pooled).

  tail  questions about text the truncated vector never saw: should improve
  head  questions about text both vectors see: must not get worse

The index and the questions are built in memory from the local corpus; nothing is
written to the vector store. Questions are cached in datasets/, so a rerun costs nothing.

    python evaluation/runners/run_long_abstract_eval.py --papers 3000 --targets 150
"""
import argparse
import json
import random
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import DATASETS_DIR, markdown_table, result_header, write_result  # noqa: E402

QUESTIONS_FILE = "long_abstract_questions.jsonl"
SEED = 11
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class Questions(BaseModel):
    questions: List[str]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE.split(" ".join(text.split())) if len(s.strip()) > 40]


def head_and_tail(em, title: str, abstract: str):
    """A sentence the truncated vector sees and one wholly past its 256-token cut, or None."""
    tokenizer = em.model.tokenizer
    limit = em.model.max_seq_length - 2
    sentences = split_sentences(abstract)
    if len(sentences) < 3:
        return None
    seen = f"{title}\n\n"
    tail = None
    for index, sentence in enumerate(sentences):
        before = len(tokenizer(seen, add_special_tokens=False)["input_ids"])
        if before > limit and index >= 2:
            tail = sentence
            break
        seen += sentence + " "
    return (sentences[0], tail) if tail else None


def write_questions(papers: List[Dict], em) -> List[Dict]:
    from src.agents.core.structured import structured_call
    from src.utils.tracing import openai_client

    client = openai_client()
    rows = []
    for start in range(0, len(papers), 10):
        batch = papers[start:start + 10]
        sentences = [s for p in batch for s in (p["head"], p["tail"])]
        listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        reply = structured_call(
            client, Questions, task="eval_questions", model="gpt-4o-mini", max_tokens=1500, temperature=0.3,
            messages=[{"role": "user", "content":
                       "Rewrite each statement below as a short search question a researcher might type to "
                       "find a paper about it. Paraphrase: avoid copying distinctive phrases or names. Return "
                       f"exactly {len(sentences)} questions in the same order.\n\n{listing}"}])
        if len(reply.questions) != len(sentences):
            continue
        for i, paper in enumerate(batch):
            rows.append({"arxiv_id": paper["arxiv_id"], "head_q": reply.questions[2 * i],
                         "tail_q": reply.questions[2 * i + 1], "head": paper["head"], "tail": paper["tail"]})
    return rows


def rank_of(query_vec, matrix, target_index: int) -> int:
    scores = matrix @ query_vec
    return int((scores > scores[target_index]).sum()) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--papers", type=int, default=3000)
    parser.add_argument("--targets", type=int, default=150)
    args = parser.parse_args()

    from src.database.models import Paper, SessionLocal
    from src.models.embeddings import EmbeddingManager

    em = EmbeddingManager()
    db = SessionLocal()
    rows = [(a, t, ab) for a, t, ab in db.query(Paper.arxiv_id, Paper.title, Paper.abstract).all() if ab]
    db.close()
    rng = random.Random(SEED)
    corpus = rng.sample(rows, min(args.papers, len(rows)))

    cache = DATASETS_DIR / QUESTIONS_FILE
    if cache.exists():
        questions = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        wanted = {q["arxiv_id"] for q in questions}
        corpus = [r for r in corpus if r[0] not in wanted] + [r for r in rows if r[0] in wanted]
    else:
        candidates = []
        for arxiv_id, title, abstract in corpus:
            pair = head_and_tail(em, title, abstract)
            if pair:
                candidates.append({"arxiv_id": arxiv_id, "head": pair[0], "tail": pair[1]})
            if len(candidates) >= args.targets:
                break
        questions = write_questions(candidates, em)
        cache.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")

    ids = [a for a, _, _ in corpus]
    position = {a: i for i, a in enumerate(ids)}
    texts = [f"{t}\n\n{ab}" for _, t, ab in corpus]
    truncated = np.asarray(em.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False))
    pooled = np.asarray(em.embed_papers([t for _, t, _ in corpus], [ab for _, _, ab in corpus]))
    long_share = sum(len(em.paper_pieces(t, ab)) > 1 for _, t, ab in corpus) / len(corpus)

    results: Dict[str, Dict[str, List[int]]] = {"truncated": {"head": [], "tail": []}, "pooled": {"head": [], "tail": []}}
    for q in questions:
        if q["arxiv_id"] not in position:
            continue
        target = position[q["arxiv_id"]]
        for kind in ("head", "tail"):
            vec = np.asarray(em.model.encode(q[f"{kind}_q"], normalize_embeddings=True))
            results["truncated"][kind].append(rank_of(vec, truncated, target))
            results["pooled"][kind].append(rank_of(vec, pooled, target))

    def metrics(ranks: List[int]) -> List[str]:
        n = len(ranks)
        return [f"{sum(r <= 1 for r in ranks) / n:.3f}", f"{sum(r <= 10 for r in ranks) / n:.3f}",
                f"{sum(1 / r for r in ranks) / n:.3f}"]

    table = [[method, kind, len(results[method][kind]), *metrics(results[method][kind])]
             for kind in ("tail", "head") for method in ("truncated", "pooled")]
    write_result(f"{date.today().isoformat()}_long_abstract_embedding.md", "\n".join([
        result_header("长摘要分段平均嵌入的检索效果", {
            "embedding_model": em.model_name, "index_papers": len(ids), "questions": len(questions),
            "long_abstract_share": f"{long_share:.2f} of indexed papers exceed 256 tokens",
            "questions_file": f"datasets/{QUESTIONS_FILE} (gpt-4o-mini paraphrases)", "search": "vector-only cosine"}),
        "", markdown_table(["向量方式", "问题来自", "n", "Recall@1", "Recall@10", "MRR"], table), "",
        "tail：问题改写自摘要中超过 256 token 的部分（截断时完全看不到）；head：改写自摘要第一句（两种方式都看得到）。", ""]))


if __name__ == "__main__":
    main()
