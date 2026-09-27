"""
Build the retrieval evaluation set from the production ChromaDB snapshot.

Three slices, each constructed so the gold label is objective:

  semantic  — gpt-4o-mini rewrites a paper's abstract into a natural question
              that deliberately avoids the title's distinctive wording.
              gold = that one paper.
  exact     — the most distinctive technical term in the title/abstract,
              used verbatim as a short query. gold = that one paper.
  topic     — a technical phrase that appears verbatim in 2-8 papers;
              gold = exactly that set, found by exact string match over the
              corpus. No LLM judgement involved in the gold labels.

Difficulty filter (applied to semantic/exact): the gold paper must sit in a
crowded region of embedding space — at least MIN_NEIGHBOURS other papers
within NEIGHBOUR_DIST. This is computed from the *paper's own* embedding, not
from any query, so it does not favour or penalise any of the four retrieval
configurations being compared.
"""
import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import dump_jsonl, install_usage_patch, USAGE  # noqa: E402

from src.models.embeddings import EmbeddingManager  # noqa: E402
from src.utils.config import settings  # noqa: E402
import openai  # noqa: E402

N_SEMANTIC = 50
N_EXACT = 40
N_TOPIC = 30
MIN_NEIGHBOURS = 3
NEIGHBOUR_DIST = 0.55      # cosine distance; lower = more similar
CANDIDATE_POOL = 400

client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)


def ask(prompt: str, max_tokens: int = 60) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        temperature=0.3,
    )
    USAGE.record(getattr(resp, "usage", None))
    return (resp.choices[0].message.content or "").strip().strip('"')


def main() -> None:
    install_usage_patch()
    em = EmbeddingManager()
    col = em.collection
    print("corpus vectors:", col.count())

    raw = col.get(where={"type": "paper"}, include=["documents", "metadatas"], limit=100000)
    # Production indexes most papers with `paper_id` only; `arxiv_id` is set by
    # just one of the two ingest paths. Use whichever is present as the id.
    papers = [
        {"pid": m.get("paper_id") or m.get("arxiv_id"), "title": m.get("title", ""), "doc": d}
        for d, m in zip(raw["documents"], raw["metadatas"])
        if (m.get("paper_id") or m.get("arxiv_id")) and d
    ]
    print("papers with text:", len(papers))

    random.seed(42)
    pool = random.sample(papers, min(CANDIDATE_POOL, len(papers)))

    # ── difficulty filter: keep papers sitting in crowded neighbourhoods ─────
    crowded = []
    for p in pool:
        hits = em.search(query=p["doc"][:800], n_results=MIN_NEIGHBOURS + 1, filter_type="paper")
        others = [h for h in hits if ((h.get("metadata") or {}).get("paper_id")
                    or (h.get("metadata") or {}).get("arxiv_id")) != p["pid"]]
        close = [h for h in others if (h.get("distance") or 1.0) <= NEIGHBOUR_DIST]
        if len(close) >= MIN_NEIGHBOURS:
            crowded.append(p)
        if len(crowded) >= N_SEMANTIC + N_EXACT:
            break
    print(f"crowded-region papers kept: {len(crowded)} (from {len(pool)} sampled)")

    rows = []

    # ── semantic slice ───────────────────────────────────────────────────────
    for i, p in enumerate(crowded[:N_SEMANTIC]):
        q = ask(
            "Below is a research paper's title and abstract. Write ONE natural "
            "question a researcher might ask that this paper would answer. "
            "Do NOT reuse the distinctive nouns from the title verbatim; "
            "paraphrase. Return only the question.\n\n" + p["doc"][:1200]
        )
        if q:
            rows.append({"id": f"rs{i:03d}", "query": q, "slice": "semantic",
                         "gold": [p["pid"]], "gold_title": p["title"][:110]})
        if i % 10 == 0:
            print(f"  semantic {i}/{N_SEMANTIC}")

    # ── exact-term slice ─────────────────────────────────────────────────────
    for i, p in enumerate(crowded[N_SEMANTIC:N_SEMANTIC + N_EXACT]):
        q = ask(
            "From this paper's title and abstract, extract the single most "
            "distinctive technical term, method name, model name or acronym "
            "(2 to 5 words). Return only that phrase, nothing else.\n\n" + p["doc"][:1200],
            max_tokens=20,
        )
        if q and 2 <= len(q) <= 60:
            rows.append({"id": f"re{i:03d}", "query": q, "slice": "exact",
                         "gold": [p["pid"]], "gold_title": p["title"][:110]})
        if i % 10 == 0:
            print(f"  exact {i}/{N_EXACT}")

    # ── topic slice: gold by exact phrase match over the corpus ──────────────
    # Hand-picked seed terms don't work on a 29k ML corpus: common terms like
    # "diffusion model" hit 1267 papers, which makes Recall@10 meaningless
    # (the ceiling would be 10/1267). Instead mine trigrams whose *document
    # frequency* lands in [MIN_DF, MAX_DF] — small, objective gold sets with
    # no LLM judgement anywhere in the label.
    MIN_DF, MAX_DF = 2, 8
    STOP = {"the", "of", "a", "an", "and", "or", "for", "to", "in", "on", "with",
            "we", "our", "this", "that", "is", "are", "be", "by", "from", "as",
            "it", "can", "which", "these", "such", "using", "based", "via"}

    lowered = [(p, p["doc"].lower()) for p in papers]
    df: dict = {}
    for p, text in lowered:
        head = re.sub(r"[^a-z0-9\- ]+", " ", text[:220])
        words = [w for w in head.split() if w]
        seen_here = set()
        for j in range(len(words) - 2):
            tri = " ".join(words[j:j + 3])
            if tri in seen_here:
                continue
            seen_here.add(tri)
            first, last = words[j], words[j + 2]
            if first in STOP or last in STOP or len(tri) < 14:
                continue
            df.setdefault(tri, []).append(p["pid"])

    candidates = [(t, ids) for t, ids in df.items() if MIN_DF <= len(ids) <= MAX_DF]
    random.shuffle(candidates)
    topic_rows = 0
    for term, ids in candidates:
        if topic_rows >= N_TOPIC:
            break
        rows.append({"id": f"rt{topic_rows:03d}",
                     "query": f"papers about {term}", "slice": "topic",
                     "gold": sorted(set(ids)),
                     "gold_title": f"{len(set(ids))} papers containing '{term}'"})
        topic_rows += 1
    print(f"  topic rows: {topic_rows} (from {len(candidates)} trigrams with df in [{MIN_DF},{MAX_DF}])")

    path = dump_jsonl("retrieval.jsonl", rows)
    by_slice = {}
    for r in rows:
        by_slice[r["slice"]] = by_slice.get(r["slice"], 0) + 1
    print(f"\nwrote {len(rows)} queries -> {path}")
    print("by slice:", by_slice)
    print("generation token usage:", USAGE.snapshot())


if __name__ == "__main__":
    main()
