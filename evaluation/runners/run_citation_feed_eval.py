"""
Does each stage of the profile feed find papers a researcher actually went on to use?

No user data is needed. A recent paper's reference list records which papers its authors
found useful while working toward that paper's goal. A feed recommends new work, so the
test splits that list by age: what the authors already knew, and the new papers they found
in the months before writing. Each test paper P becomes one simulated reader:

  profile   goal = P's title, pinned papers = 2 of P's older references (picked at random,
            as a reader pins papers they already know); nothing else
  gold      P's references from the GOLD_MONTHS before P (by arXiv id), the new papers a
            feed for this reader should have surfaced
  systems   content    the profile searched against the corpus, pre-ranked (no graph, no model)
            graph      Semantic Scholar's recommendations from the pinned papers, in its own order
            channels   content + graph quota-merged into the 30-paper shortlist, pre-rank order
            judged     that shortlist judged by the model against the profile (the shipped feed)
  metric    hits in the top 10, Precision@10 and nDCG@10 against gold, and how many gold
            papers each system's candidate pool contained at all

The ranking code is the pipeline's own (_fanout_search, _score_and_rank, channels.merge,
judge.judge / combine / select). Nothing is written to the database or the vector store:
graph papers outside the corpus are held in memory for the run.

Caveats, also written into the result: Semantic Scholar's recommender is trained on the
citation graph, which may already contain P's own citations (a leak in the graph's favour),
and references are a proxy for usefulness, not a reader's judgement. Cases are cached in
datasets/, so a rerun only pays for the model's judgements.

    python evaluation/runners/run_citation_feed_eval.py --cases 30
"""
import argparse
import json
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import DATASETS_DIR, markdown_table, result_header, write_result  # noqa: E402

CASES_FILE = "citation_feed_cases.jsonl"
SEED = 7
TOP_K = 10
MIN_GOLD = 3
PINNED = 2
RECENT_DAYS = 120
GOLD_MONTHS = 9      # a reference this much newer than P counts as new work the authors found
MAX_SCANNED = 250    # test papers whose references are fetched while looking for usable cases


def months(arxiv_id: str) -> int:
    """yymm of a new-style arXiv id as a month count, or -1 for old-style ids."""
    head = arxiv_id.split(".")[0]
    return int(head[:2]) * 12 + int(head[2:]) if len(head) == 4 and head.isdigit() else -1


def build_cases(db, n: int) -> List[Dict]:
    from src.collectors import semantic_scholar
    from src.database.models import Paper

    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    pool = [a for (a,) in db.query(Paper.arxiv_id).filter(Paper.published_date >= cutoff).all()]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    cases = []
    for arxiv_id in pool[:MAX_SCANNED]:
        if len(cases) >= n:
            break
        if months(arxiv_id) < 0:
            continue
        refs = [r for r in semantic_scholar.references(arxiv_id) if r.arxiv_id and r.arxiv_id != arxiv_id]
        refs = list({r.arxiv_id: r for r in refs}.values())
        new = [r for r in refs if 0 <= months(arxiv_id) - months(r.arxiv_id) <= GOLD_MONTHS]
        old = [r for r in refs if 0 <= months(r.arxiv_id) and months(arxiv_id) - months(r.arxiv_id) > GOLD_MONTHS]
        if len(new) < MIN_GOLD or len(old) < PINNED:
            continue
        pinned = rng.sample(old, PINNED)
        recommended = semantic_scholar.recommend([p.arxiv_id for p in pinned], limit=60)
        paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).one()
        cases.append({
            "arxiv_id": arxiv_id, "title": paper.title,
            "pinned": [{"id": p.arxiv_id, "title": p.title} for p in pinned],
            "gold": sorted(r.arxiv_id for r in new),
            "graph": [semantic_scholar.as_dict(p) for p in recommended if p.arxiv_id and p.arxiv_id != arxiv_id],
        })
        print(f"case {len(cases)}: {arxiv_id} gold={len(cases[-1]['gold'])} graph={len(cases[-1]['graph'])}")
    return cases


def ndcg(ranked: List[str], gold: set) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, a in enumerate(ranked[:TOP_K]) if a in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), TOP_K)))
    return dcg / ideal if ideal else 0.0


def run_case(case: Dict, pipe, client) -> Dict[str, Dict]:
    from src.agents.research_profile import blank, query_text
    from src.collectors import PaperData
    from src.collectors.semantic_scholar import S2Paper
    from src.database.models import Paper
    from src.pipelines.feed import channels, judge
    from src.pipelines.feed.params import _blend_weights

    profile = {**blank(), "goal": case["title"], "seed_papers": case["pinned"]}
    query = query_text(profile)
    em = pipe.embedding_manager
    profile_vec = channels.query_vector(em, query)
    excluded = {case["arxiv_id"]} | {p["id"] for p in case["pinned"]}

    content_ids = []
    for hit in pipe._fanout_search(query, 100):
        arxiv_id = hit.get("metadata", {}).get("paper_id")
        if arxiv_id and arxiv_id not in excluded and arxiv_id not in content_ids:
            content_ids.append(arxiv_id)
    rows = {p.arxiv_id: p for p in pipe.db.query(Paper).filter(Paper.arxiv_id.in_(content_ids)).all()}
    papers = {a: pipe._db_paper_to_paperdata(rows[a]) for a in content_ids if a in rows}
    graph_ids = []
    for g in (S2Paper(**{k: v for k, v in item.items() if k != "url"}) for item in case["graph"]):
        if g.arxiv_id in excluded or not g.abstract:
            continue
        graph_ids.append(g.arxiv_id)
        # Held in memory, not ingested: the evaluation must not write to the corpus.
        papers.setdefault(g.arxiv_id, PaperData(arxiv_id=g.arxiv_id, title=g.title, authors=g.authors,
                                                abstract=g.abstract, categories=[], published_date=g.published,
                                                arxiv_url=g.url, pdf_url="", citation_count=g.citation_count))

    vectors = channels.paper_vectors(em, list(papers))
    missing = [a for a in papers if a not in vectors]
    if missing:
        fresh = em.embed_papers([papers[a].title for a in missing], [papers[a].abstract for a in missing])
        vectors.update({a: channels.normalise(np.asarray(v, dtype=float)) for a, v in zip(missing, fresh)})
    semantic = {a: max(0.0, float(vectors[a] @ profile_vec)) for a in papers}

    def prerank(ids: List[str]) -> List:
        return pipe._score_and_rank([papers[a] for a in ids], query, _blend_weights(0.5),
                                    semantic_scores_override={a: semantic[a] for a in ids}, top_k=len(ids))

    content_ranked = [p.arxiv_id for p in prerank([a for a in content_ids if a in papers])]
    union = list(dict.fromkeys(content_ids + graph_ids))
    union = [a for a in union if a in papers]
    ranked_union = prerank(union)
    found = {a: (["content"] if a in content_ids else []) + (["graph"] if a in graph_ids else []) for a in union}
    by_channel = {c: [p.arxiv_id for p in ranked_union if c in found[p.arxiv_id]] for c in ("content", "graph")}
    shortlist_ids = set(channels.merge(by_channel, {"content": 18, "graph": 12}))
    shortlist = [p for p in ranked_union if p.arxiv_id in shortlist_ids]
    scores = {p.arxiv_id: float(p.relevance_score or 0) for p in shortlist}

    judgements = judge.judge(client, profile, shortlist)
    judged = judge.select(judge.combine(shortlist, scores, judgements, profile), found, judgements, TOP_K)

    gold = set(case["gold"])
    systems = {"content": (content_ranked, content_ids), "graph": (graph_ids, graph_ids),
               "channels": ([p.arxiv_id for p in shortlist], [p.arxiv_id for p in shortlist]),
               "judged": ([p.arxiv_id for p, _ in judged], [p.arxiv_id for p in shortlist])}
    return {name: {"hits": len(set(ranked[:TOP_K]) & gold), "ndcg": ndcg(ranked, gold),
                   "reachable": len(set(pool) & gold), "gold": len(gold)}
            for name, (ranked, pool) in systems.items()} | {"_judged": len(judgements)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=int, default=30)
    args = parser.parse_args()

    from src.api.deps import get_embedding_manager
    from src.database.models import SessionLocal
    from src.pipelines.daily_feed import DailyFeedPipeline
    from src.utils.tracing import openai_client

    db = SessionLocal()
    cache = DATASETS_DIR / CASES_FILE
    if cache.exists():
        cases = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        cases = build_cases(db, args.cases)
        cache.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    cases = cases[:args.cases]

    pipe = DailyFeedPipeline(user_id=-1, db_session=db, embedding_manager=get_embedding_manager())
    client = openai_client()
    names = ("content", "graph", "channels", "judged")
    results, judged_counts = {n: [] for n in names}, []
    for i, case in enumerate(cases, 1):
        outcome = run_case(case, pipe, client)
        judged_counts.append(outcome.pop("_judged"))
        for name in names:
            results[name].append(outcome[name])
        print(f"{i}/{len(cases)} " + " ".join(f"{n}={outcome[n]['hits']}" for n in names))
    db.close()

    def row(name: str, label: str) -> List:
        rs = results[name]
        hits = [r["hits"] for r in rs]
        return [label, f"{np.mean(hits):.2f}", f"{np.mean(hits) / TOP_K:.3f}",
                f"{np.mean([r['ndcg'] for r in rs]):.3f}", f"{sum(h > 0 for h in hits)}/{len(rs)}",
                f"{np.mean([r['reachable'] / r['gold'] for r in rs]):.2f}"]

    labels = {"content": "content：画像检索语料库 + 预排序", "graph": "graph：Semantic Scholar 推荐原始顺序",
              "channels": "channels：两路配额合并，预排序", "judged": "judged：模型逐篇判断后的最终排序"}
    gold_sizes = [r["gold"] for r in results["content"]]
    write_result(f"{date.today().isoformat()}_citation_feed_eval.md", "\n".join([
        result_header("推荐各阶段的引用离线评测", {
            "cases": len(cases), "gold_per_case": f"median {int(np.median(gold_sizes))} references from the {GOLD_MONTHS} months "
                                              "before the test paper",
            "profile": f"goal = test paper title, {PINNED} of its older references pinned",
            "judge_model": "gpt-4o-mini (feed_judge)", "judged_share": f"{sum(judged_counts)}/{30 * len(cases)}",
            "cases_file": f"datasets/{CASES_FILE}"}),
        "", markdown_table(["系统", f"Top{TOP_K} 命中数", f"Precision@{TOP_K}", f"nDCG@{TOP_K}", "至少命中 1 篇",
                            "候选池覆盖 gold 比例"], [row(n, labels[n]) for n in names]), "",
        f"gold 是测试论文在发表前 {GOLD_MONTHS} 个月内引用的新论文（作者写作期间找到并用上的新工作）；"
        "钉选的是它更早的参考文献。引用是“有用”的代理指标，不等于读者的判断。",
        "Semantic Scholar 的推荐模型基于引用图训练，图里可能已经包含测试论文本身的引用，对 graph 一路有利。", ""]))


if __name__ == "__main__":
    main()
