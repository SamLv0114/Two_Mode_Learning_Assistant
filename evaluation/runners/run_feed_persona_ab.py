"""
Blind A/B: does the profile feed serve a researcher better than the earlier feed?

Each persona in datasets/feed_personas.json is a researcher with a goal, a level, what
they already know and two papers they found useful. Two feeds of TOP_PAPERS_COUNT papers
are built for it from the same corpus:

  earlier   the path users without a profile get (candidates._fetch_papers_recommended),
            given the persona's goal, topics and pinned titles as its interest text, so
            both feeds start from the same information
  profile   the profile path (profile_feed._fetch_profile_feed): content, citation graph
            and trending recall, then the model's per-paper judgements

An evaluator model sees the persona and both lists (titles and abstracts only, no reasons,
no labels saying which system made which) and marks each paper worth reading this week
or not, then picks the better list. Every pair is judged twice with the lists swapped; a
win counts only when both orders agree, otherwise it is a tie.

Caveats, written into the result: the evaluator is gpt-4o-mini, the same model that judges
papers inside the profile feed, so it may favour that feed's choices; ten personas is a
small sample. Running this adds the graph and trending channels' papers to the local
corpus, exactly as a real feed does.

    python evaluation/runners/run_feed_persona_ab.py
"""
import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal
from unittest.mock import patch

import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import DATASETS_DIR, markdown_table, result_header, write_result  # noqa: E402

PERSONAS_FILE = "feed_personas.json"
EVALUATOR = "gpt-4o-mini"
ABSTRACT_CHARS = 500
FEED_SIZE = 5
SEED = 3


class FeedVerdict(BaseModel):
    useful_a: List[bool]
    useful_b: List[bool]
    winner: Literal["A", "B", "tie"]
    reason: str


def listing(papers: List[Dict]) -> str:
    return "\n\n".join(f"{i + 1}. {p['title']}\n{p['abstract'][:ABSTRACT_CHARS]}" for i, p in enumerate(papers))


def evaluate(client, persona: Dict, first: List[Dict], second: List[Dict]) -> FeedVerdict:
    from src.agents.core.structured import structured_call
    from src.agents.research_profile import prompt_block

    return structured_call(
        client, FeedVerdict, task="eval_feed_ab", model=EVALUATOR, max_tokens=600,
        messages=[{"role": "system", "content":
                   "Two systems each produced a reading list for one researcher. For every paper, decide whether "
                   "this researcher, given their goal and what they already know, would find it worth reading this "
                   "week. Then say which list serves them better overall, or tie. Judge the papers, not the order "
                   "or wording of the lists."},
                  {"role": "user", "content": prompt_block(persona)
                   + f"\n\nList A:\n{listing(first)}\n\nList B:\n{listing(second)}\n\n"
                     f"useful_a needs {len(first)} entries and useful_b needs {len(second)}, in list order."}])


def build_feeds(persona: Dict, pipe) -> Dict[str, List[Dict]]:
    profile = {**persona, "studied": [], "feedback": [], "summary": "", "version": 1}
    as_text = [persona["goal"], *persona["interests"], *(s["title"] for s in persona["seed_papers"])]
    pipe._user_interests = as_text
    earlier = pipe._fetch_papers_recommended(as_text)[:FEED_SIZE]
    current = pipe._fetch_profile_feed(profile)[:FEED_SIZE]

    def rows(papers) -> List[Dict]:
        return [{"arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract or ""} for p in papers]
    return {"earlier": rows(earlier), "profile": rows(current)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--personas", type=int, default=10)
    args = parser.parse_args()

    from src.api.deps import get_embedding_manager
    from src.database.models import SessionLocal
    from src.pipelines.daily_feed import DailyFeedPipeline
    from src.utils.config import settings
    from src.utils.tracing import openai_client

    personas = json.loads((DATASETS_DIR / PERSONAS_FILE).read_text(encoding="utf-8"))[:args.personas]
    db = SessionLocal()
    client = openai_client()
    rng = random.Random(SEED)
    rows, details = [], []
    with patch.object(settings, "TOP_PAPERS_COUNT", FEED_SIZE):
        for persona in personas:
            pipe = DailyFeedPipeline(user_id=-1, db_session=db, embedding_manager=get_embedding_manager())
            pipe._allow_ml = False
            feeds = build_feeds(persona, pipe)
            overlap = len({p["arxiv_id"] for p in feeds["earlier"]} & {p["arxiv_id"] for p in feeds["profile"]})
            # Both orders; which system is "A" first is random, so a label never means a system.
            order = ["earlier", "profile"]
            rng.shuffle(order)
            verdicts = []
            for a, b in (order, order[::-1]):
                v = evaluate(client, persona, feeds[a], feeds[b])
                useful = {a: v.useful_a[:len(feeds[a])], b: v.useful_b[:len(feeds[b])]}
                winner = {"A": a, "B": b}.get(v.winner, "tie")
                verdicts.append((winner, useful, v.reason))
            winners = {w for w, _, _ in verdicts}
            final = winners.pop() if len(winners) == 1 else "tie"
            useful = {s: np.mean([sum(u[s]) for _, u, _ in verdicts]) for s in ("earlier", "profile")}
            rows.append({"persona": persona["name"], "winner": final, "overlap": overlap, **useful})
            details.append({"persona": persona["name"], "feeds": feeds,
                            "verdicts": [{"winner": w, "reason": r} for w, _, r in verdicts]})
            print(f"{persona['name']}: winner={final} useful earlier={useful['earlier']:.1f} "
                  f"profile={useful['profile']:.1f} overlap={overlap}")
    db.close()

    wins = {s: sum(r["winner"] == s for r in rows) for s in ("profile", "earlier", "tie")}
    table = [[r["persona"], r["winner"], f"{r['earlier']:.1f}/{FEED_SIZE}", f"{r['profile']:.1f}/{FEED_SIZE}", r["overlap"]]
             for r in rows]
    detail_path = DATASETS_DIR.parent / "results" / f"{date.today().isoformat()}_feed_persona_ab_lists.json"
    detail_path.write_text(json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
    write_result(f"{date.today().isoformat()}_feed_persona_ab.md", "\n".join([
        result_header("画像推荐 vs 旧推荐：盲评 A/B", {
            "personas": len(rows), "feed_size": FEED_SIZE, "evaluator": f"{EVALUATOR}, both list orders",
            "lists": f"results/{detail_path.name}"}),
        "", f"**胜负（两种顺序结论一致才算胜）**：画像推荐 {wins['profile']}，旧推荐 {wins['earlier']}，平局 {wins['tie']}",
        "", f"**评审认为值得读的论文数（平均，满分 {FEED_SIZE}）**：旧推荐 "
            f"{np.mean([r['earlier'] for r in rows]):.2f}，画像推荐 {np.mean([r['profile'] for r in rows]):.2f}",
        "", markdown_table(["画像", "胜者", "旧推荐值得读", "画像推荐值得读", "两份重合篇数"], table), "",
        "旧推荐拿到的兴趣文本与画像推荐相同（目标、主题、钉选论文标题），差别只在召回和排序方法。",
        "评审模型与画像推荐内部的逐篇判断是同一个模型（gpt-4o-mini），可能偏向画像推荐的选择；10 个画像是小样本。", ""]))


if __name__ == "__main__":
    main()
