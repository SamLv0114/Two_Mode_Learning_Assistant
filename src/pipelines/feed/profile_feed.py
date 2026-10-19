"""
The feed for a user with a research profile: recall from several channels, pre-rank,
let the model judge a shortlist against the profile, and keep the reasons.

    profile ─┬─ content channel   (profile text → corpus vector search)
             ├─ graph channel     (pinned/saved/implemented papers → Semantic Scholar)
             └─ trending channel  (Hugging Face trending, filtered by the profile)
                    ↓ exclude what was shown, dismissed, saved, implemented or pinned
             pre-rank  (trained ranker if it has beaten the heuristic on held-out data,
                        else the heuristic score in scoring.py)
                    ↓ per-channel quotas → shortlist of ~30
             judge     (one structured judgement per paper, see judge.py)
                    ↓ fixed combination rule
             feed of TOP_PAPERS_COUNT, each with "why this paper"

Latest mode keeps its date window as the only channel; everything after recall is shared.
Users without a profile keep the earlier path in candidates.py.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import numpy as np

from src.agents import research_profile
from src.collectors import PaperData
from src.database.models import Paper, UserInteraction
from src.pipelines.feed import channels as recall
from src.pipelines.feed import judge
from src.pipelines.feed.params import _blend_weights
from src.utils.config import settings
from src.utils.tracing import openai_client

logger = logging.getLogger(__name__)

SHORTLIST_SIZE = sum(recall.CHANNEL_QUOTAS.values())
LIKED_LIMIT = 10
DISLIKED_LIMIT = 10
DISLIKED_DAYS = 60

# arXiv id -> {channel: what that channel can say about the paper}, in the order found
Found = Dict[str, Dict[str, str]]


class ProfileFeedMixin:
    """Part of DailyFeedPipeline (see daily_feed.py)."""

    def _research_profile(self) -> Optional[Dict]:
        """The user's research profile when it is set up enough to recommend from, else None."""
        data = research_profile.to_dict(research_profile.load(self.db, self.user_id))
        return data if research_profile.is_set_up(data) else None

    # ── Inputs ───────────────────────────────────────────────────────────────

    def _feedback_papers(self, kind: str, limit: Optional[int] = None,
                         since: Optional[datetime] = None) -> Dict[str, str]:
        """{arXiv id: title} of papers the user gave this feedback on, most recent first."""
        query = (self.db.query(Paper.arxiv_id, Paper.title)
                 .join(UserInteraction, UserInteraction.item_id == Paper.id)
                 .filter(UserInteraction.user_id == self.user_id, UserInteraction.item_type == "paper",
                         UserInteraction.interaction_type == kind))
        if since is not None:
            query = query.filter(UserInteraction.timestamp >= since)
        papers: Dict[str, str] = {}
        for arxiv_id, title in query.order_by(UserInteraction.timestamp.desc()):
            if arxiv_id:
                papers.setdefault(arxiv_id, title)
            if limit and len(papers) >= limit:
                break
        return papers

    def _liked_papers(self, profile: Dict) -> Dict[str, str]:
        """{id: title} of the papers the graph channel recommends from: pinned, implemented, saved."""
        liked = {s["id"]: s.get("title") or s["id"] for s in profile["seed_papers"]}
        for s in reversed(profile["studied"]):
            liked.setdefault(s["arxiv_id"], s.get("title") or s["arxiv_id"])
        for arxiv_id, title in self._feedback_papers("saved", LIKED_LIMIT).items():
            liked.setdefault(arxiv_id, title)
        return dict(list(liked.items())[:recall.GRAPH_MAX_POSITIVES])

    def _excluded_ids(self, profile: Dict) -> Set[str]:
        """Shown or dismissed lately, already saved, implemented, or pinned: never recommended again."""
        return (self._get_recently_recommended_arxiv_ids() | set(self._feedback_papers("saved"))
                | {s["arxiv_id"] for s in profile["studied"]} | {s["id"] for s in profile["seed_papers"]})

    # ── Recall ───────────────────────────────────────────────────────────────

    def _recall(self, profile: Dict, query: str, profile_vec: np.ndarray, mode: str, days: int) -> Found:
        found: Found = {}

        def add(arxiv_id: str, channel: str, note: str) -> None:
            found.setdefault(arxiv_id, {}).setdefault(channel, note)

        if mode == "latest":
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            for (arxiv_id,) in (self.db.query(Paper.arxiv_id).filter(Paper.published_date >= cutoff)
                                .order_by(Paper.published_date.desc()).limit(settings.TOP_PAPERS_COUNT * 40)):
                add(arxiv_id, "latest", f"Published in the last {days} days")
            return found

        try:
            for hit in self._fanout_search(query, settings.TOP_PAPERS_COUNT * 20):
                if hit.get("metadata", {}).get("paper_id"):
                    add(hit["metadata"]["paper_id"], "content", "Close to your research profile")
        except Exception as exc:
            logger.warning("Content channel failed: %s", exc)

        if settings.FEED_GRAPH_CHANNEL_ENABLED:
            liked = self._liked_papers(profile)
            since = datetime.now(timezone.utc) - timedelta(days=DISLIKED_DAYS)
            disliked = list(self._feedback_papers("dismissed", DISLIKED_LIMIT, since))
            try:
                graph_ids = recall.graph_recall(self.db, self.embedding_manager, list(liked), disliked)
            except Exception as exc:
                logger.warning("Graph channel failed: %s", exc)
                graph_ids = []
            for arxiv_id, note in self._graph_evidence(graph_ids, liked).items():
                add(arxiv_id, "graph", note)

        if settings.FEED_TRENDING_CHANNEL_ENABLED:
            try:
                trending = recall.trending_recall(self.db, self.embedding_manager, profile_vec)
            except Exception as exc:
                logger.warning("Trending channel failed: %s", exc)
                trending = {}
            for arxiv_id, upvotes in trending.items():
                add(arxiv_id, "trending", f"Trending on Hugging Face this week ({upvotes} upvotes)")
        return found

    def _graph_evidence(self, arxiv_ids: List[str], liked: Dict[str, str]) -> Dict[str, str]:
        """For each recommended paper, which liked paper it is closest to (by meaning), as a sentence."""
        titles = list(liked.values())
        vectors = recall.paper_vectors(self.embedding_manager, arxiv_ids)
        title_vecs = (recall.normalise(np.asarray(self.embedding_manager.generate_embeddings(titles), dtype=float))
                      if titles and vectors else None)
        notes = {}
        for arxiv_id in arxiv_ids:
            if title_vecs is not None and arxiv_id in vectors:
                nearest = titles[int(np.argmax(title_vecs @ vectors[arxiv_id]))]
                notes[arxiv_id] = f"Recommended by Semantic Scholar from papers like “{nearest}”"
            else:
                notes[arxiv_id] = "Recommended by Semantic Scholar from papers you pinned or saved"
        return notes

    # ── The feed ─────────────────────────────────────────────────────────────

    def _fetch_profile_feed(self, profile: Dict, mode: str = "recommended", days: int = 7) -> List[PaperData]:
        query = research_profile.query_text(profile)
        profile_vec = recall.query_vector(self.embedding_manager, query)
        found = self._recall(profile, query, profile_vec, mode, days)
        excluded = self._excluded_ids(profile)
        ids = [a for a in found if a not in excluded]
        papers = [self._db_paper_to_paperdata(p) for p in self.db.query(Paper).filter(Paper.arxiv_id.in_(ids))] \
            if ids else []
        if not papers:
            logger.warning("Profile feed found no candidates (mode=%s)", mode)
            return []
        logger.info("Profile feed recall: %s", {c: sum(c in found[p.arxiv_id] for p in papers)
                                                for c in ("content", "graph", "trending", "latest")})
        shortlist = self._shortlist(papers, found, query, profile_vec, profile, mode)
        return self._choose(shortlist, found, profile)

    def _shortlist(self, papers: List[PaperData], found: Found, query: str, profile_vec: np.ndarray,
                   profile: Dict, mode: str) -> List[PaperData]:
        """The best of each channel by pre-rank, up to its quota; each paper's relevance_score is its pre-rank."""
        preranked = self._prerank(papers, query, profile_vec, profile)
        by_channel: Dict[str, List[str]] = {}
        for paper in preranked:
            for channel in found[paper.arxiv_id]:
                by_channel.setdefault(channel, []).append(paper.arxiv_id)
        quotas = {"latest": SHORTLIST_SIZE} if mode == "latest" else recall.CHANNEL_QUOTAS
        chosen = set(recall.merge(by_channel, quotas))
        return [p for p in preranked if p.arxiv_id in chosen]

    def _prerank(self, papers: List[PaperData], query: str, profile_vec: np.ndarray,
                 profile: Dict) -> List[PaperData]:
        """All candidates in pre-rank order, each with relevance_score set."""
        if self._use_trained_ranker():
            return self._rank_and_select(papers, len(papers), profile["interests"], use_ml=True)
        vectors = recall.paper_vectors(self.embedding_manager, [p.arxiv_id for p in papers])
        semantic = {a: max(0.0, float(v @ profile_vec)) for a, v in vectors.items()}
        weights = _blend_weights(self._infer_novelty_preference())
        return self._score_and_rank(papers, query, weights, semantic_scores_override=semantic or None,
                                    top_k=len(papers))

    def _choose(self, shortlist: List[PaperData], found: Found, profile: Dict) -> List[PaperData]:
        """The model judges the shortlist; a fixed rule picks the feed and records why each paper is in it."""
        judgements: Dict[str, judge.PaperJudgement] = {}
        if settings.FEED_LLM_JUDGE_ENABLED:
            judgements = judge.judge(openai_client(), profile, shortlist)
            logger.info("Judged %s of %s shortlisted papers", len(judgements), len(shortlist))
        prerank = {p.arxiv_id: float(p.relevance_score or 0.0) for p in shortlist}
        scored = judge.combine(shortlist, prerank, judgements, profile)
        chosen = []
        for paper, score in judge.select(scored, found, judgements, settings.TOP_PAPERS_COUNT):
            paper.relevance_score = round(score, 4)
            notes = found[paper.arxiv_id]
            paper.reasons = {**judge.reasons(list(notes), list(notes.values()), judgements.get(paper.arxiv_id)),
                             "profile_version": profile["version"]}
            chosen.append(paper)
        return chosen
