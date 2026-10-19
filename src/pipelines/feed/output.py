"""
Turning a ranked feed into what is stored and returned: personalised
summaries, the per-user recommendation rows, the /feed/refine context
and the API payload.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.collectors import PaperData
from src.database.models import Paper, UserPaperImpression, UserPaperRecommendation
from src.pipelines.feed.params import _describe_novelty
from src.utils.redis_client import get_redis

logger = logging.getLogger(__name__)


def load_reasons(raw: Optional[str]) -> Optional[Dict]:
    """The stored "why this paper" record, or None (rows from before reasons existed have none)."""
    try:
        value = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def feed_selection(reasons: List[Optional[Dict]]) -> Optional[str]:
    """How a stored feed was chosen: "llm_judge", "prerank" (profile path, model unavailable) or None (earlier path)."""
    if any(r and r.get("goal_fit") is not None for r in reasons):
        return "llm_judge"
    return "prerank" if any(reasons) else None


class OutputMixin:
    """
    Part of DailyFeedPipeline; relies on the attributes its __init__ sets
    (user_id, db, embedding_manager, generator, feature_extractor, recommender,
    trainer and the per-run memo fields).
    """

    def _stored_feed(self) -> Dict:
        """
        This user's feed as stored today, the one payload every caller reads: the reuse
        check in run(), run()'s own result, and (through the same rows) /feed/current.
        "papers" is empty when nothing was stored today.
        """
        start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        rows = (self.db.query(Paper, UserPaperRecommendation)
                .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
                .filter(UserPaperRecommendation.user_id == self.user_id,
                        UserPaperRecommendation.recommended_date >= start_of_day)
                .order_by(UserPaperRecommendation.rank.asc())
                .all())
        first = rows[0][1] if rows else None
        reasons = [load_reasons(rec.reasons_json) for _, rec in rows]
        return {
            "mode": first.feed_mode if first else None,
            "time_window_days": first.feed_time_window_days if first else None,
            "papers": [{"rank": rec.rank or i, "title": paper.title, "arxiv_id": paper.arxiv_id,
                        "url": paper.arxiv_url, "citation_count": paper.citation_count or None,
                        "impact_score": (f"{paper.heuristic_impact_score:.2f}"
                                         if paper.heuristic_impact_score is not None else None),
                        "summary": rec.personalized_summary, "relevance_score": rec.relevance_score,
                        "db_id": paper.id, "reasons": why}
                       for i, ((paper, rec), why) in enumerate(zip(rows, reasons), 1)],
            "profile_version": (reasons[0] or {}).get("profile_version") if reasons else None,
            "selection": feed_selection(reasons),
            "used_ml_ranking": bool(first and first.ranking_source == "ml"),
            "ranking_source": (first.ranking_source if first else None) or "heuristic",
            "ranking_fallback_reason": first.ranking_fallback_reason if first else None,
        }

    def _generate_summaries(self, items: List) -> List:
        """Generate personalized summaries for items (parallel API calls)."""
        if not items:
            return items

        user_interests = getattr(self, '_user_interests', None)

        def summarize(item):
            item.personalized_summary = self.generator.generate_summary(
                title=item.title,
                content=item.abstract,
                user_interests=user_interests,
            )

        with ThreadPoolExecutor(max_workers=min(len(items), 4)) as executor:
            futures = {executor.submit(summarize, item): item for item in items}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Summary generation failed for an item: {e}")

        return items

    
    def _store_results(self, papers: List[PaperData], mode: str = "recommended", time_window_days: int = 7):
        """Store recommended items in database and vector DB"""
        now = datetime.now(timezone.utc)

        # Clear this user's previous feed (same transaction as inserts below)
        self.db.query(UserPaperRecommendation).filter(
            UserPaperRecommendation.user_id == self.user_id
        ).delete(synchronize_session=False)

        for rank, paper_data in enumerate(papers, 1):
            paper = self.db.query(Paper).filter(Paper.arxiv_id == paper_data.arxiv_id).first()
            if paper:
                # Update shared paper fields (citation count, impact score)
                if paper_data.citation_count is not None:
                    paper.citation_count = paper_data.citation_count
                # PaperData never actually carries a heuristic_impact_score
                # (no field for it, nothing ever set one), so the old
                # getattr-guarded copy here was permanently a no-op — the
                # column sat NULL for ~99.7% of papers despite being read
                # for fallback candidate sorting and shown to users as the
                # feed's "Impact" score. Compute it directly from the Paper
                # row instead, which already has title/abstract/authors.
                paper.heuristic_impact_score = self.recommender.calculate_impact_score(paper)

                # Per-user recommendation row — this table is wiped and replaced
                # on every generation (see comment above), so it only ever
                # reflects the current feed, never history.
                self.db.add(UserPaperRecommendation(
                    user_id=self.user_id,
                    paper_id=paper.id,
                    recommended_date=now,
                    personalized_summary=paper_data.personalized_summary,
                    relevance_score=(float(paper_data.relevance_score)
                                     if paper_data.relevance_score is not None else None),
                    rank=rank,
                    feed_mode=mode,
                    feed_time_window_days=time_window_days,
                    ranking_source=getattr(self, "_ranking_source", "heuristic"),
                    ranking_fallback_reason=getattr(self, "_ranking_fallback_reason", None),
                    reasons_json=(json.dumps(paper_data.reasons, ensure_ascii=False)
                                  if getattr(paper_data, "reasons", None) else None),
                ))

                # Append-only impression record — never deleted, this is what
                # _get_recently_recommended_arxiv_ids actually reads to enforce
                # the novelty window across repeated regenerations.
                self.db.add(UserPaperImpression(
                    user_id=self.user_id,
                    arxiv_id=paper_data.arxiv_id,
                    shown_date=now,
                ))

                try:
                    self.embedding_manager.add_paper(
                        paper_data.arxiv_id,
                        paper_data.title,
                        paper_data.abstract,
                        {
                            "title": paper_data.title,
                            "arxiv_id": paper_data.arxiv_id,
                            "url": paper_data.arxiv_url,
                            "published_date": str(paper_data.published_date)
                        }
                    )
                except Exception as e:
                    logger.debug(f"Paper already in vector DB or error: {e}")

        self.db.commit()

    
    def _store_feed_ctx(self, output: Dict, mode: str) -> None:
        """
        Persist feed context in Redis (TTL 2h) for the IterativeRefinementAgent.
        Stores shown paper IDs, the novelty dial, and query info so /feed/refine can
        re-score and adjust the feed without running the full pipeline again.

        The Evaluator reads prose, so the dial is also rendered as a short phrase;
        the raw number is kept alongside it for anything that wants to compute.
        """
        try:
            rc = get_redis()
            if not rc:
                return
            novelty = getattr(self, "_novelty_preference", 0.5)
            ctx = {
                "user_id": self.user_id,
                "feed_mode": mode,
                "novelty_preference": round(novelty, 3),
                "reading_profile": _describe_novelty(novelty),
                "shown_arxiv_ids": [p.get("arxiv_id") for p in output.get("papers", [])],
                "candidate_ids": [p.get("db_id") for p in output.get("papers", [])],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            rc.setex(f"feed_ctx:{self.user_id}", 7200, json.dumps(ctx))
        except Exception as e:
            logger.debug(f"feed_ctx storage skipped: {e}")
