"""
Candidate retrieval for the paper feed: Latest (by date) and Recommended
(semantic search from a research profile), cold-start seeds, and the
novelty window that keeps the feed from repeating itself.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from src.collectors import PaperData
from src.database.models import Paper, UserInteraction, UserPaperImpression
from src.pipelines.feed.params import (
    CALIBRATION_OVERSAMPLE_FACTOR,
    LIGHTGBM_MIN_INTERACTIONS,
    _blend_weights,
    _expand_focus_areas,
)
from src.utils.config import settings
from src.utils.llm_usage import model_for
from src.utils.redis_client import get_redis
from src.utils.tracing import openai_client

logger = logging.getLogger(__name__)


class CandidateMixin:
    """
    Part of DailyFeedPipeline; relies on the attributes its __init__ sets
    (user_id, db, embedding_manager, generator, feature_extractor, recommender,
    trainer and the per-run memo fields).
    """

    def get_alternate_candidates(
        self,
        exclude_arxiv_ids: set,
        selected_interests: List[str],
        limit: int = 12,
    ) -> List[Dict]:
        """
        Retrieve fresh, ranked candidates that are not already in the feed.

        This exists for /feed/refine's optimizer. That optimizer used to look for
        replacements inside the user's own UserPaperRecommendation rows — but those
        rows *are* the current feed (the pipeline deletes the previous set on every
        run), so the exclusion filter always matched everything and the optimizer
        could never find a single candidate. It scored the feed three times and
        returned it unchanged.

        Going back to the corpus is what makes replacement possible at all. Ranking
        reuses _score_and_rank, so a swapped-in paper is scored on the same terms as
        the ones it joins.
        """
        interests_text = " ".join(_expand_focus_areas(selected_interests))
        candidate_limit = max(limit * 6, settings.TOP_PAPERS_COUNT * 20)

        try:
            profile_query = self._build_user_profile_query(
                interests_text, redis_client=get_redis()
            ) or interests_text
            vector_results = self._fanout_search(profile_query, candidate_limit)
        except Exception as e:
            logger.warning(f"Alternate retrieval failed ({e}) — falling back to impact score")
            vector_results = []

        arxiv_ids, semantic_scores = [], {}
        for r in vector_results:
            aid = r.get("metadata", {}).get("paper_id")
            if aid and aid not in exclude_arxiv_ids:
                arxiv_ids.append(aid)
                distance = r.get("distance") if r.get("distance") is not None else 1.0
                semantic_scores[aid] = max(0.0, 1.0 - float(distance))

        if arxiv_ids:
            db_papers = self.db.query(Paper).filter(Paper.arxiv_id.in_(arxiv_ids)).all()
        else:
            # No vector hits — fall back to the strongest papers not already shown.
            db_papers = (
                self.db.query(Paper)
                .filter(~Paper.arxiv_id.in_(exclude_arxiv_ids or {""}))
                .order_by(Paper.heuristic_impact_score.desc().nullslast())
                .limit(candidate_limit)
                .all()
            )

        if not db_papers:
            return []

        weights = _blend_weights(self._infer_novelty_preference())
        ranked = self._score_and_rank(
            [self._db_paper_to_paperdata(p) for p in db_papers],
            interests_text,
            weights,
            semantic_scores_override=semantic_scores or None,
            top_k=limit,
        )

        # ranked papers come straight from the corpus, so .personalized_summary
        # is still unset at this point (_db_paper_to_paperdata never copies a
        # summary onto PaperData — there is nothing to copy from since Paper
        # has no personalized_summary column). Generate a real one here (same
        # call _generate_summaries makes for a normally-generated feed),
        # otherwise a paper swapped in by /feed/refine renders with no Key
        # Insight / Why It Matters / Relevance to You.
        self._user_interests = selected_interests
        self._generate_summaries(ranked)

        by_arxiv = {p.arxiv_id: p for p in db_papers}
        return [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "relevance_score": p.relevance_score or 0.0,
                "citation_count": p.citation_count,
                "url": p.arxiv_url,
                "summary": p.personalized_summary,
                "db_id": by_arxiv[p.arxiv_id].id if p.arxiv_id in by_arxiv else None,
            }
            for p in ranked
        ]

    def _fetch_papers_recommended(self, selected_interests: List[str]) -> List[PaperData]:
        """
        Recommended mode — two-stage retrieval over the indexed corpus.

        Stage 1 (ChromaDB ANN, fan-out parallel):
            _fanout_search() expands the query into 3 variants, runs parallel
            ChromaDB searches, and merges by doc_id. Richer recall than a single query.
            Falls back to single search if expansion fails.

        Stage 2 (PostgreSQL join + re-rank):
            _infer_novelty_preference() produces a continuous dial that shifts the
            ranking budget between citation authority and recency.
            _familiar_penalty() and _dislike_penalty() demote content the user has
            already seen or explicitly rejected.

        Stage 3 (calibrated selection):
            The ranked list is over-fetched and trimmed to TOP_PAPERS_COUNT by
            _calibrate_top_k(), so a profile that leans heavily toward one
            declared interest can't silently push the others out of the final
            list — see _compute_target_category_distribution().

        Fallback: impact-score DB query if ChromaDB returns too few results.

        There is exactly one candidate source here — the indexed corpus. An earlier
        version swapped the entire source to HuggingFace trending when an inferred
        "frontier" mode fired, which meant identical requests could return results
        drawn from completely different pools with no user-visible reason. Community
        trending now lives in its own clearly-labelled surface (What's Hot).
        """
        interests_text = " ".join(_expand_focus_areas(selected_interests))
        candidate_limit = settings.TOP_PAPERS_COUNT * 20  # e.g. 100

        novelty = self._infer_novelty_preference()
        weights = _blend_weights(novelty)
        logger.info(f"Novelty preference: {novelty:.2f} (weights={weights})")

        # LLM-generated user profile as richer ChromaDB query
        profile_query = self._build_user_profile_query(interests_text, redis_client=get_redis()) or interests_text
        logger.info(f"ChromaDB query: {'profile' if profile_query != interests_text else 'keywords'}")

        # ── Stage 1: fan-out parallel ChromaDB search ────────────────────────
        try:
            vector_results = self._fanout_search(profile_query, candidate_limit)
        except Exception as e:
            logger.warning(f"Fan-out search failed ({e}), falling back to single query")
            vector_results = self.embedding_manager.search(
                interests_text, n_results=candidate_limit, filter_type="paper"
            )

        MIN_CHROMA_RESULTS = settings.TOP_PAPERS_COUNT * 2
        if len(vector_results) < MIN_CHROMA_RESULTS:
            logger.warning(
                f"ChromaDB returned only {len(vector_results)} results "
                f"(need {MIN_CHROMA_RESULTS}) — using impact-score fallback"
            )
            return self._fetch_papers_recommended_fallback(selected_interests)

        # Extract arxiv_ids and semantic similarity scores
        # ChromaDB cosine space: distance = 1 - cosine_similarity → similarity = 1 - distance
        arxiv_ids = []
        semantic_scores: Dict[str, float] = {}
        for r in vector_results:
            arxiv_id = r.get("metadata", {}).get("paper_id")
            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                distance = r.get("distance") if r.get("distance") is not None else 1.0
                semantic_scores[arxiv_id] = max(0.0, 1.0 - float(distance))

        # ── Stage 2: PostgreSQL join ───────────────────────────────────────────
        db_papers = (
            self.db.query(Paper)
            .filter(Paper.arxiv_id.in_(arxiv_ids))
            .all()
        )

        recently_seen = self._get_recently_recommended_arxiv_ids()
        db_papers = [p for p in db_papers if p.arxiv_id not in recently_seen]

        if not db_papers:
            logger.warning("All ChromaDB candidates were recently recommended — using fallback")
            return self._fetch_papers_recommended_fallback(selected_interests)

        logger.info(
            f"Recommended mode: {len(db_papers)} candidates "
            f"(ChromaDB ANN → PostgreSQL join)"
        )

        # ── Stage 3 setup: how many to rank, and what mix to calibrate toward ──
        target_dist = self._compute_target_category_distribution(selected_interests)
        select_count = (
            settings.TOP_PAPERS_COUNT * CALIBRATION_OVERSAMPLE_FACTOR
            if target_dist else settings.TOP_PAPERS_COUNT
        )

        # ── Re-rank: semantic similarity + citation influence ──────────────────
        interaction_count = self.trainer.get_interaction_count()
        if self._use_trained_ranker():
            # LightGBM knows citation counts via feature extractor — just pass candidates
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            papers = [self._db_paper_to_paperdata(p) for p in db_papers]
            ranked = self._rank_and_select(
                papers, select_count, selected_interests, use_ml=True
            )
        else:
            # ── Re-rank: weighted signals + familiarity/dislike penalties ────
            paperdata_list = [self._db_paper_to_paperdata(p) for p in db_papers]
            ranked = self._score_and_rank(
                paperdata_list, interests_text, weights,
                semantic_scores_override={p.arxiv_id: semantic_scores.get(p.arxiv_id, 0.0) for p in db_papers},
                top_k=select_count,
            )

        if target_dist:
            return self._calibrate_top_k(ranked, settings.TOP_PAPERS_COUNT, target_dist)
        return ranked

    def _fetch_papers_recommended_fallback(self, selected_interests: List[str]) -> List[PaperData]:
        """
        Fallback for Recommended mode when ChromaDB has too few papers indexed.
        Ranks the strongest papers by impact score instead of by semantic retrieval.
        Typically only needed before the backfill script has been run.

        Retrieval degrades here, but ranking should not: a user past the LightGBM
        threshold still gets their trained model. This path used to skip that check
        entirely, so a thin ChromaDB silently demoted an experienced user back to
        heuristic scoring with nothing in the response to say so.
        """
        candidate_limit = settings.TOP_PAPERS_COUNT * 20

        db_papers = (
            self.db.query(Paper)
            .filter(Paper.heuristic_impact_score.isnot(None))
            .order_by(Paper.heuristic_impact_score.desc())
            .limit(candidate_limit)
            .all()
        )
        if not db_papers:
            db_papers = (
                self.db.query(Paper)
                .order_by(Paper.citation_count.desc().nullslast())
                .limit(candidate_limit)
                .all()
            )

        if not db_papers:
            logger.warning("No papers in DB — cannot generate recommendations")
            return []

        recently_seen = self._get_recently_recommended_arxiv_ids()
        db_papers = [p for p in db_papers if p.arxiv_id not in recently_seen]

        logger.info(f"Recommended fallback: {len(db_papers)} candidates by impact score")
        papers = [self._db_paper_to_paperdata(p) for p in db_papers]

        interaction_count = self.trainer.get_interaction_count()
        if self._use_trained_ranker():
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            return self._rank_and_select(
                papers, settings.TOP_PAPERS_COUNT, selected_interests, use_ml=True
            )

        return self._rank_with_interests(papers, selected_interests, settings.TOP_PAPERS_COUNT)

    def _fetch_papers_latest(self, selected_interests: List[str], days: int = 7) -> List[PaperData]:
        """
        Latest mode: papers published within the time window, ranked by interest match + influence.
        No external API — pure local DB + embeddings.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        candidate_limit = settings.TOP_PAPERS_COUNT * 20  # broader pool for latest mode

        db_papers = (
            self.db.query(Paper)
            .filter(Paper.published_date >= cutoff)
            .order_by(Paper.published_date.desc())
            .limit(candidate_limit)
            .all()
        )

        if not db_papers:
            logger.warning(f"No papers in DB for last {days} days")
            return []

        recently_seen = self._get_recently_recommended_arxiv_ids()
        db_papers = [p for p in db_papers if p.arxiv_id not in recently_seen]

        logger.info(f"Latest mode: {len(db_papers)} papers from last {days} days")

        papers = [self._db_paper_to_paperdata(p) for p in db_papers]

        interaction_count = self.trainer.get_interaction_count()
        if self._use_trained_ranker():
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            return self._rank_and_select(
                papers, settings.TOP_PAPERS_COUNT, selected_interests, use_ml=True
            )
        else:
            logger.info(f"Interest+influence ranking ({interaction_count}/{LIGHTGBM_MIN_INTERACTIONS} interactions for LightGBM)")
            return self._rank_with_interests(papers, selected_interests, settings.TOP_PAPERS_COUNT)

    def _get_user_saved_arxiv_ids(self, max_ids: int = 10) -> List[str]:
        """Return ArXiv IDs of papers the user has saved or viewed (most recent first)."""
        rows = (
            self.db.query(UserInteraction, Paper.arxiv_id)
            .join(Paper, UserInteraction.item_id == Paper.id)
            .filter(
                UserInteraction.user_id == self.user_id,
                UserInteraction.item_type == "paper",
                UserInteraction.interaction_type.in_(["saved", "viewed"]),
            )
            .order_by(UserInteraction.timestamp.desc())
            .limit(max_ids)
            .all()
        )
        return [arxiv_id for _, arxiv_id in rows]

    def _get_recently_recommended_arxiv_ids(self) -> set:
        """
        ArXiv IDs to exclude from this run's candidates — two independent
        signals, unioned:
          - dismissed within NOVELTY_LOOKBACK_DAYS: an explicit negative
            signal (UserInteraction is append-only, so this reliably covers
            the full window no matter how many times the user has since
            regenerated).
          - merely shown (no dismiss, no save) within SHOWN_LOOKBACK_DAYS: a
            much shorter cooldown, since exposure alone is not evidence of
            dislike and locking it out for the full novelty window would
            needlessly burn through the candidate pool.

        Deliberately does NOT read UserPaperRecommendation for this: that
        table is wiped and replaced by _store_results on every generation,
        so it only ever reflects the single most recent batch, not history.
        UserPaperImpression is the append-only table that actually survives
        repeated force_refresh calls.
        """
        dismiss_cutoff = datetime.now(timezone.utc) - timedelta(days=settings.NOVELTY_LOOKBACK_DAYS)
        dismissed_rows = (
            self.db.query(Paper.arxiv_id)
            .join(UserInteraction, UserInteraction.item_id == Paper.id)
            .filter(
                UserInteraction.user_id == self.user_id,
                UserInteraction.item_type == "paper",
                UserInteraction.interaction_type == "dismissed",
                UserInteraction.timestamp >= dismiss_cutoff,
            )
            .all()
        )

        shown_cutoff = datetime.now(timezone.utc) - timedelta(days=settings.SHOWN_LOOKBACK_DAYS)
        shown_rows = (
            self.db.query(UserPaperImpression.arxiv_id)
            .filter(
                UserPaperImpression.user_id == self.user_id,
                UserPaperImpression.shown_date >= shown_cutoff,
            )
            .all()
        )

        return {r[0] for r in dismissed_rows} | {r[0] for r in shown_rows}

    # ── LLM user-profile query ────────────────────────────────────────────────

    def _build_user_profile_query(self, interests_text: str = "", redis_client=None) -> Optional[str]:
        """
        Generate a 2-3 sentence research profile from the user's saved papers.
        Used as a richer ChromaDB query than a bare keyword list.
        Cached in Redis for 6 hours to avoid redundant LLM calls.

        interests_text (the user's declared focus areas) is passed as context so
        the profile doesn't describe only whatever narrow slice the saved papers
        happen to cover. Without it, an LLM asked to summarize e.g. 8 NLP saves
        has no way to know the user also declared interest in CV, and will
        confidently write a profile that omits it entirely —
        _compute_target_category_distribution() still catches this at final
        selection either way, but a profile that isn't blind to declared scope
        in the first place means fewer candidates get excluded before that
        stage ever sees them.
        """
        cache_key = f"user_profile_query:{self.user_id}"
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return cached
            except Exception as exc:
                logger.warning("Research profile cache read failed: %s", exc)

        saved_ids = self._get_user_saved_arxiv_ids(max_ids=10)
        if not saved_ids:
            return None

        saved_papers = (
            self.db.query(Paper)
            .filter(Paper.arxiv_id.in_(saved_ids))
            .all()
        )
        if not saved_papers:
            return None

        paper_snippets = "\n".join(
            f"- {p.title}: {(p.abstract or '')[:150]}" for p in saved_papers[:8]
        )

        interest_context = (
            f"They have also stated interest in: {interests_text}. If the saved "
            "papers below don't fully reflect that stated scope, still mention it "
            "briefly rather than describing only the narrower pattern in the papers.\n\n"
            if interests_text else ""
        )

        try:
            client = openai_client()
            resp = client.chat.completions.create(
                model=model_for("research_profile", "gpt-4o-mini"),
                messages=[{
                    "role": "user",
                    "content": (
                        f"{interest_context}"
                        "Given these papers a researcher has saved, write a 2-3 sentence "
                        "research profile describing their interests (for use as a search query):\n\n"
                        f"{paper_snippets}\n\nProfile (plain text, no bullets):"
                    )
                }],
                max_tokens=120,
                temperature=0.3,
            )
            profile = resp.choices[0].message.content.strip()
            if redis_client and profile:
                try:
                    redis_client.setex(cache_key, 6 * 3600, profile)
                except Exception as exc:
                    logger.warning("Research profile cache write failed: %s", exc)
            return profile
        except Exception as e:
            logger.debug(f"_build_user_profile_query failed: {e}")
            return None

    # ── Fan-out parallel ChromaDB search ──────────────────────────────────────

    def _fanout_search(self, base_query: str, candidate_limit: int) -> List[Dict]:
        """
        Expand the base query into 2 variants, run 3 parallel ChromaDB searches,
        and merge results deduped by arxiv_id (doc_id).
        """
        from src.agents.tools import _expand_query
        # _expand_query returns [base_query, variant, variant], or [base_query] on failure.
        queries = list(dict.fromkeys(_expand_query(base_query)))

        def _search(q: str) -> List[Dict]:
            try:
                return self.embedding_manager.search(q, n_results=candidate_limit, filter_type="paper")
            except Exception:
                return []

        seen: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=len(queries)) as ex:
            futures = {ex.submit(_search, q): q for q in queries}
            for future in as_completed(futures):
                for r in future.result():
                    arxiv_id = r.get("metadata", {}).get("paper_id")
                    if arxiv_id and arxiv_id not in seen:
                        seen[arxiv_id] = r
                    elif arxiv_id and arxiv_id in seen:
                        # Keep the hit with the smallest distance (closest match)
                        if r.get("distance", 1.0) < seen[arxiv_id].get("distance", 1.0):
                            seen[arxiv_id] = r

        return list(seen.values())

    # ── Cold start: active learning seeds ────────────────────────────────────

    def get_coldstart_seeds(self, n: int = 12) -> List[Dict]:
        """
        Return a diverse set of well-known papers for a new user to react to.

        Without history the pipeline can only fall back to static keyword
        interests, which produces a weak first feed and no signal to learn from.
        Asking for a handful of judgements up front is the cheaper path: rating a
        few papers yields real preference data immediately, and every downstream
        component (profile query, novelty dial, both penalties, LightGBM) already
        reads from the interaction table these ratings land in.

        Selection is high-impact papers spread across topics via MMR, so the set
        covers ground rather than showing twelve variations of one subject.
        """
        pool_size = max(n * 10, 100)

        # Which column ranks the pool is tracked explicitly. Impact score and
        # citation count live on completely different scales, so mixing them
        # inside one sorted list would let a single fallback value dominate.
        rank_field = "heuristic_impact_score"
        db_papers = (
            self.db.query(Paper)
            .filter(Paper.heuristic_impact_score.isnot(None))
            .order_by(Paper.heuristic_impact_score.desc())
            .limit(pool_size)
            .all()
        )
        if not db_papers:
            rank_field = "citation_count"
            db_papers = (
                self.db.query(Paper)
                .order_by(Paper.citation_count.desc().nullslast())
                .limit(pool_size)
                .all()
            )
        if not db_papers:
            logger.warning("Cold-start seeds requested but no papers are indexed")
            return []

        # Skip anything this user has already judged, so returning users who
        # re-open onboarding are not asked the same question twice.
        already_seen = {
            r[0] for r in (
                self.db.query(Paper.arxiv_id)
                .join(UserInteraction, UserInteraction.item_id == Paper.id)
                .filter(
                    UserInteraction.user_id == self.user_id,
                    UserInteraction.item_type == "paper",
                )
                .all()
            )
        }
        db_papers = [p for p in db_papers if p.arxiv_id not in already_seen]
        if not db_papers:
            return []

        # `or` would fall through on a legitimate 0.0 impact score and substitute a
        # citation count in the hundreds, putting that paper at the top of a list
        # scored on a different scale entirely.
        scored = [
            (p, float(getattr(p, rank_field) or 0.0))
            for p in db_papers
        ]
        try:
            # _apply_mmr returns (paper, score) pairs for ranking callers.
            # Cold-start's response below needs Paper objects, not pairs.
            diverse = [paper for paper, _score in self._apply_mmr(scored, top_k=n)]
        except Exception as e:
            logger.warning(f"MMR failed for cold-start seeds ({e}) — using impact order")
            diverse = [p for p, _ in scored[:n]]

        return [
            {
                "db_id": p.id,
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": (p.abstract or "")[:400],
                "categories": p.categories,
                "citation_count": p.citation_count,
                "url": p.arxiv_url or (f"https://arxiv.org/abs/{p.arxiv_id}" if p.arxiv_id else ""),
            }
            for p in diverse[:n]
        ]

    def _db_paper_to_paperdata(self, p: Paper) -> PaperData:
        return PaperData(
            arxiv_id=p.arxiv_id,
            title=p.title,
            abstract=p.abstract or "",
            authors=p.authors.split(", ") if p.authors else [],
            categories=p.categories.split(", ") if p.categories else [],
            published_date=p.published_date,
            arxiv_url=p.arxiv_url or (f"https://arxiv.org/abs/{p.arxiv_id}" if p.arxiv_id else ""),
            pdf_url=p.pdf_url or (f"https://arxiv.org/pdf/{p.arxiv_id}" if p.arxiv_id else ""),
            citation_count=p.citation_count,
        )
