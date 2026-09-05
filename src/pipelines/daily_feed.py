"""
Paper recommendation pipeline.

Entry point is DailyFeedPipeline.run(), triggered by POST /feed/generate. The
user picks one of two modes in the UI; everything after retrieval is shared.

  Latest       papers published inside a time window, newest-first candidates
  Recommended  semantic retrieval across the whole indexed corpus, no date filter

RETRIEVAL
  Latest       PostgreSQL filtered by published_date >= cutoff.
  Recommended  An LLM writes a short research profile from the user's saved
               papers (_build_user_profile_query) and that prose — not a keyword
               list — becomes the ChromaDB query. _fanout_search expands it into
               three variants, searches in parallel, and merges by paper id.
               If ChromaDB returns too little, falls back to top papers by
               impact score.

  Both paths drop papers already recommended inside the novelty lookback window,
  so the feed does not repeat itself day to day.

RANKING  (_score_and_rank — the single heuristic implementation)
      score = w_sem*semantic + w_cit*citation + w_rec*recency
      score *= (1 - 0.30 * familiarity)    similar to papers already saved
      score *= (1 - 0.20 * dislike)        similar to a stated dismiss reason

  Weights come from one continuous dial, not discrete modes. Semantic similarity
  is fixed at 0.55 because it is the point of personalisation; the remaining 0.45
  slides between citation authority and recency according to how the user reads
  (_infer_novelty_preference), shrunk toward neutral when evidence is thin.

  Signals are normalised against the candidates in hand, and a signal that cannot
  separate them is dropped with its weight redistributed. This is what makes
  Latest work: inside a 7-day window no paper has citations yet, so that term
  would otherwise hold weight while contributing nothing.

  Past 50 interactions, LightGBM replaces this scoring entirely
  (_rank_and_select); the heuristic path is what runs before there is enough
  data to train on.

FEEDBACK
  Saves, views and dismissals all land in UserInteraction. Dismissals also go to
  an LLM that records *why*, and those reasons come back as the dislike penalty
  above. New users are offered a diverse set of papers to rate up front
  (get_coldstart_seeds) rather than waiting for this to accumulate.

Community trending content is deliberately NOT here — see What's Hot
(src/agents/hot_news_collector.py). This pipeline personalises; that one does not.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import json
import logging
import numpy as np
import random

from sqlalchemy import exc as sa_exc

from sqlalchemy.orm import Session

from src.collectors import PaperData
from src.models import EmbeddingManager, FeatureExtractor
from src.models.user_recommender import UserRecommender
from src.models.user_trainer import UserModelTrainer
from src.rag import Generator
from src.database.models import Paper, Article, SessionLocal, init_db, UserInteraction, UserPaperRecommendation, UserArticleRecommendation
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Ranking weights are derived from one continuous dial rather than three discrete
# modes. The old learning/frontier/balanced buckets re-derived the same
# authority-vs-recency axis the user already picks explicitly at the Recommended /
# Latest toggle, and a hard threshold on it (recent_ratio > 0.7) could flip a feed
# between very different results after a handful of interactions.
#
# Semantic similarity is how well a paper matches *this* user, so it stays fixed —
# it is the point of personalisation. What actually trades off is how the remaining
# budget is split between established influence and freshness.
SEMANTIC_WEIGHT = 0.55
TRADEOFF_BUDGET = 0.45  # split between citation and recency by the novelty dial

# Multiplicative demotions applied after the weighted score. Both cap well below
# 1.0 so a penalised paper can still surface if nothing better exists — these are
# preferences, not filters.
# Interactions needed before the trained ranker replaces heuristic scoring.
# Named because it gates four separate branches; a literal in each was how one of
# them (the impact-score fallback) ended up never checking it at all.
LIGHTGBM_MIN_INTERACTIONS = 50

FAMILIAR_PENALTY_STRENGTH = 0.30   # similar to something already saved

# Deliberately weaker than the familiar penalty despite representing a stronger
# preference, because the measurement behind it is the shakier of the two.
# Dismiss reasons describe a *property* ("too theoretical", "no code examples"),
# but cosine similarity against a title+abstract matches on *topic*. A paper whose
# abstract merely mentions theory can score close to "too theoretical", and
# embeddings handle negation poorly, so "no code examples" is not reliably
# anti-correlated with papers that do ship code. Until there is real dismiss data
# to check this against, it stays a nudge rather than a strong signal.
DISLIKE_PENALTY_STRENGTH = 0.20


def _blend_weights(novelty: float) -> Dict[str, float]:
    """
    Map a novelty preference in [0, 1] to ranking weights.

      novelty = 0.0  ->  {semantic .55, citation .45, recency .00}   established work
      novelty = 0.5  ->  {semantic .55, citation .23, recency .23}   mixed
      novelty = 1.0  ->  {semantic .55, citation .00, recency .45}   fresh work

    Continuous by construction, so a shift in reading habits nudges the ordering
    instead of swapping it wholesale.
    """
    novelty = min(1.0, max(0.0, novelty))
    return {
        "semantic": SEMANTIC_WEIGHT,
        "citation": TRADEOFF_BUDGET * (1.0 - novelty),
        "recency": TRADEOFF_BUDGET * novelty,
    }


def _describe_novelty(novelty: float) -> str:
    """Render the dial as a phrase for LLM prompts, which reason over prose."""
    if novelty >= 0.70:
        return "prefers recent, cutting-edge work over established literature"
    if novelty >= 0.55:
        return "leans toward recent work while still valuing established results"
    if novelty <= 0.30:
        return "prefers well-established, highly-cited literature"
    if novelty <= 0.45:
        return "leans toward established literature while open to recent work"
    return "balances established literature and recent work"

FOCUS_AREA_TERMS = {
    "ML": "machine learning",
    "NLP": "natural language processing",
    "CV": "computer vision",
    "AI": "artificial intelligence",
    "DL": "deep learning",
}


def _expand_focus_areas(areas: List[str]) -> List[str]:
    return [FOCUS_AREA_TERMS.get(a.upper(), a) for a in areas]


class DailyFeedPipeline:
    """
    Builds one user's paper feed. Construct per request — several intermediate
    results are memoised on the instance for the duration of a single run.

    See the module docstring for how retrieval, ranking and feedback fit together.
    """
    
    def __init__(self, user_id: int, db_session: Session, embedding_manager: EmbeddingManager = None):
        self.user_id = user_id
        self.db = db_session
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.generator = Generator()
        self.feature_extractor = FeatureExtractor()
        self.recommender = UserRecommender(user_id, db_session)
        self.trainer = UserModelTrainer(user_id, db_session, self.embedding_manager)

        # Per-run memoisation. A run is one moment in time, so these are stable
        # once computed, and several ranking paths need the same answers.
        self._novelty_preference: Optional[float] = None
        self._disliked_embs_cached: bool = False
        self._disliked_embs: Optional[np.ndarray] = None
    
    def run(
        self,
        time_window_days: int = 7,
        focus_areas: List[str] = None,
        user_interests: List[str] = None,
        mode: str = "recommended",
        force_refresh: bool = False,
    ):
        """
        Build today's feed.

        Idempotent within a day: a second call returns the feed already generated
        rather than producing a new one. Generating unconditionally made the button
        destructive — it deleted the list the user was reading, spent LLM calls
        re-summarising, and, because papers shown in the last NOVELTY_LOOKBACK_DAYS
        are excluded from candidates, handed back strictly lower-ranked papers each
        time. Clicking twice should not be a downgrade.

        force_refresh=True is the explicit "show me a different batch" path, where
        consuming the novelty window is what the user actually asked for.
        """
        if not force_refresh:
            existing = self._todays_feed()
            if existing:
                logger.info(
                    f"Feed already generated today ({len(existing['papers'])} papers) "
                    f"— returning it unchanged. Use force_refresh for a new batch."
                )
                # Still check training. Interactions accrue between clicks — a user
                # who finishes onboarding after generating would otherwise wait
                # until tomorrow for the model to see any of it.
                self._maybe_retrain()
                return existing

        logger.info(f"Starting daily feed pipeline (mode={mode}, force_refresh={force_refresh})...")

        selected_interests = focus_areas if focus_areas else settings.USER_INTERESTS
        self._user_interests = _expand_focus_areas(focus_areas) if focus_areas else (user_interests or selected_interests)

        # Step 1: Papers. The two modes differ only in how candidates are
        # retrieved; both hand off to the same scoring path afterwards.
        #   latest      — PostgreSQL, published_date >= cutoff
        #   recommended — ChromaDB semantic retrieval over the whole corpus
        logger.info(f"Step 1: Fetching papers (mode={mode})...")
        if mode == "latest":
            top_papers = self._fetch_papers_latest(selected_interests, days=time_window_days)
        else:
            top_papers = self._fetch_papers_recommended(selected_interests)

        # Step 2: Articles — retired.
        # Trending discovery moved to the What's Hot section (HuggingFace + GitHub,
        # free public APIs, Redis-cached and shared across users). Running Tavily
        # discovery here would burn API + LLM calls on content the UI no longer shows.
        top_articles = []

        # Step 3: Summarize papers (parallel LLM calls)
        logger.info("Step 3: Generating personalized summaries...")
        top_papers = self._generate_summaries(top_papers, item_type="paper")

        # Step 4: Store and return
        logger.info("Step 4: Storing results...")
        self._store_results(top_papers, top_articles)
        output = self._format_output(top_papers, top_articles)

        # Persist feed context in Redis for IterativeRefinementAgent
        self._store_feed_ctx(output, mode)

        # Step 5: Retrain if enough interactions
        self._maybe_retrain()

        logger.info("Daily feed pipeline completed!")
        return output

    def _maybe_retrain(self) -> None:
        """
        Retrain the ranker when enough interactions have accumulated.

        Called from both the generate path and the idempotent early return, so
        training keeps up with feedback regardless of which one a click takes.
        Failures are logged, never raised — a training problem should not fail
        the request that happened to trigger it.
        """
        try:
            interaction_count = self.trainer.get_interaction_count()
            if interaction_count >= LIGHTGBM_MIN_INTERACTIONS:
                logger.info(f"Retraining model with {interaction_count} interactions...")
                self.trainer.retrain_model(
                    self.recommender,
                    min_interactions=LIGHTGBM_MIN_INTERACTIONS,
                    use_validation=True,
                )
            else:
                logger.info(
                    f"Skipping retrain "
                    f"({interaction_count}/{LIGHTGBM_MIN_INTERACTIONS} interactions)"
                )
        except Exception as e:
            logger.warning(f"Retrain skipped due to error: {e}")

    def _todays_feed(self) -> Optional[Dict]:
        """
        Return the feed already generated for this user today, or None.

        Keyed on the recommendation timestamp rather than a separate marker so
        there is one source of truth: if the rows are there, the feed exists.
        """
        try:
            start_of_day = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            rows = (
                self.db.query(Paper, UserPaperRecommendation)
                .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
                .filter(
                    UserPaperRecommendation.user_id == self.user_id,
                    UserPaperRecommendation.recommended_date >= start_of_day,
                )
                .order_by(UserPaperRecommendation.rank.asc())
                .all()
            )
            if not rows:
                return None

            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "papers": [
                    {
                        "rank": rec.rank or i,
                        "title": paper.title,
                        "arxiv_id": paper.arxiv_id,
                        "url": paper.arxiv_url,
                        "citation_count": paper.citation_count or None,
                        "impact_score": (
                            f"{paper.heuristic_impact_score:.2f}"
                            if paper.heuristic_impact_score is not None else None
                        ),
                        "summary": rec.personalized_summary,
                        "relevance_score": rec.relevance_score,
                        "db_id": paper.id,
                    }
                    for i, (paper, rec) in enumerate(rows, 1)
                ],
                "articles": [],
                "reused": True,  # lets callers distinguish a cache hit from a fresh run
            }
        except Exception as e:
            # A lookup failure should never block generation — fall through and
            # build the feed normally.
            logger.debug(f"_todays_feed lookup failed ({e}) — will generate")
            return None

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
                redis_client=self._get_redis()
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
        profile_query = self._build_user_profile_query(redis_client=self._get_redis()) or interests_text
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

        # ── Re-rank: semantic similarity + citation influence ──────────────────
        interaction_count = self.trainer.get_interaction_count()
        if interaction_count >= LIGHTGBM_MIN_INTERACTIONS:
            # LightGBM knows citation counts via feature extractor — just pass candidates
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            papers = [self._db_paper_to_paperdata(p) for p in db_papers]
            return self._rank_and_select(
                papers, settings.TOP_PAPERS_COUNT, "paper", selected_interests, use_ml=True
            )

        # ── Re-rank: weighted signals + familiarity/dislike penalties ────────
        paperdata_list = [self._db_paper_to_paperdata(p) for p in db_papers]
        return self._score_and_rank(paperdata_list, interests_text, weights,
                                               semantic_scores_override={p.arxiv_id: semantic_scores.get(p.arxiv_id, 0.0) for p in db_papers})

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
        if interaction_count >= LIGHTGBM_MIN_INTERACTIONS:
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            return self._rank_and_select(
                papers, settings.TOP_PAPERS_COUNT, "paper", selected_interests, use_ml=True
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
        if interaction_count >= LIGHTGBM_MIN_INTERACTIONS:
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            return self._rank_and_select(
                papers, settings.TOP_PAPERS_COUNT, "paper", selected_interests, use_ml=True
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
        """ArXiv IDs already recommended to this user in the novelty lookback window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.NOVELTY_LOOKBACK_DAYS)
        rows = (
            self.db.query(Paper.arxiv_id)
            .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
            .filter(
                UserPaperRecommendation.user_id == self.user_id,
                UserPaperRecommendation.recommended_date >= cutoff,
            )
            .all()
        )
        return {r[0] for r in rows}

    # ── Shared Redis handle ───────────────────────────────────────────────────

    def _get_redis(self):
        """Return a Redis client, or None when Redis is unconfigured/unreachable."""
        try:
            import redis
            if not settings.REDIS_URL:
                return None
            rc = redis.from_url(settings.REDIS_URL, decode_responses=True)
            rc.ping()
            return rc
        except Exception:
            return None

    # ── Novelty preference (continuous) ───────────────────────────────────────

    # Evidence needed before the dial is trusted at full strength. Below this the
    # estimate is shrunk toward neutral, so a couple of clicks cannot swing the feed.
    NOVELTY_CONFIDENCE_N = 20
    # Citation count at which a reader is treated as fully authority-oriented.
    NOVELTY_CITATION_SATURATION = 200.0

    def _infer_novelty_preference(self) -> float:
        """
        Estimate how much this user favours fresh work over established work,
        as a continuous value in [0, 1] (0 = authority, 1 = novelty, 0.5 = neutral).

        Two behavioural signals, both read from recent saved/viewed papers:
          - how often they engage with papers published in the last 30 days
          - how highly cited the papers they engage with tend to be (inverted)

        The raw estimate is shrunk toward 0.5 in proportion to how little evidence
        exists. With no history the result is exactly neutral, which is the honest
        answer for a new user rather than a guess dressed up as a mode.

        The result is memoised on the instance. A pipeline run is a single point in
        time, so the answer cannot change mid-run, and every path that ranks —
        Recommended, Latest, and the impact-score fallback — needs it. Caching here
        rather than at each call site is also what guarantees _store_feed_ctx sees
        the real value: the Latest path used to compute it into a local and drop it,
        leaving feed_ctx to report a default the ranking never used.
        """
        if self._novelty_preference is not None:
            return self._novelty_preference

        self._novelty_preference = self._compute_novelty_preference()
        return self._novelty_preference

    def _compute_novelty_preference(self) -> float:
        """Uncached estimate — see _infer_novelty_preference for semantics."""
        try:
            cutoff_30d = datetime.now(timezone.utc) - timedelta(days=30)
            rows = (
                self.db.query(UserInteraction, Paper)
                .join(Paper, UserInteraction.item_id == Paper.id)
                .filter(
                    UserInteraction.user_id == self.user_id,
                    UserInteraction.item_type == "paper",
                    UserInteraction.interaction_type.in_(["saved", "viewed"]),
                )
                .order_by(UserInteraction.timestamp.desc())
                .limit(30)
                .all()
            )
            if not rows:
                return 0.5

            recent_count = sum(
                1 for _, p in rows
                if p.published_date and p.published_date.replace(tzinfo=timezone.utc) >= cutoff_30d
            )
            recency_signal = recent_count / len(rows)

            citations = [p.citation_count or 0 for _, p in rows]
            avg_citations = sum(citations) / len(citations)
            # Heavy citation reading pulls toward authority, i.e. low novelty.
            authority_signal = min(avg_citations / self.NOVELTY_CITATION_SATURATION, 1.0)
            citation_signal = 1.0 - authority_signal

            # Recency of what they read is the more direct evidence of the two.
            raw = 0.7 * recency_signal + 0.3 * citation_signal

            confidence = min(len(rows) / self.NOVELTY_CONFIDENCE_N, 1.0)
            novelty = 0.5 + (raw - 0.5) * confidence

            logger.debug(
                f"Novelty preference: raw={raw:.2f} confidence={confidence:.2f} "
                f"-> {novelty:.2f} (n={len(rows)}, recent={recency_signal:.2f}, "
                f"avg_cit={avg_citations:.0f})"
            )
            return min(1.0, max(0.0, novelty))
        except Exception as e:
            logger.debug(f"_infer_novelty_preference failed ({e}) — using neutral")
            return 0.5

    # ── LLM user-profile query ────────────────────────────────────────────────

    def _build_user_profile_query(self, redis_client=None) -> Optional[str]:
        """
        Generate a 2-3 sentence research profile from the user's saved papers.
        Used as a richer ChromaDB query than a bare keyword list.
        Cached in Redis for 6 hours to avoid redundant LLM calls.
        """
        cache_key = f"user_profile_query:{self.user_id}"
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return cached
            except Exception:
                pass

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

        try:
            from openai import OpenAI
            from src.utils.config import settings
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
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
                except Exception:
                    pass
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
        variants = _expand_query(base_query)  # returns up to 2 expanded strings
        queries = [base_query] + variants

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

    # ── Familiar-paper penalty (diversity signal) ─────────────────────────────

    def _get_saved_paper_embeddings(self) -> Optional[np.ndarray]:
        """Return stacked embeddings of the user's 10 most-recently saved papers."""
        saved_ids = self._get_user_saved_arxiv_ids(max_ids=10)
        if not saved_ids:
            return None
        saved_papers = self.db.query(Paper).filter(Paper.arxiv_id.in_(saved_ids)).all()
        if not saved_papers:
            return None
        texts = [f"{p.title} {(p.abstract or '')[:300]}" for p in saved_papers]
        embs = np.array(self.embedding_manager.generate_embeddings(texts))
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    def _familiar_penalty(self, paper_emb: np.ndarray, saved_embs: Optional[np.ndarray]) -> float:
        """
        Similarity of a candidate paper to the user's saved papers.
        High penalty → paper is too similar to what the user already has.
        Returns a value in [0, 1].
        """
        if saved_embs is None or len(saved_embs) == 0:
            return 0.0
        sims = saved_embs @ paper_emb
        return float(np.max(sims))

    # ── Dislike penalty (closes the dismiss-reflexion loop) ───────────────────

    def _get_disliked_embeddings(self) -> Optional[np.ndarray]:
        """
        Embed the reasons this user has dismissed papers, from UserFactMemory.

        Dismiss reflexion already asks an LLM why a paper was rejected and stores
        the answer as a negative fact, but nothing on the ranking side ever read
        those facts back — the loop was written but never closed. This is the read
        end of it.

        Memoised per run: this hits Redis and embeds up to ten strings, and several
        ranking paths can reach it within one generation.

        Returns normalised embeddings, or None when there is nothing to avoid.
        """
        if self._disliked_embs_cached:
            return self._disliked_embs
        self._disliked_embs_cached = True
        self._disliked_embs = self._compute_disliked_embeddings()
        return self._disliked_embs

    def _compute_disliked_embeddings(self) -> Optional[np.ndarray]:
        """Uncached — see _get_disliked_embeddings."""
        try:
            from src.agents.memory import UserFactMemory

            facts = UserFactMemory()._load(self.user_id, self._get_redis())
            reasons = [
                # Strip the "user disliked: " prefix — embedding the bare reason
                # keeps the vector on the topic rather than on the phrasing.
                f["fact"].split("user disliked:", 1)[-1].strip()
                for f in facts
                if f.get("negative") and f.get("fact")
            ]
            reasons = [r for r in reasons if r][:10]
            if not reasons:
                return None

            embs = np.array(self.embedding_manager.generate_embeddings(reasons))
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            logger.info(f"Dislike penalty active: {len(reasons)} negative fact(s)")
            return embs / norms
        except Exception as e:
            logger.debug(f"_get_disliked_embeddings skipped: {e}")
            return None

    def _dislike_penalty(self, paper_emb: np.ndarray, disliked_embs: Optional[np.ndarray]) -> float:
        """
        Similarity of a candidate paper to reasons this user has rejected papers.
        Returns a value in [0, 1]; higher means closer to something they disliked.
        """
        if disliked_embs is None or len(disliked_embs) == 0:
            return 0.0
        sims = disliked_embs @ paper_emb
        return float(np.max(sims))


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
            diverse = self._apply_mmr(scored, top_k=n, item_type="paper")
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

    def _score_and_rank(
        self,
        papers: List[PaperData],
        interests_text: str,
        weights: Dict,
        semantic_scores_override: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> List[PaperData]:
        """
        The single heuristic ranking implementation. Every non-ML path routes
        through it, which is what keeps the novelty dial and both penalties
        applying uniformly rather than only on whichever path was edited last.

        score = w_sem*semantic + w_cit*citation + w_rec*recency
                then demoted by familiarity and by stated dislikes

        Weights come from the caller (see _blend_weights); signals that cannot
        separate the candidates in hand are dropped and their weight redistributed.
        LightGBM ranking is a separate path — see _rank_and_select.
        """
        if not papers:
            return []

        now = datetime.now(timezone.utc)

        # Compute semantic scores (batch, unless provided by caller)
        if semantic_scores_override:
            sem_scores = semantic_scores_override
        else:
            paper_texts = [f"{p.title} {(p.abstract or '')[:400]}" for p in papers]
            all_texts = [interests_text] + paper_texts
            all_embs = np.array(self.embedding_manager.generate_embeddings(all_texts))
            interest_emb = all_embs[0]
            paper_embs = all_embs[1:]
            i_norm = interest_emb / (np.linalg.norm(interest_emb) + 1e-8)
            norms = np.linalg.norm(paper_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sims = (paper_embs / norms) @ i_norm
            sem_scores = {p.arxiv_id: float(s) for p, s in zip(papers, sims)}

        # ── Signal normalisation, scaled to this candidate set ────────────────
        # Both signals are normalised against the candidates actually in hand
        # rather than a fixed scale. A fixed scale degenerates whenever the pool
        # is narrow: over a 7-day Latest window every paper sits within 2% of
        # every other on a 365-day recency curve, and none of them have citation
        # data yet, so both terms collapse to near-constants that add nothing to
        # the ordering while still holding weight.
        cit_vals = [float(p.citation_count or 0) for p in papers]
        cit_lo, cit_hi = min(cit_vals), max(cit_vals)
        cit_spread = cit_hi - cit_lo

        def _ages() -> List[float]:
            out = []
            for p in papers:
                if not p.published_date:
                    out.append(None)
                    continue
                pub = p.published_date
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                out.append(float(max(0, (now - pub).days)))
            return out

        ages = _ages()
        known_ages = [a for a in ages if a is not None]
        age_lo = min(known_ages) if known_ages else 0.0
        age_hi = max(known_ages) if known_ages else 0.0
        age_spread = age_hi - age_lo

        # ── Drop weight from signals that cannot separate these candidates ────
        # A signal with no spread adds the same constant to every score, so its
        # weight does no ranking work. Redistributing it proportionally to the
        # signals that *can* separate candidates keeps the full budget useful,
        # and does so from the data rather than from a mode flag — Latest mode
        # gets this automatically, and so does any other pool that happens to be
        # uniform. As citation_refresh backfills real counts, the citation term
        # re-enters on its own with no code change.
        has_citation_signal = cit_spread > 0
        has_recency_signal = age_spread > 0

        w_sem = weights.get("semantic", SEMANTIC_WEIGHT)
        w_cit = weights.get("citation", TRADEOFF_BUDGET / 2) if has_citation_signal else 0.0
        w_rec = weights.get("recency", TRADEOFF_BUDGET / 2) if has_recency_signal else 0.0

        active_total = w_sem + w_cit + w_rec
        if active_total > 0:
            scale = 1.0 / active_total
            w_sem, w_cit, w_rec = w_sem * scale, w_cit * scale, w_rec * scale
        if not (has_citation_signal and has_recency_signal):
            logger.info(
                f"Degenerate signals dropped (citation={has_citation_signal}, "
                f"recency={has_recency_signal}) — weights rescaled to "
                f"sem={w_sem:.2f} cit={w_cit:.2f} rec={w_rec:.2f}"
            )

        def _norm_citation(p: PaperData) -> float:
            if not has_citation_signal:
                return 0.0
            return ((p.citation_count or 0) - cit_lo) / cit_spread

        def _norm_recency(age: Optional[float]) -> float:
            """1.0 = newest in this pool, 0.0 = oldest. Unknown dates rank last."""
            if not has_recency_signal or age is None:
                return 0.0
            return (age_hi - age) / age_spread

        recency_by_id = {p.arxiv_id: _norm_recency(a) for p, a in zip(papers, ages)}

        # Penalty references: what the user already has, and what they rejected.
        saved_embs = None
        try:
            saved_embs = self._get_saved_paper_embeddings()
        except Exception:
            pass

        disliked_embs = self._get_disliked_embeddings()

        # Candidate embeddings are needed by either penalty, so compute once.
        paper_embs_norm_map: Dict[str, np.ndarray] = {}
        if saved_embs is not None or disliked_embs is not None:
            texts = [f"{p.title} {(p.abstract or '')[:300]}" for p in papers]
            raw_embs = np.array(self.embedding_manager.generate_embeddings(texts))
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = raw_embs / norms
            for p, emb in zip(papers, normed):
                paper_embs_norm_map[p.arxiv_id] = emb

        scored = []
        for p in papers:
            sem = sem_scores.get(p.arxiv_id, 0.0)
            cit = _norm_citation(p)
            rec = recency_by_id.get(p.arxiv_id, 0.0)
            base_score = w_sem * sem + w_cit * cit + w_rec * rec

            emb = paper_embs_norm_map.get(p.arxiv_id)
            if emb is not None:
                # Already-covered ground: demote, don't exclude.
                if saved_embs is not None:
                    familiar = self._familiar_penalty(emb, saved_embs)
                    base_score *= (1.0 - FAMILIAR_PENALTY_STRENGTH * familiar)

                # Explicitly rejected ground: demote harder, since this is a
                # stated preference rather than an inference about coverage.
                if disliked_embs is not None:
                    disliked = self._dislike_penalty(emb, disliked_embs)
                    base_score *= (1.0 - DISLIKE_PENALTY_STRENGTH * disliked)

            scored.append((p, base_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        limit = top_k if top_k is not None else settings.TOP_PAPERS_COUNT
        final = self._reserve_exploration_slots(scored, limit)

        result = []
        for p, score in final:
            p.relevance_score = round(score, 4)
            result.append(p)
        return result

    def _reserve_exploration_slots(self, scored: List, limit: int) -> List:
        """
        Give up the last slot or two of the result to papers from outside the top.

        Distinct from _apply_exploration, which widens a candidate pool for MMR to
        choose from. There is no MMR on this path, so appending past the cut would
        do nothing — the slots have to be taken from the winners to have any effect.

        Exploration only ran on the LightGBM path before this, which left new users
        with none of it: the readers whose profile the ranker is least sure about
        were the ones never shown anything that could correct it.
        """
        if len(scored) <= limit:
            return scored[:limit]

        explore_count = min(
            max(1, int(limit * settings.EXPLORATION_RATE)),
            max(0, limit - 1),  # never crowd out the whole result
        )
        if explore_count == 0:
            return scored[:limit]

        keep = scored[:limit - explore_count]
        tail = scored[limit - explore_count:]
        sampled = self._daily_rng().sample(tail, min(explore_count, len(tail)))

        logger.info(
            f"Exploration: {len(keep)} top-ranked + {len(sampled)} sampled from "
            f"{len(tail)} lower-ranked candidates"
        )
        return keep + sampled

    def _rank_with_interests(self, papers: List[PaperData], selected_interests: List[str], top_k: int) -> List[PaperData]:
        """
        Rank papers for the Latest-mode and fallback paths.

        This once hard-coded `0.6 * similarity + 0.4 * citations`, a second copy of
        the main formula, so these two paths silently missed every ranking
        improvement made to the other one. It delegates now, which is what keeps
        the novelty dial and both penalties applying everywhere.
        """
        if not papers:
            return []

        interests_text = " ".join(_expand_focus_areas(selected_interests))
        novelty = self._infer_novelty_preference()
        weights = _blend_weights(novelty)
        logger.info(f"Ranking with novelty preference {novelty:.2f} (weights={weights})")

        return self._score_and_rank(
            papers, interests_text, weights, top_k=top_k
        )

    def _save_papers_to_db(self, papers: List[PaperData]) -> None:
        """Upsert papers from Semantic Scholar into PostgreSQL."""
        existing_ids = {
            r[0] for r in self.db.query(Paper.arxiv_id)
            .filter(Paper.arxiv_id.in_([p.arxiv_id for p in papers])).all()
        }
        for p in papers:
            if p.arxiv_id not in existing_ids:
                self.db.add(Paper(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    authors=", ".join(p.authors),
                    abstract=p.abstract,
                    categories=", ".join(p.categories),
                    published_date=p.published_date,
                    arxiv_url=p.arxiv_url,
                    pdf_url=p.pdf_url,
                    citation_count=p.citation_count,
                ))
        self.db.commit()

    
    def record_interaction(self, item_type: str, item_id: int, interaction_type: str):
        """
        Record user interaction (saved/viewed/dismissed)
        Call this from UI when user clicks buttons
        
        Args:
            item_type: "paper" or "article"
            item_id: Database ID of the item
            interaction_type: "saved", "viewed", or "dismissed"
        """
        self.trainer.record_interaction(item_type, item_id, interaction_type)
        logger.info(f"Recorded {interaction_type} for {item_type} {item_id}")
    
    
    
    
    
    
    def _filter_papers_with_quality(
        self,
        papers: List,
        selected_interests: List[str],
        r_min: float = None,
        min_papers_target: int = 10,
    ) -> List:
        """
        Quality hard filter for papers:
        (relevance >= r_min) AND ((impact_score >= i_min) OR has_doi/journal_ref)
        
        Auto-relaxes thresholds if not enough papers pass.
        """
        if not papers:
            return []
        
        interests_text = " ".join(selected_interests)
        r_threshold = r_min if r_min is not None else settings.MIN_SIMILARITY_THRESHOLD
        i_min = 0.3  # Minimum impact score threshold
        
        # Step 1: Calculate relevance and impact scores for all papers
        paper_data = []
        seen_titles = set()
        
        for paper in papers:
            title_key = getattr(paper, "title", "").strip().lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            
            # Calculate relevance
            paper_text = f"{paper.title} {paper.abstract}"
            relevance = self.embedding_manager.get_similarity_score(interests_text, paper_text)
            
            # Calculate impact score (heuristic)
            impact_score = self.recommender.calculate_impact_score(paper)
            
            # Check for DOI/journal_ref
            has_venue_metadata = bool(getattr(paper, 'doi', None) or getattr(paper, 'journal_ref', None))
            
            paper_data.append({
                'paper': paper,
                'relevance': relevance,
                'impact_score': impact_score,
                'has_venue_metadata': has_venue_metadata,
            })
        
        # Step 2: Apply quality hard filter
        # quality_pass = (impact_score >= i_min) OR has_venue_metadata
        # Keep if: (relevance >= r_threshold) AND quality_pass
        filtered = []
        for data in paper_data:
            quality_pass = (data['impact_score'] >= i_min) or data['has_venue_metadata']
            relevance_pass = data['relevance'] >= r_threshold
            
            if relevance_pass and quality_pass:
                filtered.append(data['paper'])
        
        logger.info(f"Initial filter: {len(filtered)}/{len(paper_data)} papers passed quality hard filter")
        
        # Step 3: Auto-relax if not enough papers
        if len(filtered) < min_papers_target and len(paper_data) > 0:
            logger.info(f"Only {len(filtered)} papers passed. Auto-relaxing thresholds...")
            
            # Strategy 1: Take top X% by relevance
            # Sort by relevance descending
            paper_data.sort(key=lambda x: x['relevance'], reverse=True)
            top_pct = 0.3  # Top 30% by relevance
            top_count = max(min_papers_target, int(len(paper_data) * top_pct))
            
            filtered = []
            for data in paper_data[:top_count]:
                quality_pass = (data['impact_score'] >= i_min * 0.7) or data['has_venue_metadata']  # Lower i_min slightly
                if quality_pass:
                    filtered.append(data['paper'])
            
            # Strategy 2: If still not enough, lower i_min further
            if len(filtered) < min_papers_target:
                logger.info(f"Still only {len(filtered)} papers. Lowering impact threshold...")
                filtered = []
                relaxed_i_min = i_min * 0.5  # Much lower threshold
                for data in paper_data[:top_count]:
                    quality_pass = (data['impact_score'] >= relaxed_i_min) or data['has_venue_metadata']
                    if quality_pass:
                        filtered.append(data['paper'])
            
            # Strategy 3: If still not enough, just take top by relevance (no quality filter)
            if len(filtered) < min_papers_target:
                logger.info(f"Taking top {min_papers_target} by relevance (quality filter disabled)")
                filtered = [data['paper'] for data in paper_data[:min_papers_target]]
            
            logger.info(f"After relaxation: {len(filtered)} papers selected")
        
        # Log stats
        if paper_data:
            relevances = [d['relevance'] for d in paper_data]
            impacts = [d['impact_score'] for d in paper_data]
            logger.info(f"Stats - Relevance: max={max(relevances):.3f}, avg={sum(relevances)/len(relevances):.3f}")
            logger.info(f"Stats - Impact: max={max(impacts):.3f}, avg={sum(impacts)/len(impacts):.3f}")
            venue_metadata_count = sum(1 for d in paper_data if d['has_venue_metadata'])
            logger.info(f"Papers with DOI/journal_ref: {venue_metadata_count}/{len(paper_data)}")
        
        return filtered
    
    def _filter_articles(self, articles: List, selected_interests: List[str], min_threshold: float = None) -> List:
        """Simple relevance filter for articles. Sets relevance_score on each passing item."""
        filtered = []
        interests_text = " ".join(selected_interests)
        threshold = min_threshold if min_threshold is not None else settings.MIN_SIMILARITY_THRESHOLD
        seen_titles = set()
        similarities = []

        for item in articles:
            item_text = f"{item.title} {item.content if hasattr(item, 'content') else ''}"
            similarity = self.embedding_manager.get_similarity_score(interests_text, item_text)
            similarities.append(similarity)
            item.relevance_score = similarity  # store for downstream sorting

            if similarity >= threshold:
                title_key = getattr(item, "title", "").strip().lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                filtered.append(item)

        if similarities:
            logger.info(f"Article similarity stats: max={max(similarities):.3f}, avg={sum(similarities)/len(similarities):.3f}, threshold={threshold}")
        logger.info(f"Filtered {len(filtered)}/{len(articles)} articles above threshold {threshold}")
        return filtered
    
    def _rank_and_select(self, items: List, top_k: int, item_type: str, selected_interests: List[str], use_ml: bool) -> List:
        """Rank items using ML Recommender (learns from your interactions)"""
        if not items:
            return []
        
        # Extract features for each item
        features = []
        recent_texts = self._get_recent_item_texts(item_type)
        for item in items:
            feat = self.feature_extractor.extract_features(
                item, item_type, self.embedding_manager, selected_interests,
                recent_texts=recent_texts,
            )
            features.append(feat)

        if use_ml:
            # Use ML recommender
            ranked = self.recommender.rank_items(items, features)
        else:
            # Heuristic-only scoring (pre-training)
            scored = []
            for item, feat in zip(items, features):
                if item_type == "paper":
                    score = self.recommender.calculate_impact_score(item)
                else:
                    score = feat.get("impact", 0.0)
                scored.append((item, score))
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        # Select top K (with optional exploration + diversity)
        candidates = ranked
        if settings.EXPLORATION_RATE > 0 and len(ranked) > top_k:
            candidates = self._apply_exploration(ranked, top_k)

        selected = candidates
        if settings.USE_MMR_DIVERSITY and len(candidates) > top_k:
            selected = self._apply_mmr(candidates, top_k, item_type)

        # Store relevance scores
        for item, score in selected:
            item.relevance_score = score

        return [item for item, score in selected]

    def _get_recent_item_texts(self, item_type: str) -> List[str]:
        """Get recent recommended item texts for novelty scoring."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.NOVELTY_LOOKBACK_DAYS)
        texts = []
        if item_type == "paper":
            items = (
                self.db.query(Paper)
                .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
                .filter(
                    UserPaperRecommendation.user_id == self.user_id,
                    UserPaperRecommendation.recommended_date >= cutoff,
                )
                .order_by(UserPaperRecommendation.recommended_date.desc())
                .limit(settings.NOVELTY_MAX_ITEMS)
                .all()
            )
            for item in items:
                texts.append(f"{item.title} {item.abstract or ''}")
        else:
            items = (
                self.db.query(Article)
                .join(UserArticleRecommendation, Article.id == UserArticleRecommendation.article_id)
                .filter(
                    UserArticleRecommendation.user_id == self.user_id,
                    UserArticleRecommendation.recommended_date >= cutoff,
                )
                .order_by(UserArticleRecommendation.recommended_date.desc())
                .limit(settings.NOVELTY_MAX_ITEMS)
                .all()
            )
            for item in items:
                texts.append(f"{item.title} {item.content or ''}")
        return texts

    def _apply_mmr(self, ranked_items: List, top_k: int, item_type: str) -> List:
        """Apply Maximal Marginal Relevance (MMR) to diversify results."""
        if not ranked_items:
            return []
        candidate_limit = min(len(ranked_items), max(top_k, top_k * settings.MMR_CANDIDATE_MULTIPLIER))
        candidates = ranked_items[:candidate_limit]

        texts = []
        for item, _score in candidates:
            if item_type == "paper":
                text = f"{item.title} {getattr(item, 'abstract', '')}"
            else:
                text = f"{item.title} {getattr(item, 'content', '')[:2000]}"
            texts.append(text)

        embeddings = np.array(self.embedding_manager.generate_embeddings(texts))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm = embeddings / norms

        scores = np.array([score for _item, score in candidates], dtype=float)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        selected_indices = []
        available = set(range(len(candidates)))

        while len(selected_indices) < min(top_k, len(candidates)) and available:
            if not selected_indices:
                idx = int(np.argmax(scores))
                selected_indices.append(idx)
                available.remove(idx)
                continue

            mmr_scores = []
            for idx in list(available):
                candidate_vec = emb_norm[idx]
                max_sim = 0.0
                for sel_idx in selected_indices:
                    sim = float(np.dot(candidate_vec, emb_norm[sel_idx]))
                    if sim > max_sim:
                        max_sim = sim
                mmr_score = settings.MMR_LAMBDA * scores[idx] - (1.0 - settings.MMR_LAMBDA) * max_sim
                mmr_scores.append((idx, mmr_score))

            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            available.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _daily_rng(self) -> random.Random:
        """
        Randomness seeded on (user, date).

        Exploration needs to be random across days but stable within one, or it
        would contradict the idempotent feed: two clicks on the same day would
        surface different papers even though the feed is meant to be settled.
        Seeding this way keeps both — and keeps a result reproducible when asking
        why a particular paper appeared.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return random.Random(f"{self.user_id}:{today}")

    def _apply_exploration(self, ranked_items: List, top_k: int) -> List:
        """
        Widen the candidate set with a few lower-ranked items.

        Pure exploitation locks a reader inside whatever the ranker already
        believes about them; nothing outside the top slice ever gets a chance to
        prove otherwise. Sampled from the tail rather than the near-miss band so
        the picks are genuinely different, not near-duplicates of the top.
        """
        if not ranked_items:
            return []
        explore_count = max(1, int(top_k * settings.EXPLORATION_RATE))
        candidate_limit = min(
            len(ranked_items),
            max(top_k * settings.MMR_CANDIDATE_MULTIPLIER, top_k)
        )
        candidates = ranked_items[:candidate_limit]
        tail = ranked_items[candidate_limit:]
        if tail:
            sampled = self._daily_rng().sample(tail, min(explore_count, len(tail)))
            candidates.extend(sampled)
        return candidates
    
    # _discover_articles() was removed — trending discovery now lives in
    # the What's Hot section (HotNewsCollector: HuggingFace + GitHub, free APIs,
    # Redis-cached). The agents' search_web tool calls Tavily directly.

    def _generate_summaries(self, items: List, item_type: str) -> List:
        """Generate personalized summaries for items (parallel API calls)."""
        if not items:
            return items

        user_interests = getattr(self, '_user_interests', None)

        def summarize(item):
            content = item.abstract if item_type == "paper" else (
                item.content[:1000] if hasattr(item, 'content') else ""
            )
            item.personalized_summary = self.generator.generate_summary(
                title=item.title,
                content=content,
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
    
    def _store_results(self, papers: List[PaperData], articles: List):
        """Store recommended items in database and vector DB"""
        now = datetime.now(timezone.utc)

        # Clear this user's previous feed (same transaction as inserts below)
        self.db.query(UserPaperRecommendation).filter(
            UserPaperRecommendation.user_id == self.user_id
        ).delete(synchronize_session=False)
        self.db.query(UserArticleRecommendation).filter(
            UserArticleRecommendation.user_id == self.user_id
        ).delete(synchronize_session=False)

        for rank, paper_data in enumerate(papers, 1):
            paper = self.db.query(Paper).filter(Paper.arxiv_id == paper_data.arxiv_id).first()
            if paper:
                # Update shared paper fields (citation count, impact score)
                if paper_data.citation_count is not None:
                    paper.citation_count = paper_data.citation_count
                if getattr(paper_data, "heuristic_impact_score", None) is not None:
                    paper.heuristic_impact_score = paper_data.heuristic_impact_score

                # Per-user recommendation row
                self.db.add(UserPaperRecommendation(
                    user_id=self.user_id,
                    paper_id=paper.id,
                    recommended_date=now,
                    personalized_summary=paper_data.personalized_summary,
                    relevance_score=paper_data.relevance_score,
                    rank=rank,
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

        for rank, article_data in enumerate(articles, 1):
            article = self.db.query(Article).filter(Article.url == article_data.url).first()
            if article:
                self.db.add(UserArticleRecommendation(
                    user_id=self.user_id,
                    article_id=article.id,
                    recommended_date=now,
                    personalized_summary=article_data.personalized_summary,
                    relevance_score=article_data.relevance_score,
                    rank=rank,
                ))

                try:
                    self.embedding_manager.add_article(
                        article_data.source_id,
                        article_data.title,
                        article_data.content,
                        {
                            "title": article_data.title,
                            "url": article_data.url,
                            "source": article_data.source,
                            "published_date": str(article_data.published_date) if article_data.published_date else ""
                        }
                    )
                except Exception as e:
                    logger.debug(f"Article already in vector DB or error: {e}")

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
            rc = self._get_redis()
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

    def _format_output(self, papers: List[PaperData], articles: List) -> Dict:
        """Format output for display"""
        output = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "papers": [],
            "articles": []
        }
        
        for i, paper in enumerate(papers, 1):
            # Get database ID for interaction tracking
            db_paper = self.db.query(Paper).filter(Paper.arxiv_id == paper.arxiv_id).first()
            db_id = db_paper.id if db_paper else None
            
            # Use citation from paper data or database, default to "—" if unavailable
            citation_count = paper.citation_count
            heuristic_impact = getattr(paper, "heuristic_impact_score", None)
            if db_paper:
                if citation_count is None:
                    citation_count = db_paper.citation_count
                if heuristic_impact is None:
                    heuristic_impact = db_paper.heuristic_impact_score
            citation_display = citation_count if citation_count else None
            impact_display = f"{heuristic_impact:.2f}" if heuristic_impact is not None else None
            
            output["papers"].append({
                "rank": i,
                "title": paper.title,
                "arxiv_id": paper.arxiv_id,
                "url": paper.arxiv_url,
                "citation_count": citation_display,
                "impact_score": impact_display,
                "summary": paper.personalized_summary,
                "relevance_score": paper.relevance_score,
                "db_id": db_id  # For interaction tracking
            })
        
        for i, article in enumerate(articles, 1):
            # Get database ID for interaction tracking
            db_id = getattr(article, 'db_id', None)
            if db_id is None:
                db_article = self.db.query(Article).filter(Article.url == article.url).first()
                db_id = db_article.id if db_article else None
            
            output["articles"].append({
                "rank": i,
                "title": article.title,
                "url": article.url,
                "source": article.source,
                "upvotes": article.upvotes,
                "summary": article.personalized_summary,
                "relevance_score": article.relevance_score,
                "db_id": db_id  # For interaction tracking
            })
        
        return output
    
    def format_for_display(self, output: Dict) -> str:
        """Format output as a readable string"""
        lines = []
        lines.append(f"\n📚 Your Daily ML Reading ({output['date']})\n")
        lines.append("=" * 60)
        
        if output["papers"]:
            lines.append(f"\nRESEARCH PAPERS ({len(output['papers'])}):")
            for paper in output["papers"]:
                lines.append(f"\n{paper['rank']}. \"{paper['title']}\" [arXiv:{paper['arxiv_id']}]")
                if paper.get("citation_count"):
                    lines.append(f"   ⭐ {paper['citation_count']} citations")
                elif paper.get("impact_score"):
                    lines.append(f"   📊 Impact score: {paper['impact_score']}")
                summary = paper.get('summary') or "Summary not available"
                lines.append(f"   💡 {summary}")
                lines.append(f"   🔗 {paper['url']}")
        
        if output["articles"]:
            lines.append(f"\n\nTECH ARTICLES ({len(output['articles'])}):")
            for article in output["articles"]:
                lines.append(f"\n{article['rank']}. \"{article['title']}\"")
                lines.append(f"   🔥 {article['upvotes']} upvotes | Source: {article['source']}")
                summary = article.get('summary') or "Summary not available"
                lines.append(f"   💡 {summary}")
                lines.append(f"   🔗 {article['url']}")
        
        # Estimate reading time
        total_items = len(output["papers"]) + len(output["articles"])
        reading_time = total_items * 10  # ~10 minutes per item
        lines.append(f"\n\n⏱️  Estimated reading time: {reading_time} minutes")
        
        return "\n".join(lines)


if __name__ == "__main__":
    db = SessionLocal()
    try:
        pipeline = DailyFeedPipeline(user_id=1, db_session=db)
        result = pipeline.run()
        print(pipeline.format_for_display(result))
    finally:
        db.close()
