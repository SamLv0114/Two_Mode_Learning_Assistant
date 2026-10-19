"""
Paper recommendation pipeline.

Entry point is DailyFeedPipeline.run(), triggered by POST /feed/generate. The
user picks one of two modes in the UI; everything after retrieval is shared.

TWO PATHS
  With a research profile (src/agents/research_profile.py), run() takes the
  profile path in feed/profile_feed.py: content, citation-graph and trending
  recall, a pre-rank, the model's per-paper judgements against the profile, and
  a stored "why this paper" for each result. Editing the profile rebuilds the
  day's feed. Everything below describes the earlier path, which users without
  a profile keep, and whose ranking pieces the profile path reuses as its pre-rank.

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
      score *= (1 - 0.08 * dislike)        similar to a dismissed paper

  Weights come from one continuous dial, not discrete modes. Semantic similarity
  is fixed at 0.55 because it is the point of personalisation; the remaining 0.45
  slides between citation authority and recency according to how the user reads
  (_infer_novelty_preference), shrunk toward neutral when evidence is thin.

  Signals are normalised against the candidates in hand, and a signal that cannot
  separate them is dropped with its weight redistributed. This is what makes
  Latest work: inside a 7-day window no paper has citations yet, so that term
  would otherwise hold weight while contributing nothing.

  Once enough distinct paper feedback exists, a trained LightGBM model can
  replace this score in _rank_and_select, but only if it beat this score on the
  held-out split of the user's own feedback (_use_trained_ranker). Prediction
  failures fall back to it.

FEEDBACK
  Saves, views and dismissals all land in UserInteraction. A dismissal is only
  a weak topic signal; the system does not invent a reason for it. New users
  are offered a diverse set of papers to rate up front
  (get_coldstart_seeds) rather than waiting for this to accumulate.

LAYOUT
  This file orchestrates (run, retraining). The steps live in src/pipelines/feed/:
  candidates.py retrieval, scoring.py ranking and selection, output.py
  summaries/storage/payload, params.py the tuning constants they share.

Community trending content is deliberately NOT here — see What's Hot
(src/agents/hot_news_collector.py). This pipeline personalises; that one does not.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from src.database.models import Paper, UserInteraction, UserModelState
from src.models import EmbeddingManager, FeatureExtractor
from src.models.user_recommender import UserRecommender
from src.models.user_trainer import UserModelTrainer
from src.pipelines.feed.candidates import CandidateMixin
from src.pipelines.feed.output import OutputMixin
from src.pipelines.feed.profile_feed import ProfileFeedMixin
from src.pipelines.feed.params import (
    LIGHTGBM_MIN_INTERACTIONS,
    _expand_focus_areas,
)
from src.pipelines.feed.scoring import ScoringMixin
from src.rag import Generator
from src.utils.config import settings
from src.utils.llm_usage import request_budget

logger = logging.getLogger(__name__)


class DailyFeedPipeline(ProfileFeedMixin, CandidateMixin, ScoringMixin, OutputMixin):
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
        self._ranking_source = "heuristic"
        self._ranking_fallback_reason: Optional[str] = None
        self._allow_ml = True
        self._trained_ranker_ok: Optional[bool] = None   # see _use_trained_ranker

    
    def run(
        self,
        time_window_days: int = 7,
        focus_areas: List[str] = None,
        user_interests: List[str] = None,
        mode: str = "recommended",
        force_refresh: bool = False,
        use_ml: bool = True,
    ):
        """
        Build today's feed.

        Idempotent within a day: a second call with the same mode/time_window
        returns the feed already generated rather than producing a new one.
        Generating unconditionally made the button destructive — it deleted the
        list the user was reading, spent LLM calls re-summarising, and, because
        papers shown in the last NOVELTY_LOOKBACK_DAYS are excluded from
        candidates, handed back strictly lower-ranked papers each time. Clicking
        twice should not be a downgrade.

        force_refresh=True is the explicit "show me a different batch" path, where
        consuming the novelty window is what the user actually asked for. A mode
        or time_window_days that differs from what today's stored feed was
        generated with is treated the same way — the UI toggle is an explicit
        signal, and silently ignoring it (returning a Recommended-mode feed after
        the user switched to Latest, say) is worse than spending the regeneration.
        """
        self._allow_ml = use_ml
        profile = self._research_profile()
        if not force_refresh:
            try:
                existing = self._stored_feed()
            except Exception as exc:  # a failed lookup must not block generation
                logger.warning("Could not read today's feed, generating a new one: %s", exc)
                existing = {"papers": []}
            if existing["papers"]:
                existing_mode = existing.get("mode") or "recommended"
                mode_matches = existing_mode == mode
                window_matches = (
                    mode != "latest"
                    or existing.get("time_window_days") == time_window_days
                )
                ranking_matches = use_ml or existing.get("ranking_source") != "ml"
                # An edited research profile is a request for a feed built from it.
                profile_matches = existing.get("profile_version") == (profile["version"] if profile else None)
                if mode_matches and window_matches and ranking_matches and profile_matches:
                    logger.info(
                        f"Feed already generated today ({len(existing['papers'])} papers) "
                        f"— returning it unchanged. Use force_refresh for a new batch."
                    )
                    return {**existing, "reused": True}
                logger.info(
                    f"Today's feed (mode={existing_mode!r}, window={existing.get('time_window_days')}, "
                    f"profile version {existing.get('profile_version')}) does not match this request — regenerating."
                )

        logger.info(f"Starting daily feed pipeline (mode={mode}, force_refresh={force_refresh})...")

        # Train before ranking so the newly trained model serves this batch.
        # A reused batch does not retrain or block on model work.
        if use_ml:
            self._maybe_retrain()

        selected_interests = focus_areas if focus_areas else settings.USER_INTERESTS
        self._user_interests = _expand_focus_areas(focus_areas) if focus_areas else (user_interests or selected_interests)

        # Step 1: Papers. The two modes differ only in how candidates are
        # retrieved; both hand off to the same scoring path afterwards.
        #   latest      — PostgreSQL, published_date >= cutoff
        #   recommended — ChromaDB semantic retrieval over the whole corpus
        logger.info(f"Step 1: Fetching papers (mode={mode})...")
        with request_budget("feed", settings.FEED_TOKEN_BUDGET):
            if profile:
                # The profile path: several recall channels, then the model's judgements
                # against the profile (src/pipelines/feed/profile_feed.py).
                self._user_interests = profile["interests"] + ([profile["goal"]] if profile["goal"] else [])
                top_papers = self._fetch_profile_feed(profile, mode=mode, days=time_window_days)
            elif mode == "latest":
                top_papers = self._fetch_papers_latest(selected_interests, days=time_window_days)
            else:
                top_papers = self._fetch_papers_recommended(selected_interests)

            # Step 2: Summarize papers (parallel LLM calls)
            logger.info("Step 2: Generating personalized summaries...")
            top_papers = self._generate_summaries(top_papers)

        # Step 3: Store and return
        logger.info("Step 3: Storing results...")
        self._store_results(top_papers, mode=mode, time_window_days=time_window_days)
        output = {**self._stored_feed(), "reused": False}

        # Persist feed context in Redis for IterativeRefinementAgent
        self._store_feed_ctx(output, mode)

        logger.info("Daily feed pipeline completed!")
        return output

    def _maybe_retrain(self) -> None:
        """
        Retrain the ranker when enough interactions have accumulated and something changed.

        Called when a new batch is generated; returning an existing batch does not block
        on model training. Failures are logged, never raised: a training problem should not
        fail the request that happened to trigger it.
        """
        try:
            interaction_count = self.trainer.get_interaction_count()
            if interaction_count < LIGHTGBM_MIN_INTERACTIONS:
                logger.info(f"Skipping retrain ({interaction_count}/{LIGHTGBM_MIN_INTERACTIONS} interactions)")
                return
            state = self.db.query(UserModelState).filter(UserModelState.user_id == self.user_id).first()
            now = datetime.now(timezone.utc)
            if not self._retrain_due(state, interaction_count, now):
                return
            if state is None:
                state = UserModelState(user_id=self.user_id)
                self.db.add(state)
            state.last_training_attempt_at = now
            state.interaction_count_at_attempt = interaction_count
            self.db.commit()
            logger.info(f"Retraining model with {interaction_count} interactions...")
            trained = self.trainer.retrain_model(self.recommender, min_interactions=LIGHTGBM_MIN_INTERACTIONS,
                                                 use_validation=True)
            if not trained:
                logger.warning("User %s: Training did not produce a model from %s labels", self.user_id, interaction_count)
        except Exception as e:
            logger.warning(f"Retrain skipped due to error: {e}")

    def _retrain_due(self, state: Optional[UserModelState], interaction_count: int, now: datetime) -> bool:
        """
        Two reasons to skip, both meaning "the same feedback would be fit again":
          - the last attempt (successful or not) already saw this feedback, within the interval
          - a model is trained and no feedback arrived since, or it was trained within the interval
        """
        if state is None:
            return True
        interval = timedelta(hours=settings.AUTO_RETRAIN_INTERVAL_HOURS)

        def new_feedback_since(moment: datetime) -> bool:
            latest = self._latest_feedback_at()
            return latest is not None and latest > moment

        attempted = _as_utc(state.last_training_attempt_at)
        if (attempted and now - attempted < interval
                and interaction_count <= (state.interaction_count_at_attempt or 0)
                and not new_feedback_since(attempted)):
            return False
        trained = _as_utc(state.last_trained_at)
        if (trained and state.is_trained and self.recommender.is_trained
                and (state.training_sample_count or 0) >= LIGHTGBM_MIN_INTERACTIONS):
            if interaction_count <= (state.interaction_count_at_training or 0) and not new_feedback_since(trained):
                return False
            if now - trained < interval:
                return False
        return True

    def _latest_feedback_at(self) -> Optional[datetime]:
        latest = (self.db.query(UserInteraction.timestamp)
                  .join(Paper, Paper.id == UserInteraction.item_id)
                  .filter(UserInteraction.user_id == self.user_id, UserInteraction.item_type == "paper",
                          UserInteraction.interaction_type.in_(("saved", "viewed", "dismissed")))
                  .order_by(UserInteraction.timestamp.desc()).first())
        return _as_utc(latest[0]) if latest else None


def _as_utc(moment: Optional[datetime]) -> Optional[datetime]:
    """Timestamps are stored naive in UTC; make them comparable with aware ones."""
    if moment is None:
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment
