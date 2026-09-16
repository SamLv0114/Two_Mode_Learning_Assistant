"""
Feature extraction for ranking models.

Papers only — the Article subsystem is retired (superseded by What's Hot),
nothing produces new UserArticleRecommendation rows anymore, so this no
longer needs to branch on item_type.
"""
from typing import List, Dict, Optional
from datetime import datetime, timezone
from src.utils.config import settings
from src.models.user_recommender import UserRecommender

# Deliberately NOT citation_refresh.py's 14-day min_age_days — that number
# answers a different question ("has Semantic Scholar had time to index
# this paper at all"), not "has enough time passed for zero citations to
# mean anything." Real citation accumulation takes months: someone has to
# write, review (3-9 months for a typical ML venue), publish, and get
# indexed themselves before this paper's citation count moves. Formal
# bibliometrics citation windows run 1-2+ years; 270 days (~9 months) is
# chosen instead to match the empirical average time-to-first-citation for
# CS papers specifically (Zhu & Yan, "Citation Analysis of Computer Systems
# Papers", arXiv:2301.12663) — shorter than the formal literature standard,
# but grounded in a real number rather than an arbitrary compromise, and
# still leaves most of a daily-refreshed, mostly-recent-papers corpus with
# a non-NaN citation_velocity.
CITATION_DATA_MIN_AGE_DAYS = 270


class FeatureExtractor:

    # Categories central to this platform's DL/LLM/agent focus score highest
    # on the "category" feature; the rest of the tracked ARXIV_CATEGORIES
    # score in the middle; anything untracked scores lowest. Replaces a prior
    # binary cs.LG/cs.AI-only check that gave every other tracked category
    # (including cs.CL and cs.MA, i.e. LLM and agent papers) no credit at all.
    _CORE_CATEGORIES = {"cs.CL", "cs.LG", "cs.AI", "cs.MA", "cs.IR", "cs.RO"}

    @staticmethod
    def _basic_readability(text: str) -> float:
        """Heuristic readability score in [0, 1]; higher means easier to read."""
        if not text:
            return 0.0
        sentences = [s for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        words = [w for w in text.split() if w.strip()]
        if not words:
            return 0.0
        avg_sentence_len = len(words) / max(len(sentences), 1)
        avg_word_len = sum(len(w) for w in words) / len(words)
        score = 1.0 / (1.0 + (avg_sentence_len / 25.0) + (avg_word_len / 6.0))
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _has_code(text: str) -> float:
        if not text:
            return 0.0
        keywords = ["github", "gitlab", "bitbucket", "code available", "open source", "open-source"]
        return 1.0 if any(k in text for k in keywords) else 0.0

    @staticmethod
    def _is_survey_or_tutorial(text: str) -> float:
        if not text:
            return 0.0
        keywords = ["survey", "tutorial", "review", "overview"]
        return 1.0 if any(k in text for k in keywords) else 0.0

    @staticmethod
    def _category_score(categories) -> float:
        """Graded topical match — see _CORE_CATEGORIES above."""
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",") if c.strip()]
        if not categories:
            return 0.2
        if any(c in FeatureExtractor._CORE_CATEGORIES for c in categories):
            return 1.0
        if any(c in settings.ARXIV_CATEGORIES for c in categories):
            return 0.6
        return 0.2

    @staticmethod
    def _to_utc(dt) -> Optional[datetime]:
        if not dt:
            return None
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    @staticmethod
    def _recency_score(published_date) -> float:
        """
        Real decay from published_date. Previously hardcoded to a constant
        0.5 for every paper — a constant feature has zero variance, so
        LightGBM could never learn whether a user favors newer work, even
        though the heuristic ranking path (_score_and_rank in daily_feed.py)
        already used real recency. This brings the two paths in line.
        """
        pub = FeatureExtractor._to_utc(published_date)
        if pub is None:
            return 0.5
        days_old = max(0, (datetime.now(timezone.utc) - pub).days)
        return 1.0 / (1.0 + days_old / 30.0)

    @staticmethod
    def _citation_velocity(citation_count, published_date) -> float:
        """
        Citations per day since publication, replacing the old "venue"
        feature. That feature searched the abstract text for conference
        names, but arXiv abstracts almost never mention venue — acceptance
        is metadata, not prose, and most preprints are posted before
        acceptance anyway — so it rarely matched anything. This uses data
        already on hand (citation_count, published_date) and distinguishes
        a paper accumulating citations quickly from one with a merely large
        but old, stagnant count.

        Real citation accumulation runs on a timescale of months to years
        (someone has to write, review, and publish a paper that cites this
        one), not the 2-4 weeks Semantic Scholar needs to index a new paper.
        A paper younger than CITATION_DATA_MIN_AGE_DAYS returns NaN — not a
        guessed neutral score — because "no citations yet" isn't a real
        signal at that age, it's just missing data. LightGBM has native
        support for missing values and learns how to treat them from the
        data itself, which is more honest than us picking a stand-in score.
        (Note: this relies on the LGBMRanker path specifically — the sklearn
        GradientBoostingRegressor fallback used when USE_LTR is off does NOT
        support NaN natively.)
        """
        pub = FeatureExtractor._to_utc(published_date)
        if pub is None:
            return float("nan")
        days_old = max(1, (datetime.now(timezone.utc) - pub).days)

        if days_old < CITATION_DATA_MIN_AGE_DAYS:
            return float("nan")

        if not citation_count:
            return 0.0

        velocity = citation_count / days_old
        # 0.5 citations/day (~3.5/week) already reads as a strong signal.
        return max(0.0, min(velocity / 0.5, 1.0))

    @staticmethod
    def _novelty_score(text: str, recent_texts: Optional[List[str]], embedding_manager) -> float:
        if not text or not recent_texts:
            return 0.5
        max_sim = 0.0
        for recent in recent_texts[:20]:
            sim = embedding_manager.get_similarity_score(text, recent)
            if sim > max_sim:
                max_sim = sim
        novelty = 1.0 - max_sim
        return max(0.0, min(novelty, 1.0))

    @staticmethod
    def extract_features(
        item,
        embedding_manager,
        user_interests: List[str],
        recent_texts: Optional[List[str]] = None,
    ) -> Dict:
        """Extract ranking features for a paper."""
        interests_text = " ".join(user_interests)

        body_text = getattr(item, "abstract", "") or ""
        full_text = f"{item.title} {body_text}"
        full_text_lower = full_text.lower()

        similarity = embedding_manager.get_similarity_score(interests_text, full_text)
        recency_score = FeatureExtractor._recency_score(getattr(item, "published_date", None))
        impact_score = UserRecommender.calculate_impact_score(item, include_venue=False)
        category_score = FeatureExtractor._category_score(getattr(item, "categories", []))

        title_length = len(getattr(item, "title", ""))
        title_score = 1.0 - min(title_length / 200.0, 0.5)
        content_score = min(len(body_text) / 3000.0, 1.0)

        readability = FeatureExtractor._basic_readability(body_text)
        has_code = FeatureExtractor._has_code(full_text_lower)
        is_survey = FeatureExtractor._is_survey_or_tutorial(full_text_lower)
        novelty = FeatureExtractor._novelty_score(full_text, recent_texts, embedding_manager)
        citation_velocity = FeatureExtractor._citation_velocity(
            getattr(item, "citation_count", None), getattr(item, "published_date", None)
        )

        return {
            "similarity": similarity,
            "recency": recency_score,
            "impact": impact_score,
            "category": category_score,
            "title_length": title_score,
            "content_length": content_score,
            "readability": readability,
            "has_code": has_code,
            "is_survey": is_survey,
            "novelty": novelty,
            "citation_velocity": citation_velocity,
        }
