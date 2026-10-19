"""
Ranking for the paper feed: the heuristic score, the novelty dial it is
weighted by, familiarity and dislike penalties, category calibration, and
the LightGBM / MMR / exploration selection steps.
"""
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from src.collectors import PaperData
from src.pipelines.feed.channels import normalise
from src.database.models import Paper, UserInteraction, UserModelState, UserPaperRecommendation
from src.pipelines.feed.params import (
    CALIBRATION_EVIDENCE_MAX_IDS,
    CALIBRATION_MIN_QUALITY,
    CALIBRATION_PRIOR_STRENGTH,
    DISLIKE_PENALTY_STRENGTH,
    FAMILIAR_PENALTY_STRENGTH,
    FOCUS_AREA_CATEGORY_MAP,
    LIGHTGBM_MIN_INTERACTIONS,
    SEMANTIC_WEIGHT,
    TRADEOFF_BUDGET,
    _blend_weights,
    _expand_focus_areas,
)
from src.utils.config import settings

logger = logging.getLogger(__name__)


def _age_days(paper: PaperData, now: datetime) -> float:
    published = paper.published_date
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return float(max(0, (now - published).days))


def _scaled_within(values: List[Optional[float]]) -> Optional[List[float]]:
    """
    Each value scaled to 0..1 against the others in this list (None becomes 0), or None
    when the known values are all equal and so cannot separate the items.
    """
    known = [v for v in values if v is not None]
    if not known or max(known) == min(known):
        return None
    low, spread = min(known), max(known) - min(known)
    return [(v - low) / spread if v is not None else 0.0 for v in values]


class ScoringMixin:
    """
    Part of DailyFeedPipeline; relies on the attributes its __init__ sets
    (user_id, db, embedding_manager, generator, feature_extractor, recommender,
    trainer and the per-run memo fields).
    """

    def _compute_target_category_distribution(
        self, selected_interests: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Blend declared interests with saved/viewed papers' arXiv categories into
        a target mix for _calibrate_top_k(), Dirichlet-style: declared interests
        are a prior (CALIBRATION_PRIOR_STRENGTH "virtual" saved papers, split
        across whichever declared areas map to a known category), and real
        saved/viewed papers are the evidence. With nothing saved yet, the result
        is just the declared-interest split; as saves accumulate, evidence
        dominates the prior on its own — there is no hand-picked blend ratio to
        justify or re-tune later.

        Returns None when there's nothing to calibrate against (no declared
        interest maps to a known category and nothing has been saved yet), so
        the caller can skip calibration entirely rather than calibrate toward
        an empty target.
        """
        counts: Dict[str, float] = {}

        mapped_categories = [
            FOCUS_AREA_CATEGORY_MAP[a.upper()]
            for a in selected_interests
            if a.upper() in FOCUS_AREA_CATEGORY_MAP
        ]
        if mapped_categories:
            share = CALIBRATION_PRIOR_STRENGTH / len(mapped_categories)
            for cat in mapped_categories:
                counts[cat] = counts.get(cat, 0.0) + share

        saved_ids = self._get_user_saved_arxiv_ids(max_ids=CALIBRATION_EVIDENCE_MAX_IDS)
        if saved_ids:
            rows = (
                self.db.query(Paper.categories)
                .filter(Paper.arxiv_id.in_(saved_ids))
                .all()
            )
            for (categories_str,) in rows:
                if not categories_str:
                    continue
                # First-listed category only, matching _calibrate_top_k's
                # _primary_category — crediting every category on a multi-labeled
                # paper would count it once per label there but only ever bucket
                # candidates by their first label, silently mismatched.
                primary = categories_str.split(", ")[0].strip()
                if primary:
                    counts[primary] = counts.get(primary, 0.0) + 1.0

        total = sum(counts.values())
        if total <= 0:
            return None
        return {cat: n / total for cat, n in counts.items()}

    def _calibrate_top_k(
        self, ranked: List[PaperData], limit: int, target_dist: Dict[str, float]
    ) -> List[PaperData]:
        """
        Trim an already-ranked (oversampled) list down to `limit` so its
        category mix reflects target_dist — Steck, "Calibrated Recommendations"
        (RecSys 2018) — instead of whatever plain top-K truncation happens to
        produce, which lets one category with systematically higher scores
        (e.g. whatever the profile currently leans toward) crowd out every
        other declared interest.

        Two passes:
          1. Convert target_dist into an integer quota per category (largest-
             remainder rounding, so quotas sum to exactly `limit`), then fill
             each category's quota from its own best-scoring candidates —
             skipping any below CALIBRATION_MIN_QUALITY so an under-populated
             category can't be padded with near-irrelevant papers.
          2. Anything left unfilled (a category didn't have enough qualifying
             candidates to meet its quota) is topped up with the next-best
             remaining candidates overall, regardless of category — a
             shortfall in one category becomes extra room for whichever
             category actually has the depth, rather than forcing in a weak
             paper just to hit an exact ratio.

        A linear "relevance vs. divergence" tradeoff was tried first and
        rejected: at TOP_PAPERS_COUNT's list sizes there's no single weight
        that both reaches the target ratio and refuses to trade a strong paper
        for a near-zero-relevance one — the two goals pulled in opposite
        directions at any weight tested. Explicit quotas plus a quality floor
        get both without a fragile constant standing in for either.
        """
        if len(ranked) <= limit:
            return ranked[:limit]

        scores = [p.relevance_score or 0.0 for p in ranked]
        lo, hi = min(scores), max(scores)
        spread = hi - lo

        def _norm(score: float) -> float:
            return (score - lo) / spread if spread > 0 else 1.0

        def _primary_category(paper: PaperData) -> str:
            cats = getattr(paper, "categories", None) or []
            return cats[0] if cats else "unknown"

        # ── Quotas: largest-remainder rounding so they sum to exactly `limit` ──
        raw_targets = {cat: p * limit for cat, p in target_dist.items()}
        quotas = {cat: int(v) for cat, v in raw_targets.items()}
        leftover_slots = limit - sum(quotas.values())
        by_remainder = sorted(
            raw_targets.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True
        )
        for cat, _ in by_remainder[:leftover_slots]:
            quotas[cat] += 1

        by_category: Dict[str, List[PaperData]] = {}
        for p in ranked:  # already sorted best-first, so each bucket stays sorted
            by_category.setdefault(_primary_category(p), []).append(p)

        selected: List[PaperData] = []
        selected_ids = set()

        # ── Pass 1: each category's own best candidates, up to its quota ──────
        for cat, quota in quotas.items():
            qualifying = [
                p for p in by_category.get(cat, [])
                if _norm(p.relevance_score or 0.0) >= CALIBRATION_MIN_QUALITY
            ]
            for p in qualifying[:quota]:
                selected.append(p)
                selected_ids.add(id(p))

        # ── Pass 2: top up any shortfall with the next-best candidates overall ─
        if len(selected) < limit:
            for p in ranked:
                if len(selected) >= limit:
                    break
                if id(p) not in selected_ids:
                    selected.append(p)
                    selected_ids.add(id(p))

        # Composition is now calibrated; display order still goes by relevance.
        selected.sort(key=lambda p: p.relevance_score or 0.0, reverse=True)
        selected = selected[:limit]

        picked_counts: Dict[str, int] = {}
        for p in selected:
            cat = _primary_category(p)
            picked_counts[cat] = picked_counts.get(cat, 0) + 1
        logger.info(
            f"Calibrated selection: target={ {k: round(v, 2) for k, v in target_dist.items()} }, "
            f"quotas={quotas}, picked={picked_counts}"
        )

        # A category with a real target share but zero final representation means
        # CALIBRATION_OVERSAMPLE_FACTOR wasn't wide enough for this request — the
        # category's candidates never reached this function at all (they were cut
        # by the ranking step's own top-K before calibration ran), as distinct
        # from correctly finding no qualifying candidates within a wide-enough
        # window. This is a heuristic window, not a guarantee; logging it is how
        # that gap gets noticed instead of silently reverting to plain top-K.
        starved = [cat for cat, share in target_dist.items() if share > 0 and picked_counts.get(cat, 0) == 0]
        if starved:
            logger.warning(
                f"Calibration target category(ies) {starved} got zero representation — "
                f"likely cut by ranking's own top-K before reaching calibration, not a "
                f"lack of qualifying candidates. Consider raising CALIBRATION_OVERSAMPLE_FACTOR."
            )
        return selected

    def _use_trained_ranker(self) -> bool:
        """
        Whether this user's trained ranker should order the feed.

        Having enough labels used to be the only condition, so a model that ranked
        held-out feedback worse than the heuristic, or was never validated at all,
        still replaced it. It now has to have beaten the heuristic on the held-out
        split of the user's own feedback (UserModelTrainer._train_with_validation).
        """
        if self._trained_ranker_ok is not None:
            return self._trained_ranker_ok
        ok = False
        if self._allow_ml and self.trainer.get_interaction_count() >= LIGHTGBM_MIN_INTERACTIONS:
            state = self.db.query(UserModelState).filter(UserModelState.user_id == self.user_id).first()
            validated = (state is not None and state.is_trained and state.val_ndcg is not None
                         and state.val_baseline_ndcg is not None)
            ok = validated and state.val_ndcg > state.val_baseline_ndcg
            if not ok:
                self._ranking_fallback_reason = ("model_below_heuristic_on_validation" if validated
                                                 else "model_not_validated")
        self._trained_ranker_ok = ok
        return ok

    # ── Shared Redis handle ───────────────────────────────────────────────────

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

    # ── Weak dismissal-topic penalty ──────────────────────────────────────────

    def _get_disliked_embeddings(self) -> Optional[np.ndarray]:
        """
        Embed recently dismissed papers as a weak topic signal. A click says
        nothing about *why* the paper was dismissed.

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
            papers = (
                self.db.query(Paper)
                .join(UserInteraction, UserInteraction.item_id == Paper.id)
                .filter(UserInteraction.user_id == self.user_id,
                        UserInteraction.item_type == "paper",
                        UserInteraction.interaction_type == "dismissed")
                .order_by(UserInteraction.timestamp.desc())
                .limit(10).all()
            )
            if not papers:
                return None
            texts = [f"{p.title} {(p.abstract or '')[:300]}" for p in papers]
            embs = np.array(self.embedding_manager.generate_embeddings(texts))
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            logger.info("Weak dismiss-topic signal active: %s paper(s)", len(texts))
            return embs / norms
        except Exception as e:
            logger.debug(f"_get_disliked_embeddings skipped: {e}")
            return None

    def _dislike_penalty(self, paper_emb: np.ndarray, disliked_embs: Optional[np.ndarray]) -> float:
        """
        Similarity of a candidate paper to papers this user dismissed.
        Returns a value in [0, 1]; higher means closer to something they disliked.
        """
        if disliked_embs is None or len(disliked_embs) == 0:
            return 0.0
        sims = disliked_embs @ paper_emb
        return float(np.max(sims))

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
                then weakly demoted by familiarity and similar dismissed papers

        Weights come from the caller (see _blend_weights); signals that cannot
        separate the candidates in hand are dropped and their weight redistributed.
        LightGBM ranking is a separate path — see _rank_and_select.
        """
        if not papers:
            return []

        now = datetime.now(timezone.utc)
        sem_scores = semantic_scores_override or self._semantic_scores(papers, interests_text)

        # ── Signal normalisation, scaled to this candidate set ────────────────
        # Both signals are normalised against the candidates actually in hand
        # rather than a fixed scale. A fixed scale degenerates whenever the pool
        # is narrow: over a 7-day Latest window every paper sits within 2% of
        # every other on a 365-day recency curve, and none of them have citation
        # data yet, so both terms collapse to near-constants that add nothing to
        # the ordering while still holding weight.
        citation = _scaled_within([float(p.citation_count or 0) for p in papers])
        # Negated age, so the newest paper scales to 1; an unknown date ranks last.
        recency = _scaled_within([-_age_days(p, now) if p.published_date else None for p in papers])

        # ── Drop weight from signals that cannot separate these candidates ────
        # A signal with no spread adds the same constant to every score, so its
        # weight does no ranking work. Redistributing it proportionally to the
        # signals that *can* separate candidates keeps the full budget useful,
        # and does so from the data rather than from a mode flag — Latest mode
        # gets this automatically, and so does any other pool that happens to be
        # uniform. As citation_refresh backfills real counts, the citation term
        # re-enters on its own with no code change.
        w_sem = weights.get("semantic", SEMANTIC_WEIGHT)
        w_cit = weights.get("citation", TRADEOFF_BUDGET / 2) if citation else 0.0
        w_rec = weights.get("recency", TRADEOFF_BUDGET / 2) if recency else 0.0
        total = w_sem + w_cit + w_rec
        if total > 0:
            w_sem, w_cit, w_rec = w_sem / total, w_cit / total, w_rec / total
        if not (citation and recency):
            logger.info(f"Degenerate signals dropped (citation={bool(citation)}, recency={bool(recency)}) — "
                        f"weights rescaled to sem={w_sem:.2f} cit={w_cit:.2f} rec={w_rec:.2f}")
        citation = citation or [0.0] * len(papers)
        recency = recency or [0.0] * len(papers)

        # Penalty references: what the user already has, and what they rejected.
        saved_embs, disliked_embs, paper_embs_norm_map = self._penalty_vectors(papers)

        scored = []
        for p, cit, rec in zip(papers, citation, recency):
            base_score = w_sem * sem_scores.get(p.arxiv_id, 0.0) + w_cit * cit + w_rec * rec

            emb = paper_embs_norm_map.get(p.arxiv_id)
            if emb is not None:
                # Already-covered ground: demote, don't exclude.
                if saved_embs is not None:
                    familiar = self._familiar_penalty(emb, saved_embs)
                    base_score *= (1.0 - FAMILIAR_PENALTY_STRENGTH * familiar)

                # A dismissal has an unknown reason, so use a modest demotion.
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

    def _semantic_scores(self, papers: List[PaperData], interests_text: str) -> Dict[str, float]:
        """Cosine similarity of each paper (title + start of abstract) to the interest text."""
        texts = [interests_text] + [f"{p.title} {(p.abstract or '')[:400]}" for p in papers]
        embeddings = normalise(np.array(self.embedding_manager.generate_embeddings(texts), dtype=float))
        sims = embeddings[1:] @ embeddings[0]
        return {p.arxiv_id: float(sim) for p, sim in zip(papers, sims)}

    def _penalty_vectors(self, papers: List[PaperData]):
        """
        (saved paper vectors, dismissed paper vectors, {arXiv id: candidate vector}): what
        the two penalties compare against. Candidate vectors are only computed when one of
        the penalties has something to compare with.
        """
        saved_embs = None
        try:
            saved_embs = self._get_saved_paper_embeddings()
        except Exception as exc:
            logger.warning("Saved-paper embeddings unavailable, familiarity penalty disabled for this feed: %s", exc)
        disliked_embs = self._get_disliked_embeddings()
        if saved_embs is None and disliked_embs is None:
            return saved_embs, disliked_embs, {}
        texts = [f"{p.title} {(p.abstract or '')[:300]}" for p in papers]
        vectors = normalise(np.array(self.embedding_manager.generate_embeddings(texts), dtype=float))
        return saved_embs, disliked_embs, {p.arxiv_id: v for p, v in zip(papers, vectors)}

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

    
    def _rank_and_select(self, items: List, top_k: int, selected_interests: List[str], use_ml: bool) -> List:
        """Rank items using ML Recommender (learns from your interactions)"""
        if not items:
            return []
        
        # Extract features for each item
        features = []
        recent_texts = self._get_recent_item_texts()
        for item in items:
            feat = self.feature_extractor.extract_features(
                item, self.embedding_manager, selected_interests,
                recent_texts=recent_texts,
            )
            features.append(feat)

        if use_ml:
            # Use ML recommender
            ranked = self.recommender.rank_items(items, features)
            self._ranking_source = getattr(self.recommender, "last_ranking_source", "heuristic")
            self._ranking_fallback_reason = getattr(self.recommender, "last_ranking_fallback_reason", None)
        else:
            # Heuristic-only scoring (pre-training)
            scored = [(item, self.recommender.calculate_impact_score(item)) for item in items]
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        # Select top K (with optional exploration + diversity)
        candidates = ranked
        if settings.EXPLORATION_RATE > 0 and len(ranked) > top_k:
            candidates = self._apply_exploration(ranked, top_k)

        selected = candidates
        if settings.USE_MMR_DIVERSITY and len(candidates) > top_k:
            selected = self._apply_mmr(candidates, top_k)

        # Store relevance scores
        for item, score in selected:
            item.relevance_score = float(score)

        return [item for item, score in selected]

    def _get_recent_item_texts(self) -> List[str]:
        """Texts of papers recently recommended to this user, for novelty scoring."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.NOVELTY_LOOKBACK_DAYS)
        papers = (
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
        return [f"{paper.title} {paper.abstract or ''}" for paper in papers]

    def _apply_mmr(self, ranked_items: List, top_k: int) -> List:
        """Apply Maximal Marginal Relevance (MMR) to diversify results."""
        if not ranked_items:
            return []
        candidate_limit = min(len(ranked_items), max(top_k, top_k * settings.MMR_CANDIDATE_MULTIPLIER))
        candidates = ranked_items[:candidate_limit]

        texts = [f"{item.title} {getattr(item, 'abstract', '')}" for item, _score in candidates]

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
