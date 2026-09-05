"""
MODE 1: Daily Recommendation Feed Pipeline
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


# V4: research-mode-aware ranking weights (inspired by LLM4RS)
MODE_WEIGHTS = {
    "learning":  {"semantic": 0.5, "citation": 0.4, "recency": 0.1},
    "frontier":  {"semantic": 0.4, "citation": 0.1, "recency": 0.5},
    "balanced":  {"semantic": 0.6, "citation": 0.4, "recency": 0.0},
}

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
    """Main pipeline for daily feed generation"""
    
    def __init__(self, user_id: int, db_session: Session, embedding_manager: EmbeddingManager = None):
        self.user_id = user_id
        self.db = db_session
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.generator = Generator()
        self.feature_extractor = FeatureExtractor()
        self.recommender = UserRecommender(user_id, db_session)
        self.trainer = UserModelTrainer(user_id, db_session, self.embedding_manager)
    
    def run(self, time_window_days: int = 7, focus_areas: List[str] = None, user_interests: List[str] = None, mode: str = "recommended"):
        logger.info(f"Starting daily feed pipeline (mode={mode})...")

        selected_interests = focus_areas if focus_areas else settings.USER_INTERESTS
        self._user_interests = _expand_focus_areas(focus_areas) if focus_areas else (user_interests or selected_interests)

        # Step 1: Papers
        # latest      — published_date >= cutoff, ranked by interest match + influence
        # recommended — no date filter, ranked by interest match + influence across all DB papers
        logger.info(f"Step 1: Fetching papers (mode={mode})...")
        if mode == "latest":
            top_papers = self._fetch_papers_latest(selected_interests, days=time_window_days)
        else:
            top_papers = self._fetch_papers_recommended(selected_interests)

        # Step 2: Articles — retired in V4.
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

        # V4: persist feed context in Redis for IterativeRefinementAgent
        self._store_feed_ctx(output, mode)

        # Step 5: Retrain if enough interactions
        interaction_count = self.trainer.get_interaction_count()
        if interaction_count >= 50:
            logger.info(f"Step 5: Retraining model with {interaction_count} interactions...")
            self.trainer.retrain_model(self.recommender, min_interactions=50, use_validation=True)
        else:
            logger.info(f"Step 5: Skipping retrain ({interaction_count}/50 interactions)")

        logger.info("Daily feed pipeline completed!")
        return output

    def _fetch_papers_recommended(self, selected_interests: List[str]) -> List[PaperData]:
        """
        Recommended mode — two-stage retrieval with V4 enhancements:

        Stage 1 (ChromaDB ANN, fan-out parallel):
            V4: _fanout_search() expands the query into 3 variants, runs parallel
            ChromaDB searches, and merges by doc_id. Richer recall than a single query.
            Falls back to single search if expansion fails.

        Stage 2 (PostgreSQL join + mode-aware re-rank):
            V4: Uses _infer_research_mode() + MODE_WEIGHTS to weight semantic/citation/recency.
            V4: Applies _familiar_penalty() to penalise already-seen content.

        Fallback: impact-score DB query if ChromaDB returns too few results.
        """
        interests_text = " ".join(_expand_focus_areas(selected_interests))
        candidate_limit = settings.TOP_PAPERS_COUNT * 20  # e.g. 100

        # V4: infer research mode for downstream re-ranking
        research_mode = self._infer_research_mode()
        self._research_mode = research_mode  # surfaced in feed_ctx for /feed/refine
        weights = MODE_WEIGHTS.get(research_mode, MODE_WEIGHTS["balanced"])
        logger.info(f"Research mode inferred: {research_mode} (weights={weights})")

        # V4: HuggingFace trending for frontier mode
        if research_mode == "frontier":
            hf_papers = self._fetch_hf_trending_papers(candidate_limit)
            if len(hf_papers) >= settings.TOP_PAPERS_COUNT:
                logger.info("Frontier mode: using HuggingFace trending papers")
                # Persist first — _store_results only writes recommendations for
                # papers that already exist in PostgreSQL, so unsaved HF papers
                # would be dropped and the feed would come back empty.
                self._save_papers_to_db(hf_papers)
                return self._rerank_papers_mode_aware(hf_papers, interests_text, weights)

        # V4: LLM-generated user profile as richer ChromaDB query
        profile_query = self._build_user_profile_query(redis_client=self._get_redis()) or interests_text
        logger.info(f"ChromaDB query: {'profile' if profile_query != interests_text else 'keywords'}")

        # ── Stage 1: fan-out parallel ChromaDB search (V4) ───────────────────
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
        if interaction_count >= 50:
            # LightGBM knows citation counts via feature extractor — just pass candidates
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            papers = [self._db_paper_to_paperdata(p) for p in db_papers]
            return self._rank_and_select(
                papers, settings.TOP_PAPERS_COUNT, "paper", selected_interests, use_ml=True
            )

        # ── Re-rank: mode-aware weights + familiar penalty (V4) ──────────────
        paperdata_list = [self._db_paper_to_paperdata(p) for p in db_papers]
        return self._rerank_papers_mode_aware(paperdata_list, interests_text, weights,
                                               semantic_scores_override={p.arxiv_id: semantic_scores.get(p.arxiv_id, 0.0) for p in db_papers})

    def _fetch_papers_recommended_fallback(self, selected_interests: List[str]) -> List[PaperData]:
        """
        Fallback for Recommended mode when ChromaDB has too few papers indexed.
        Uses batch embeddings on top-100 papers by impact score.
        Typically only needed before the backfill script has been run.
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
        if interaction_count >= 50:
            logger.info(f"LightGBM re-ranking ({interaction_count} interactions)")
            return self._rank_and_select(
                papers, settings.TOP_PAPERS_COUNT, "paper", selected_interests, use_ml=True
            )
        else:
            logger.info(f"Interest+influence ranking ({interaction_count}/50 interactions for LightGBM)")
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

    # ── V4: shared Redis handle ───────────────────────────────────────────────

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

    # ── V4: research mode inference ───────────────────────────────────────────

    def _infer_research_mode(self) -> str:
        """
        Infer user's current research mode from interaction history.
        - recent_ratio (fraction of saved/viewed papers < 30 days old) > 0.7 → frontier
        - avg citations of saved papers > 100 → learning
        - else → balanced
        """
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
                return "balanced"

            recent_count = sum(
                1 for _, p in rows
                if p.published_date and p.published_date.replace(tzinfo=timezone.utc) >= cutoff_30d
            )
            recent_ratio = recent_count / len(rows)

            citations = [p.citation_count or 0 for _, p in rows]
            avg_citations = sum(citations) / len(citations)

            if recent_ratio > 0.7:
                return "frontier"
            elif avg_citations > 100:
                return "learning"
            else:
                return "balanced"
        except Exception:
            return "balanced"

    # ── V4: LLM user-profile query (Mem0-inspired) ────────────────────────────

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

    # ── V4: fan-out parallel ChromaDB search (GPT-Researcher-inspired) ────────

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

    # ── V4: familiar-paper penalty (diversity signal) ─────────────────────────

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

    # ── V4: HuggingFace trending papers for frontier mode ─────────────────────

    def _fetch_hf_trending_papers(self, limit: int = 20) -> List[PaperData]:
        """
        Fetch trending ML papers from the HuggingFace papers API.

        publishedAt is parsed into published_date — frontier mode weights recency
        at 0.5, so dropping the date would silently zero out half the ranking signal.
        """
        import urllib.request
        try:
            url = f"https://huggingface.co/api/papers?sort=trending&limit={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": "LearningAssistant/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            papers = []
            for item in data:
                arxiv_id = item.get("id", "")
                if not arxiv_id:
                    continue

                published = None
                raw_date = item.get("publishedAt")
                if raw_date:
                    try:
                        published = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    except Exception:
                        published = None

                papers.append(PaperData(
                    arxiv_id=arxiv_id,
                    title=item.get("title", ""),
                    abstract=item.get("summary", ""),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    categories=[],
                    published_date=published,
                    arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    citation_count=item.get("upvotes", 0),
                ))
            logger.info(f"HuggingFace trending: fetched {len(papers)} papers")
            return papers
        except Exception as e:
            logger.warning(f"HuggingFace trending fetch failed: {e}")
            return []

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

    def _rerank_papers_mode_aware(
        self,
        papers: List[PaperData],
        interests_text: str,
        weights: Dict,
        semantic_scores_override: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> List[PaperData]:
        """
        V4: Score candidates using mode-specific weights for semantic, citation, and recency signals.
        Applies familiar_penalty to reduce score for papers too similar to already-saved content.

        This is the single ranking implementation — every non-ML path routes through
        it so mode inference and the familiar penalty apply consistently.
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

        # Citation normalization
        cit_vals = [p.citation_count or 0 for p in papers]
        max_cit = max(cit_vals) if max(cit_vals) > 0 else 1

        # Recency normalization (0=oldest known, 1=today)
        def _recency(p: PaperData) -> float:
            if not p.published_date:
                return 0.0
            pub = p.published_date
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age = max(0, (now - pub).days)
            return max(0.0, 1.0 - age / 365)

        # V4: familiar penalty embeddings
        saved_embs = None
        try:
            saved_embs = self._get_saved_paper_embeddings()
        except Exception:
            pass

        w_sem = weights.get("semantic", 0.6)
        w_cit = weights.get("citation", 0.4)
        w_rec = weights.get("recency", 0.0)

        # Pre-compute paper embeddings if familiar penalty is needed
        paper_embs_norm_map: Dict[str, np.ndarray] = {}
        if saved_embs is not None:
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
            cit = (p.citation_count or 0) / max_cit
            rec = _recency(p)
            base_score = w_sem * sem + w_cit * cit + w_rec * rec

            # Penalise overly familiar papers (similarity to already-saved work)
            if saved_embs is not None and p.arxiv_id in paper_embs_norm_map:
                penalty = self._familiar_penalty(paper_embs_norm_map[p.arxiv_id], saved_embs)
                base_score = base_score * (1.0 - 0.3 * penalty)

            scored.append((p, base_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        limit = top_k if top_k is not None else settings.TOP_PAPERS_COUNT
        result = []
        for p, score in scored[:limit]:
            p.relevance_score = round(score, 4)
            result.append(p)
        return result

    def _rank_with_interests(self, papers: List[PaperData], selected_interests: List[str], top_k: int) -> List[PaperData]:
        """
        Rank papers for the Latest-mode and fallback paths.

        Previously this hard-coded `0.6 * similarity + 0.4 * citations` — byte-for-byte
        MODE_WEIGHTS["balanced"]. Duplicating the formula meant these two paths silently
        missed every V4 ranking improvement: research-mode inference, the recency signal,
        and the familiar-paper penalty. It now delegates to the single ranking
        implementation so all non-ML paths behave consistently.
        """
        if not papers:
            return []

        interests_text = " ".join(_expand_focus_areas(selected_interests))
        research_mode = self._infer_research_mode()
        weights = MODE_WEIGHTS.get(research_mode, MODE_WEIGHTS["balanced"])
        logger.info(f"Ranking with inferred mode: {research_mode} (weights={weights})")

        return self._rerank_papers_mode_aware(
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

    def _apply_exploration(self, ranked_items: List, top_k: int) -> List:
        """Add a small exploration pool to the candidate set."""
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
            sampled = random.sample(tail, min(explore_count, len(tail)))
            candidates.extend(sampled)
        return candidates
    
    # _discover_articles() was removed in V4 — trending discovery now lives in
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
        V4: Persist feed context in Redis (TTL 2h) for the IterativeRefinementAgent.
        Stores shown paper IDs, research mode, and query info so /feed/refine can
        re-score and adjust the feed without running the full pipeline again.

        research_mode (learning/frontier/balanced) is what the Evaluator prompt
        expects — the feed mode (recommended/latest) is kept separately.
        """
        try:
            rc = self._get_redis()
            if not rc:
                return
            ctx = {
                "user_id": self.user_id,
                "feed_mode": mode,
                "research_mode": getattr(self, "_research_mode", "balanced"),
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
