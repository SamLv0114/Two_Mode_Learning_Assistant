"""
MODE 1: Daily Recommendation Feed Pipeline
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import List, Dict
import logging
import numpy as np
import random

from sqlalchemy import exc as sa_exc

from sqlalchemy.orm import Session

from src.collectors import ArxivCollector, PaperData, HNCollector, MediumCollector, DevToCollector
from src.models import EmbeddingManager, FeatureExtractor
from src.models.user_recommender import UserRecommender
from src.models.user_trainer import UserModelTrainer
from src.rag import Generator
from src.database.models import Paper, Article, SessionLocal, init_db, UserInteraction, UserPaperRecommendation, UserArticleRecommendation
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Step 2: Articles — disabled for now, re-enable once papers are stable
        # scale = min(max(1, time_window_days) / 7.0, 52)
        # article_fetch_limit = int(10 * scale)
        # logger.info("Step 2: Collecting and filtering articles...")
        # raw_articles = self._collect_articles(time_window_days, article_fetch_limit)
        # candidate_articles = self._filter_articles(raw_articles, selected_interests, min_threshold=0.15)
        # top_articles = sorted(
        #     candidate_articles,
        #     key=lambda a: (getattr(a, "relevance_score", 0) or 0) * 0.7 + min((getattr(a, "upvotes", 0) or 0) / 500, 1.0) * 0.3,
        #     reverse=True,
        # )[:settings.TOP_ARTICLES_COUNT]
        top_articles = []

        # Step 3: Summarize papers (parallel LLM calls)
        logger.info("Step 3: Generating personalized summaries...")
        top_papers = self._generate_summaries(top_papers, item_type="paper")

        # Step 4: Store and return
        logger.info("Step 4: Storing results...")
        self._store_results(top_papers, top_articles)
        output = self._format_output(top_papers, top_articles)

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
        Recommended mode — two-stage retrieval:

        Stage 1 (ChromaDB ANN, ~10ms):
            Semantic search across ALL indexed papers using HNSW.
            Returns top-N candidates with cosine similarity scores.

        Stage 2 (PostgreSQL join + re-rank):
            Fetch full metadata + citation counts for those candidates.
            Score = 0.6 * semantic_similarity + 0.4 * normalized_citations.

        Fallback: impact-score DB query if ChromaDB returns too few results.
        """
        interests_text = " ".join(_expand_focus_areas(selected_interests))
        candidate_limit = settings.TOP_PAPERS_COUNT * 20  # e.g. 100

        # ── Stage 1: ChromaDB ANN search ──────────────────────────────────────
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

        # Heuristic: 0.6 * semantic + 0.4 * normalized citations
        citations = [p.citation_count or 0 for p in db_papers]
        max_cit = max(citations) if max(citations) > 0 else 1

        scored = []
        for p in db_papers:
            sem = semantic_scores.get(p.arxiv_id, 0.0)
            cit = (p.citation_count or 0) / max_cit
            score = 0.6 * sem + 0.4 * cit
            scored.append((p, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:settings.TOP_PAPERS_COUNT]

        papers = []
        for p, score in top:
            pd = self._db_paper_to_paperdata(p)
            pd.relevance_score = round(score, 4)
            papers.append(pd)

        return papers

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

    def _rank_with_interests(self, papers: List[PaperData], selected_interests: List[str], top_k: int) -> List[PaperData]:
        """
        Rank papers by combining embedding cosine similarity to user interests + normalized citations.
        Uses batch embedding for efficiency (one model.encode call per batch).
        Combined score = 0.6 * similarity + 0.4 * normalized_citation_count
        """
        if not papers:
            return []

        interests_text = " ".join(_expand_focus_areas(selected_interests))
        paper_texts = [f"{p.title} {(p.abstract or '')[:400]}" for p in papers]

        all_texts = [interests_text] + paper_texts
        all_embeddings = np.array(self.embedding_manager.generate_embeddings(all_texts))

        interest_emb = all_embeddings[0]
        paper_embs = all_embeddings[1:]

        interest_norm = interest_emb / (np.linalg.norm(interest_emb) + 1e-8)
        norms = np.linalg.norm(paper_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        paper_embs_norm = paper_embs / norms
        similarities = paper_embs_norm @ interest_norm

        citations = np.array([p.citation_count or 0 for p in papers], dtype=float)
        max_cit = citations.max() if citations.max() > 0 else 1.0
        cit_scores = citations / max_cit

        combined = 0.6 * similarities + 0.4 * cit_scores

        for paper, score in zip(papers, combined.tolist()):
            paper.relevance_score = round(float(score), 4)

        ranked_indices = np.argsort(combined)[::-1][:top_k]
        return [papers[i] for i in ranked_indices]

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

    def _rank_papers_by_signals(self, papers: List[PaperData]) -> List[PaperData]:
        """
        Rank papers without embedding.
        Score = 0.6 * normalized_citations + 0.4 * recency
        """
        now = datetime.now(timezone.utc)
        citations = [p.citation_count or 0 for p in papers]
        max_cit = max(citations) if citations else 1

        def score(p: PaperData) -> float:
            cit_score = (p.citation_count or 0) / max(max_cit, 1)
            if p.published_date:
                age_days = (now - p.published_date.replace(tzinfo=timezone.utc) if p.published_date.tzinfo is None else now - p.published_date).days
                recency = max(0.0, 1.0 - age_days / 365)
            else:
                recency = 0.0
            return 0.6 * cit_score + 0.4 * recency

        ranked = sorted(papers, key=score, reverse=True)
        for p in ranked:
            p.relevance_score = score(p)
        return ranked
    
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
    
    def get_interaction_stats(self) -> dict:
        """Get statistics about collected interactions"""
        count = self.trainer.get_interaction_count()
        return {
            "total_interactions": count,
            "ready_for_retrain": count >= 50,
            "interactions_needed": max(0, 50 - count)
        }
    
    def _collect_papers(self, time_window_days: int, selected_interests: List[str], max_results: int, use_time_slices: bool = True) -> List[PaperData]:
        """Collect new papers from ArXiv"""
        collector = ArxivCollector()
        # Map focus areas to arXiv categories
        focus_to_categories = {
            "NLP": ["cs.CL"],
            "ML": ["cs.LG"],
            "AI": ["cs.AI"],
            "DL": ["cs.LG", "cs.CV", "cs.AI", "cs.NE"],
            "CV": ["cs.CV"],
        }
        categories = []
        for area in selected_interests:
            categories.extend(focus_to_categories.get(area, []))
        # Deduplicate categories
        # if none matched, use defaults
        category_override = list(dict.fromkeys(categories)) if categories else None

        # Use time-slice bucketing to ensure balanced sampling across time window
        if use_time_slices:
            # Calculate number of slices and candidates per slice
            if time_window_days >= 365:
                num_slices = 12  # Monthly slices for 1 year
            elif time_window_days >= 30:
                num_slices = 4  # Weekly slices for 1 month
            elif time_window_days >= 7:
                num_slices = 7  # Daily slices for 1 week
            else:
                num_slices = time_window_days  # Daily slices for < 7 days
            
            candidates_per_slice = min(20, max(5, max_results // num_slices * 2))
            logger.info(f"Using time-slice bucketing: {num_slices} slices, ~{candidates_per_slice} candidates/slice")
            papers = collector.fetch_by_time_slices(
                days=time_window_days,
                num_slices=num_slices,
                candidates_per_slice=candidates_per_slice,
                categories=category_override,
            )
        else:
            papers = collector.fetch_recent_papers(
                days=time_window_days,
                categories=category_override,
                max_results=max_results,
            )

        # Deduplicate by arxiv_id
        dedup = {}
        for p in papers:
            dedup[p.arxiv_id] = p
        papers = list(dedup.values())
        
        for paper_data in papers:
            existing = self.db.query(Paper).filter(Paper.arxiv_id == paper_data.arxiv_id).first()
            if not existing:
                paper = Paper(
                    arxiv_id=paper_data.arxiv_id,
                    title=paper_data.title,
                    authors=", ".join(paper_data.authors),
                    abstract=paper_data.abstract,
                    categories=", ".join(paper_data.categories),
                    published_date=paper_data.published_date,
                    arxiv_url=paper_data.arxiv_url,
                    pdf_url=paper_data.pdf_url,
                    citation_count=paper_data.citation_count
                )
                self.db.add(paper)
        
        self.db.commit()
        return papers
    
    def _collect_articles(self, time_window_days: int, per_source_limit: int) -> List:
        """Collect new articles from tech sources"""
        all_articles = []
        
        if "hackernews" in settings.TECH_SOURCES:
            collector = HNCollector()
            hn_articles = collector.fetch(days=time_window_days, limit=per_source_limit)
            logger.info(f"Collected {len(hn_articles)} articles from Hacker News (last {time_window_days} days)")
            all_articles.extend(hn_articles)
        
        if "devto" in settings.TECH_SOURCES:
            collector = DevToCollector()
            devto_articles = collector.fetch(days=time_window_days, limit=per_source_limit)
            logger.info(f"Collected {len(devto_articles)} articles from Dev.to (last {time_window_days} days)")
            all_articles.extend(devto_articles)
        
        if "medium" in settings.TECH_SOURCES:
            collector = MediumCollector()
            medium_articles = collector.fetch(days=time_window_days, limit=per_source_limit)
            logger.info(f"Collected {len(medium_articles)} articles from Medium (last {time_window_days} days)")
            all_articles.extend(medium_articles)
        
        logger.info(f"Total articles collected: {len(all_articles)} (from last {time_window_days} days)")
        
        # Store in database
        new_count = 0
        for article_data in all_articles:
            existing = self.db.query(Article).filter(Article.url == article_data.url).first()
            if not existing:
                article = Article(
                    source=article_data.source,
                    source_id=article_data.source_id,
                    title=article_data.title,
                    url=article_data.url,
                    content=article_data.content[:5000],  # Limit content length
                    author=article_data.author,
                    published_date=article_data.published_date,
                    upvotes=article_data.upvotes
                )
                try:
                    self.db.add(article)
                    self.db.flush()  # detect unique violations early
                    new_count += 1
                except sa_exc.IntegrityError:
                    self.db.rollback()
                    logger.debug(f"Duplicate article skipped (url={article_data.url})")
                except Exception as e:
                    self.db.rollback()
                    logger.error(f"Error saving article {article_data.url}: {e}")
        
        self.db.commit()

        # Store db_id on each collector object for interaction tracking
        for article_data in all_articles:
            db_art = self.db.query(Article).filter(Article.url == article_data.url).first()
            if db_art:
                article_data.db_id = db_art.id

        logger.info(f"Stored {new_count} new articles in database")
        return all_articles
    
    
    def _filter_candidates(
        self, 
        items: List, 
        item_type: str, 
        selected_interests: List[str], 
        min_threshold: float = None,
        min_papers_per_slice: int = 10,  # Minimum papers we want per time slice
    ) -> List:
        """
        Filter candidates with quality hard filter for papers:
        (relevance >= r_min) AND ((impact_score >= i_min) OR has_doi/journal_ref)
        
        For articles, uses simple relevance threshold.
        Auto-relaxes thresholds if not enough papers pass.
        """
        if item_type == "paper":
            return self._filter_papers_with_quality(
                items, selected_interests, min_threshold, min_papers_per_slice
            )
        else:
            return self._filter_articles(items, selected_interests, min_threshold)
    
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
