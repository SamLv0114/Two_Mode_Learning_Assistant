"""
MODE 1: Daily Recommendation Feed Pipeline
"""
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
from src.database.models import Paper, Article, SessionLocal, init_db
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        init_db()
    
    def run(self, time_window_days: int = 7, focus_areas: List[str] = None, user_interests: List[str] = None):
        logger.info("Starting daily feed pipeline...")

        # Normalize user selections
        time_window_days = max(1, time_window_days)
        selected_interests = focus_areas if focus_areas else settings.USER_INTERESTS
        # Store full interest phrases for personalized summaries
        self._user_interests = user_interests or selected_interests
        logger.info("Daily feed mode")
        # Scale fetch budgets with time window (simple linear scale, capped)
        scale = min(time_window_days / 7.0, 52)  # up to ~1 year
        paper_fetch_limit = int(settings.MAX_PAPERS_PER_DAY * scale)
        article_fetch_limit = int(30 * scale)  # base 30 per source
        logger.info(f"Fetch scale factor={scale:.2f}, paper_limit={paper_fetch_limit}, article_limit_per_source={article_fetch_limit}")

        # Step 1: Collect new content
        logger.info("Step 1: Collecting new content...")
        # Use time-slice bucketing to ensure coverage across the time window
        # This divides the time window into equal slices and samples from each
        use_time_slices = True
        new_papers = self._collect_papers(
            time_window_days,
            selected_interests,
            paper_fetch_limit,
            use_time_slices=use_time_slices,
        )
        new_articles = self._collect_articles(time_window_days, article_fetch_limit)
        
        # Step 2: Filter candidates with quality hard filter
        logger.info(f"Step 2: Filtering candidates with interests: {', '.join(selected_interests)}...")
        # Calculate min papers per slice for auto-relaxation
        # Divide time window into slices (same logic as collection)
        if time_window_days >= 365:
            num_slices = 12
        elif time_window_days >= 30:
            num_slices = 4
        elif time_window_days >= 7:
            num_slices = 7
        else:
            num_slices = time_window_days
        min_papers_per_slice = max(5, settings.TOP_PAPERS_COUNT // num_slices)
        
        candidate_papers = self._filter_candidates(
            new_papers,
            item_type="paper",
            selected_interests=selected_interests,
            min_threshold=0.25,  # slightly lower than default for broader windows
            min_papers_per_slice=min_papers_per_slice,
        )
        candidate_articles = self._filter_candidates(
            new_articles,
            item_type="article",
            selected_interests=selected_interests,
            min_threshold=0.15,  # articles often have shorter content; lower threshold
        )
        
        # Step 3: Rank items (heuristic before training threshold, ML after)
        logger.info("Step 3: Ranking items using unified signals...")
        interaction_count = self.trainer.get_interaction_count()
        use_ml = interaction_count >= 50
        # Calculate heuristic impact scores for all papers
        if candidate_papers:
            logger.info(f"Computing heuristic scores for {len(candidate_papers)} papers...")
            for paper in candidate_papers:
                # Calculate impact using proxy signals (part of unified Ranker)
                impact_score = self.recommender.calculate_impact_score(paper)
                # Store heuristic impact separately from citations
                paper.heuristic_impact_score = impact_score
            logger.info("Heuristic scoring complete (no API calls needed)")
        top_papers = self._rank_and_select(
            candidate_papers,
            settings.TOP_PAPERS_COUNT,
            item_type="paper",
            selected_interests=selected_interests,
            use_ml=use_ml,
        )
        top_articles = self._rank_and_select(
            candidate_articles,
            settings.TOP_ARTICLES_COUNT,
            item_type="article",
            selected_interests=selected_interests,
            use_ml=use_ml,
        )
        
        # Step 4: Generate personalized summaries
        logger.info("Step 4: Generating personalized summaries...")
        top_papers = self._generate_summaries(top_papers, item_type="paper")
        top_articles = self._generate_summaries(top_articles, item_type="article")
        
        # Step 5: Store in database and vector DB
        logger.info("Step 5: Storing results...")
        self._store_results(top_papers, top_articles)
        
        # Step 6: Format and return
        logger.info("Step 6: Formatting output...")
        output = self._format_output(top_papers, top_articles)
        
        # Step 7: Check if model should be retrained
        if use_ml:
            logger.info(f"Step 7: Retraining model with {interaction_count} interactions...")
            # Use validation split if we have enough data (>=100 examples)
            if self.trainer.retrain_model(self.recommender, min_interactions=50, use_validation=True):
                logger.info("Model retrained successfully! Recommendations will be more personalized.")
        else:
            logger.info(f"Step 7: Skipping retrain ({interaction_count}/50 interactions)")
        
        # Note: Interactions are recorded when user explicitly clicks Save/View/Dismiss via the Streamlit UI
        
        logger.info("Daily feed pipeline completed!")
        return output
    
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
            
            candidates_per_slice = min(500, max(200, max_results // num_slices))
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
        """Simple relevance filter for articles"""
        
        filtered = []
        interests_text = " ".join(selected_interests)
        threshold = min_threshold if min_threshold is not None else settings.MIN_SIMILARITY_THRESHOLD

        # simple seen titles set to drop near-duplicates (case-insensitive exact)
        seen_titles = set()
        similarities = []
        for item in articles:
            item_text = f"{item.title} {item.content if hasattr(item, 'content') else ''}"
            similarity = self.embedding_manager.get_similarity_score(interests_text, item_text)
            similarities.append(similarity)
            
            if similarity >= threshold:
                title_key = getattr(item, "title", "").strip().lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                filtered.append(item)
        
        if similarities:
            max_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            logger.info(f"Article similarity stats: max={max_sim:.3f}, avg={avg_sim:.3f}, threshold={threshold}")
        
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
                .filter(Paper.recommended == True)
                .filter(Paper.recommended_date >= cutoff)
                .order_by(Paper.recommended_date.desc())
                .limit(settings.NOVELTY_MAX_ITEMS)
                .all()
            )
            for item in items:
                texts.append(f"{item.title} {item.abstract or ''}")
        else:
            items = (
                self.db.query(Article)
                .filter(Article.recommended == True)
                .filter(Article.recommended_date >= cutoff)
                .order_by(Article.recommended_date.desc())
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
        """Generate personalized summaries for items"""
        for item in items:
            if item_type == "paper":
                content = item.abstract
            else:
                content = item.content[:1000] if hasattr(item, 'content') else ""

            summary = self.generator.generate_summary(
                title=item.title,
                content=content,
                user_interests=getattr(self, '_user_interests', None)
            )
            item.personalized_summary = summary

        return items
    
    def _store_results(self, papers: List[PaperData], articles: List):
        """Store recommended items in database and vector DB"""
        # Update database
        for paper_data in papers:
            paper = self.db.query(Paper).filter(Paper.arxiv_id == paper_data.arxiv_id).first()
            if paper:
                paper.relevance_score = paper_data.relevance_score
                paper.personalized_summary = paper_data.personalized_summary
                paper.recommended = True
                paper.recommended_date = datetime.now(timezone.utc)
                # Update citation count if fetched
                if paper_data.citation_count is not None:
                    paper.citation_count = paper_data.citation_count
                if getattr(paper_data, "heuristic_impact_score", None) is not None:
                    paper.heuristic_impact_score = paper_data.heuristic_impact_score
                
                # Add to vector DB if not already there
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
        
        for article_data in articles:
            article = self.db.query(Article).filter(Article.url == article_data.url).first()
            if article:
                article.relevance_score = article_data.relevance_score
                article.personalized_summary = article_data.personalized_summary
                article.recommended = True
                article.recommended_date = datetime.now(timezone.utc)
                
                # Add to vector DB
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
