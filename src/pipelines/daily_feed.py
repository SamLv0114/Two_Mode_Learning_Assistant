"""
MODE 1: Daily Recommendation Feed Pipeline
"""
from datetime import datetime, timezone, timedelta
from typing import List, Dict
import logging

from sqlalchemy import exc as sa_exc

from src.collectors import ArxivCollector, PaperData, HNCollector, MediumCollector, DevToCollector
from src.models import EmbeddingManager, Recommender, FeatureExtractor, ModelTrainer
from src.rag import Generator
from src.database.models import Paper, Article, SessionLocal, init_db
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyFeedPipeline:
    """Main pipeline for daily feed generation"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.generator = Generator()
        self.feature_extractor = FeatureExtractor()
        self.recommender = Recommender()  # Unified: heuristic + ML personalization
        self.db = SessionLocal()
        self.trainer = ModelTrainer(self.db, self.embedding_manager)  # Learns from your clicks
        init_db()
    
    def run(self, time_window_days: int = 7, focus_areas: List[str] = None):
        logger.info("Starting daily feed pipeline...")
        
        # Normalize user selections
        time_window_days = max(1, time_window_days)
        selected_interests = focus_areas if focus_areas else settings.USER_INTERESTS
        logger.info("Daily feed mode")
        # Scale fetch budgets with time window (simple linear scale, capped)
        scale = min(time_window_days / 7.0, 52)  # up to ~1 year
        paper_fetch_limit = int(settings.MAX_PAPERS_PER_DAY * scale)
        article_fetch_limit = int(30 * scale)  # base 30 per source
        logger.info(f"Fetch scale factor={scale:.2f}, paper_limit={paper_fetch_limit}, article_limit_per_source={article_fetch_limit}")

        # Step 1: Collect new content
        logger.info("Step 1: Collecting new content...")
        new_papers = self._collect_papers(
            time_window_days,
            selected_interests,
            paper_fetch_limit,
            enrich_citations=settings.CITATION_ENRICHMENT_ENABLED,
        )
        new_articles = self._collect_articles(time_window_days, article_fetch_limit)
        
        # Step 2: Filter candidates
        logger.info(f"Step 2: Filtering candidates with interests: {', '.join(selected_interests)}...")
        candidate_papers = self._filter_candidates(
            new_papers,
            item_type="paper",
            selected_interests=selected_interests,
            min_threshold=0.25,  # slightly lower than default for broader windows
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

        # Optional citation enrichment for shortlist if enabled and we plan to use ML
        if use_ml and settings.CITATION_ENRICHMENT_ENABLED and candidate_papers:
            shortlist = candidate_papers[: max(settings.TOP_PAPERS_COUNT * 5, 50)]
            cita_fetcher = ArxivCollector()
            for p in shortlist:
                if p.citation_count is None:
                    p.citation_count = cita_fetcher._fetch_citation_count(p.arxiv_id)

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
            if self.trainer.retrain_model(self.recommender, min_interactions=50):
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
    
    def _collect_papers(self, time_window_days: int, selected_interests: List[str], max_results: int, enrich_citations: bool = False,) -> List[PaperData]:
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
        # Deduplicate categories; if none matched, use defaults
        category_override = list(dict.fromkeys(categories)) if categories else None

        papers = collector.fetch_recent_papers(
            days=time_window_days,
            categories=category_override,
            max_results=max_results,
        )

        # Try to fetch high impact papers in addition to the main feed
        impact_queries = [
            "survey OR benchmark OR dataset",
            "state-of-the-art OR sota",
        ]
        for q in impact_queries:
            try:
                extra = collector.fetch_by_query(
                    q,
                    max_results=max_results // 4,
                    categories=category_override,
                )
                cutoff = datetime.now(timezone.utc) - timedelta(days=time_window_days)
                for p in extra:
                    if p.published_date and p.published_date >= cutoff:
                        papers.append(p)
            except Exception as e:
                logger.debug(f"Impact lane fetch failed for query '{q}': {e}")

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
        logger.info(f"Stored {new_count} new articles in database")
        return all_articles
    
    def _filter_candidates(self, items: List, item_type: str, selected_interests: List[str], min_threshold: float = None) -> List:
        """Filter candidates by similarity threshold"""
        filtered = []
        interests_text = " ".join(selected_interests)
        threshold = min_threshold if min_threshold is not None else settings.MIN_SIMILARITY_THRESHOLD

        # Diversity prep: simple seen titles set to drop near-duplicates (case-insensitive exact)
        seen_titles = set()
        
        # Debug: log user interests
        if items:
            logger.info(f"Filtering {len(items)} {item_type}s with user interests: {', '.join(selected_interests)}...")
        
        similarities = []
        for item in items:
            if item_type == "paper":
                item_text = f"{item.title} {item.abstract}"
            else:
                item_text = f"{item.title} {item.content if hasattr(item, 'content') else ''}"
            
            similarity = self.embedding_manager.get_similarity_score(interests_text, item_text)
            similarities.append(similarity)
            
            if similarity >= threshold:
                title_key = getattr(item, "title", "").strip().lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                filtered.append(item)
        
        # Debug: log similarity stats
        if similarities:
            max_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            logger.info(f"Similarity stats: max={max_sim:.3f}, avg={avg_sim:.3f}, threshold={threshold}")
        
        logger.info(f"Filtered {len(filtered)}/{len(items)} {item_type}s above threshold {threshold}")
        return filtered
    
    def _rank_and_select(self, items: List, top_k: int, item_type: str, selected_interests: List[str], use_ml: bool) -> List:
        """Rank items using ML Recommender (learns from your interactions)"""
        if not items:
            return []
        
        # Extract features for each item
        features = []
        for item in items:
            if item_type == "paper":
                feat = self.feature_extractor.extract_paper_features(item, self.embedding_manager, selected_interests)
            else:
                feat = self.feature_extractor.extract_article_features(item, self.embedding_manager, selected_interests)
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

        # Select top K
        top_items = [item for item, score in ranked[:top_k]]

        # Store relevance scores
        for item, score in ranked[:top_k]:
            item.relevance_score = score

        return top_items
    
    def _generate_summaries(self, items: List, item_type: str) -> List:
        """Generate personalized summaries for items"""
        for item in items:
            if item_type == "paper":
                content = item.abstract
            else:
                content = item.content[:1000] if hasattr(item, 'content') else ""
            
            summary = self.generator.generate_summary(
                title=item.title,
                content=content
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
            if citation_count is None and db_paper:
                citation_count = db_paper.citation_count
            citation_display = citation_count if citation_count is not None else "—"
            
            output["papers"].append({
                "rank": i,
                "title": paper.title,
                "arxiv_id": paper.arxiv_id,
                "url": paper.arxiv_url,
                "citation_count": citation_display,
                "summary": paper.personalized_summary,
                "relevance_score": paper.relevance_score,
                "db_id": db_id  # For interaction tracking
            })
        
        for i, article in enumerate(articles, 1):
            # Get database ID for interaction tracking
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
                lines.append(f"   ⭐ {paper['citation_count']} citations")
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
    pipeline = DailyFeedPipeline()
    result = pipeline.run()
    print(pipeline.format_for_display(result))
