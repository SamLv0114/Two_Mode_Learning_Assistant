"""
MODE 1: Daily Recommendation Feed Pipeline
"""
from datetime import datetime, timezone
from typing import List, Dict
import logging

from src.collectors import ArxivCollector, PaperData, HNCollector, MediumCollector, DevToCollector
from src.models import EmbeddingManager, Recommender, FeatureExtractor
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
        self.recommender = Recommender()
        self.db = SessionLocal()
        init_db()
    
    def run(self):
        logger.info("Starting daily feed pipeline...")
        
        # Step 1: Collect new content
        logger.info("Step 1: Collecting new content...")
        new_papers = self._collect_papers()
        new_articles = self._collect_articles()
        
        # Step 2: Filter candidates
        logger.info("Step 2: Filtering candidates...")
        candidate_papers = self._filter_candidates(new_papers, item_type="paper")
        candidate_articles = self._filter_candidates(new_articles, item_type="article")
        
        # Step 3: Rank items
        logger.info("Step 3: Ranking items...")
        top_papers = self._rank_and_select(candidate_papers, settings.TOP_PAPERS_COUNT, item_type="paper")
        top_articles = self._rank_and_select(candidate_articles, settings.TOP_ARTICLES_COUNT, item_type="article")
        
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
        
        logger.info("Daily feed pipeline completed!")
        return output
    
    def _collect_papers(self) -> List[PaperData]:
        """Collect new papers from ArXiv"""
        collector = ArxivCollector()
        papers = collector.fetch_recent_papers(days=7)
        
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
    
    def _collect_articles(self) -> List:
        """Collect new articles from tech sources"""
        all_articles = []
        
        if "hackernews" in settings.TECH_SOURCES:
            collector = HNCollector()
            hn_articles = collector.fetch(days=7)
            logger.info(f"Collected {len(hn_articles)} articles from Hacker News (last 7 days)")
            all_articles.extend(hn_articles)
        
        if "devto" in settings.TECH_SOURCES:
            collector = DevToCollector()
            devto_articles = collector.fetch(days=7)
            logger.info(f"Collected {len(devto_articles)} articles from Dev.to (last 7 days)")
            all_articles.extend(devto_articles)
        
        if "medium" in settings.TECH_SOURCES:
            collector = MediumCollector()
            medium_articles = collector.fetch(days=7)
            logger.info(f"Collected {len(medium_articles)} articles from Medium (last 7 days)")
            all_articles.extend(medium_articles)
        
        logger.info(f"Total articles collected: {len(all_articles)} (from last 7 days)")
        
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
                self.db.add(article)
                new_count += 1
        
        self.db.commit()
        logger.info(f"Stored {new_count} new articles in database")
        return all_articles
    
    def _filter_candidates(self, items: List, item_type: str) -> List:
        """Filter candidates by similarity threshold"""
        filtered = []
        interests_text = " ".join(settings.USER_INTERESTS)
        
        # Debug: log user interests
        if items:
            logger.info(f"Filtering {len(items)} {item_type}s with user interests: {', '.join(settings.USER_INTERESTS[:3])}...")
        
        similarities = []
        for item in items:
            if item_type == "paper":
                item_text = f"{item.title} {item.abstract}"
            else:
                item_text = f"{item.title} {item.content if hasattr(item, 'content') else ''}"
            
            similarity = self.embedding_manager.get_similarity_score(interests_text, item_text)
            similarities.append(similarity)
            
            if similarity >= settings.MIN_SIMILARITY_THRESHOLD:
                filtered.append(item)
        
        # Debug: log similarity stats
        if similarities:
            max_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            logger.info(f"Similarity stats: max={max_sim:.3f}, avg={avg_sim:.3f}, threshold={settings.MIN_SIMILARITY_THRESHOLD}")
        
        logger.info(f"Filtered {len(filtered)}/{len(items)} {item_type}s above threshold")
        return filtered
    
    def _rank_and_select(self, items: List, top_k: int, item_type: str) -> List:
        """Rank items and select top K"""
        if not items:
            return []
        
        # Extract features
        features = []
        for item in items:
            if item_type == "paper":
                feat = self.feature_extractor.extract_paper_features(item, self.embedding_manager, settings.USER_INTERESTS)
            else:
                feat = self.feature_extractor.extract_article_features(item, self.embedding_manager, settings.USER_INTERESTS)
            features.append(feat)
        
        # Rank
        ranked = self.recommender.rank_items(items, features)
        
        # Select top K
        top_items = [item for item, score in ranked[:top_k]]
        
        # Store relevance scores
        for (item, score), top_item in zip(ranked[:top_k], top_items):
            top_item.relevance_score = score
        
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
            output["papers"].append({
                "rank": i,
                "title": paper.title,
                "arxiv_id": paper.arxiv_id,
                "url": paper.arxiv_url,
                "citation_count": paper.citation_count,
                "summary": paper.personalized_summary,
                "relevance_score": paper.relevance_score
            })
        
        for i, article in enumerate(articles, 1):
            output["articles"].append({
                "rank": i,
                "title": article.title,
                "url": article.url,
                "source": article.source,
                "upvotes": article.upvotes,
                "summary": article.personalized_summary,
                "relevance_score": article.relevance_score
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

