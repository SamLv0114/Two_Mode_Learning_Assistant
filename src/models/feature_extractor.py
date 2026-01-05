"""
Feature extraction for ranking models
"""
from typing import List, Dict
from datetime import datetime, timezone
import math
from src.models.recommender import Recommender


class FeatureExtractor:
    """Extracts features for ranking models"""
    
    @staticmethod
    def extract_paper_features(paper, embedding_manager, user_interests: List[str]) -> Dict:
        interests_text = " ".join(user_interests)
        paper_text = f"{paper.title} {paper.abstract if hasattr(paper, 'abstract') else ''}"
        
        similarity = embedding_manager.get_similarity_score(interests_text, paper_text)
        
        # Recency (days since publication)
        if hasattr(paper, 'published_date') and paper.published_date:
            # Normalize both datetimes to UTC-aware for comparison
            now = datetime.now(timezone.utc)
            pub_date = paper.published_date
            # assume it's UTC
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            else:
                pub_date = pub_date.astimezone(timezone.utc)
            days_old = (now - pub_date).days
            recency_score = 1.0 / (1.0 + days_old / 30.0)  # Decay over 30 days
        else:
            recency_score = 0.5
        
        # Heuristic impact (use recommender heuristic as the impact/citation proxy)
        impact_score = Recommender.calculate_impact_score(paper)
        citation_score = impact_score  # treat heuristic as citation proxy for now
        
        # Category relevance (check if in preferred categories)
        categories = getattr(paper, 'categories', [])
        category_score = 1.0 if any(cat in categories for cat in ["cs.LG", "cs.AI"]) else 0.5
        
        # Title length (shorter titles often more focused)
        title_length = len(getattr(paper, 'title', ''))
        title_score = 1.0 - min(title_length / 200.0, 0.5)
        
        return {
            "similarity": similarity,
            # "recency": recency_score,
            # "citations": citation_score,
            "impact": impact_score,
            "category": category_score,
            "title_length": title_score
        }
    
    @staticmethod
    # Similar to the paper features extractor
    def extract_article_features(article, embedding_manager, user_interests: List[str]) -> Dict:
        interests_text = " ".join(user_interests)
        article_text = f"{article.title} {article.content if hasattr(article, 'content') else ''}"
        
        similarity = embedding_manager.get_similarity_score(interests_text, article_text)
        
        # Recency
        if hasattr(article, 'published_date') and article.published_date:
            # Normalize both datetimes to UTC-aware for comparison
            now = datetime.now(timezone.utc)
            pub_date = article.published_date
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            else:
                pub_date = pub_date.astimezone(timezone.utc)
            days_old = (now - pub_date).days
            recency_score = 1.0 / (1.0 + days_old / 30.0)  # Articles decay faster (7 days)
        else:
            recency_score = 0.5
        
        # Engagement (upvotes, normalized)
        upvotes = getattr(article, 'upvotes', 0)
        engagement_score = min(upvotes / 500.0, 1.0)  # Normalize to 0-1
        impact_score = engagement_score  # alias for stars scoring
        
        # Source quality (Hacker News typically higher quality)
        source = getattr(article, 'source', '')
        source_score = 1.0 if source == "hackernews" else 0.7
        
        # Content length (longer articles often more comprehensive)
        content_length = len(getattr(article, 'content', ''))
        content_score = min(content_length / 2000.0, 1.0)
        
        return {
            "similarity": similarity,
            # "recency": recency_score,
            # "engagement": engagement_score,
            "impact": impact_score,
            "source": source_score,
            "content_length": content_score
        }

