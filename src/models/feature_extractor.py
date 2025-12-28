"""
Feature extraction for ranking models
"""
from typing import List, Dict
from datetime import datetime, timezone


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
            # If published_date is naive, assume it's UTC
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            # If published_date is aware, convert to UTC
            else:
                pub_date = pub_date.astimezone(timezone.utc)
            days_old = (now - pub_date).days
            recency_score = 1.0 / (1.0 + days_old / 30.0)  # Decay over 30 days
        else:
            recency_score = 0.5
        
        # Citation count (normalized)
        citation_count = getattr(paper, 'citation_count', 0)
        citation_score = min(citation_count / 100.0, 1.0)  # Normalize to 0-1
        
        # Category relevance (check if in preferred categories)
        categories = getattr(paper, 'categories', [])
        category_score = 1.0 if any(cat in categories for cat in ["cs.LG", "cs.AI"]) else 0.5
        
        # Title length (shorter titles often more focused)
        title_length = len(getattr(paper, 'title', ''))
        title_score = 1.0 - min(title_length / 200.0, 0.5)
        
        return {
            "similarity": similarity,
            "recency": recency_score,
            "citations": citation_score,
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
            # If published_date is naive, assume it's UTC
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            # If published_date is aware, convert to UTC
            else:
                pub_date = pub_date.astimezone(timezone.utc)
            days_old = (now - pub_date).days
            recency_score = 1.0 / (1.0 + days_old / 7.0)  # Articles decay faster (7 days)
        else:
            recency_score = 0.5
        
        # Engagement (upvotes, normalized)
        upvotes = getattr(article, 'upvotes', 0)
        engagement_score = min(upvotes / 500.0, 1.0)  # Normalize to 0-1
        
        # Source quality (Hacker News typically higher quality)
        source = getattr(article, 'source', '')
        source_score = 1.0 if source == "hackernews" else 0.7
        
        # Content length (longer articles often more comprehensive)
        content_length = len(getattr(article, 'content', ''))
        content_score = min(content_length / 2000.0, 1.0)
        
        return {
            "similarity": similarity,
            "recency": recency_score,
            "engagement": engagement_score,
            "source": source_score,
            "content_length": content_score
        }

