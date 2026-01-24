"""
Feature extraction for ranking models
"""
from typing import List, Dict, Optional
from datetime import datetime, timezone
import math
from src.models.recommender import Recommender


class FeatureExtractor:
    """Extracts features for ranking models"""

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
    def _venue_score(text: str) -> float:
        if not text:
            return 0.0
        venues = {
            'neurips': 1.0, 'nips': 1.0, 'icml': 1.0, 'iclr': 1.0,
            'cvpr': 0.9, 'iccv': 0.9, 'eccv': 0.85,
            'acl': 0.9, 'emnlp': 0.85, 'naacl': 0.8, 'coling': 0.75,
            'aaai': 0.8, 'ijcai': 0.8,
        }
        for venue, weight in venues.items():
            if venue in text:
                return weight
        return 0.0

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
    def extract_paper_features(
        paper,
        embedding_manager,
        user_interests: List[str],
        recent_texts: Optional[List[str]] = None
    ) -> Dict:
        interests_text = " ".join(user_interests)
        paper_text = f"{paper.title} {paper.abstract if hasattr(paper, 'abstract') else ''}"
        paper_text_lower = paper_text.lower()
        
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
        
        # Heuristic impact without venue to avoid double-counting
        impact_score = Recommender.calculate_impact_score(paper, include_venue=False)
        citation_score = impact_score  # treat heuristic as citation proxy for now
        
        # Category relevance (check if in preferred categories)
        categories = getattr(paper, 'categories', [])
        category_score = 1.0 if any(cat in categories for cat in ["cs.LG", "cs.AI"]) else 0.5

        venue_score = FeatureExtractor._venue_score(paper_text_lower)
        
        # Title length (shorter titles often more focused)
        title_length = len(getattr(paper, 'title', ''))
        title_score = 1.0 - min(title_length / 200.0, 0.5)

        # Content length (cap to keep within 0-1 range)
        content_length = len(getattr(paper, 'abstract', '') or '')
        content_score = min(content_length / 3000.0, 1.0)

        readability = FeatureExtractor._basic_readability(getattr(paper, 'abstract', '') or '')
        has_code = FeatureExtractor._has_code(paper_text_lower)
        is_survey = FeatureExtractor._is_survey_or_tutorial(paper_text_lower)
        novelty = FeatureExtractor._novelty_score(paper_text, recent_texts, embedding_manager)
        
        return {
            "similarity": similarity,
            "recency": recency_score,
            "impact": impact_score,
            "category": category_score,
            "source": 0.0,
            "title_length": title_score,
            "content_length": content_score,
            "readability": readability,
            "has_code": has_code,
            "is_survey": is_survey,
            "novelty": novelty,
            "venue": venue_score
        }
    
    @staticmethod
    # Similar to the paper features extractor
    def extract_article_features(
        article,
        embedding_manager,
        user_interests: List[str],
        recent_texts: Optional[List[str]] = None
    ) -> Dict:
        interests_text = " ".join(user_interests)
        article_text = f"{article.title} {article.content if hasattr(article, 'content') else ''}"
        article_text_lower = article_text.lower()
        
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
        
        # Title length (shorter titles often more focused)
        title_length = len(getattr(article, 'title', ''))
        title_score = 1.0 - min(title_length / 200.0, 0.5)

        # Content length (longer articles often more comprehensive)
        content_length = len(getattr(article, 'content', ''))
        content_score = min(content_length / 2000.0, 1.0)
        readability = FeatureExtractor._basic_readability(getattr(article, 'content', '') or '')
        has_code = FeatureExtractor._has_code(article_text_lower)
        is_survey = FeatureExtractor._is_survey_or_tutorial(article_text_lower)
        novelty = FeatureExtractor._novelty_score(article_text, recent_texts, embedding_manager)
        
        return {
            "similarity": similarity,
            "recency": recency_score,
            "impact": impact_score,
            "category": 0.0,
            "source": source_score,
            "title_length": title_score,
            "content_length": content_score,
            "readability": readability,
            "has_code": has_code,
            "is_survey": is_survey,
            "novelty": novelty,
            "venue": 0.0
        }

