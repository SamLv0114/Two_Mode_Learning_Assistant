"""
Feature extraction for ranking models
"""
from typing import List, Dict, Optional
from datetime import datetime, timezone
from src.models.recommender import Recommender


class FeatureExtractor:

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
    def _author_reputation_score(paper) -> float:
        """Heuristic based on author count: values collaboration but not excessively."""
        authors = getattr(paper, 'authors', [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(',') if a.strip()]

        num_authors = len(authors)
        if 2 <= num_authors <= 5:
            return 1.0
        elif 6 <= num_authors <= 10:
            return 0.8
        elif num_authors > 10:
            return 0.6
        elif num_authors == 1:
            return 0.7
        else:
            return 0.0

    @staticmethod
    def extract_features(
        item,
        item_type: str,
        embedding_manager,
        user_interests: List[str],
        recent_texts: Optional[List[str]] = None,
    ) -> Dict:
        """Extract 13 ranking features for a paper or article."""
        is_paper = (item_type == "paper")
        interests_text = " ".join(user_interests)

        # Text fields differ: papers have abstract, articles have content
        body_field = 'abstract' if is_paper else 'content'
        body_text = getattr(item, body_field, '') or ''
        full_text = f"{item.title} {body_text}"
        full_text_lower = full_text.lower()

        similarity = embedding_manager.get_similarity_score(interests_text, full_text)

        # Recency
        if hasattr(item, 'published_date') and item.published_date:
            now = datetime.now(timezone.utc)
            pub_date = item.published_date
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            else:
                pub_date = pub_date.astimezone(timezone.utc)
            days_old = (now - pub_date).days
            recency_score = 1.0 / (1.0 + days_old / 30.0)
        else:
            recency_score = 0.5

        # Impact: papers use heuristic score, articles use upvote count
        if is_paper:
            impact_score = Recommender.calculate_impact_score(item, include_venue=False)
        else:
            upvotes = getattr(item, 'upvotes', 0)
            impact_score = min(upvotes / 500.0, 1.0)

        # Category relevance (papers only)
        if is_paper:
            categories = getattr(item, 'categories', [])
            category_score = 1.0 if any(cat in categories for cat in ["cs.LG", "cs.AI"]) else 0.5
        else:
            category_score = 0.0

        # Source quality (articles only)
        if is_paper:
            source_score = 0.0
        else:
            source = getattr(item, 'source', '')
            source_score = 1.0 if source == "hackernews" else 0.7

        title_length = len(getattr(item, 'title', ''))
        title_score = 1.0 - min(title_length / 200.0, 0.5)

        content_max = 3000.0 if is_paper else 2000.0
        content_score = min(len(body_text) / content_max, 1.0)

        readability = FeatureExtractor._basic_readability(body_text)
        has_code = FeatureExtractor._has_code(full_text_lower)
        is_survey = FeatureExtractor._is_survey_or_tutorial(full_text_lower)
        novelty = FeatureExtractor._novelty_score(full_text, recent_texts, embedding_manager)
        venue_score = FeatureExtractor._venue_score(full_text_lower) if is_paper else 0.0
        author_reputation = FeatureExtractor._author_reputation_score(item) if is_paper else 0.5

        return {
            "similarity": similarity,
            "recency": recency_score,
            "impact": impact_score,
            "category": category_score,
            "source": source_score,
            "title_length": title_score,
            "content_length": content_score,
            "readability": readability,
            "has_code": has_code,
            "is_survey": is_survey,
            "novelty": novelty,
            "venue": venue_score,
            "author_reputation": author_reputation,
        }
