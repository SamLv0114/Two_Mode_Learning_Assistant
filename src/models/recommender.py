"""
Unified ranking system combining heuristic scoring and ML personalization
"""
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from src.utils.config import settings
import logging

logger = logging.getLogger(__name__)


class Recommender:
    """
    Unified recommendation system that combines:
    1. Heuristic scoring (quality baseline, always works)
    2. ML personalization (learns from user interactions)
    """
    
    def __init__(self):
        self.model_path = settings.MODELS_DIR / "ranker_model.pkl"
        self.model = None
        self.feature_names = None
        self._load_or_create_model()
    
    # ============================================================================
    # ML MODEL (Learns from user interactions)
    # ============================================================================
    
    def _load_or_create_model(self):
        """Load existing ML model or create a new one"""
        expected_features = [
            "similarity", "recency", "impact", "category", "title_length"
        ]
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.feature_names = data['feature_names']
                if self.feature_names != expected_features:
                    logger.warning("Feature names changed; recreating model with new feature set.")
                    self._create_new_model()
                    return
                logger.info("Loaded existing ML ranking model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Creating new model.")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new ML ranking model"""
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Feature names (should match FeatureExtractor output)
        self.feature_names = ["similarity", "recency", "impact", "category", "title_length"]
        
        # Train on dummy data to initialize
        X_dummy = np.random.rand(100, len(self.feature_names))
        y_dummy = np.random.rand(100)
        self.model.fit(X_dummy, y_dummy)
        
        self._save_model()
        logger.info("Created new ML ranking model")
    
    def _save_model(self):
        """Save the ML model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names
                }, f)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def rank_items(self, items: List, features: List[Dict]) -> List[Tuple]:
        """
        Rank items using ML model (learns from user interactions)
        
        Args:
            items: List of items to rank
            features: List of feature dicts (one per item)
            
        Returns:
            List of (item, score) tuples sorted by score (descending)
        """
        if len(items) != len(features):
            raise ValueError("Items and features must have same length")
        
        # Convert features to array
        X = np.array([[f.get(name, 0.0) for name in self.feature_names] for f in features])
        
        # Predict scores using ML model
        scores = self.model.predict(X)
        
        # Sort by score (descending)
        ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def update_model(self, X: np.ndarray, y: np.ndarray):
        """
        Update ML model with new training data
        
        Args:
            X: Feature matrix
            y: Target scores (e.g., from user interactions: saved=1.0, viewed=0.6, dismissed=0.0)
        """
        self.model.fit(X, y)
        self._save_model()
        logger.info("Updated ML ranking model with new training data")
    
    # ============================================================================
    # HEURISTIC SCORING (Quality baseline, no API needed)
    # ============================================================================
    
    @staticmethod
    def calculate_impact_score(paper) -> float:
        """
        Calculate heuristic impact score (used as "citations" feature for ML)
        
        This provides a quality baseline even before any user interactions.
        Signals that correlate with high-impact papers:
        - High-value keywords (survey, benchmark, SOTA)
        - Code availability (GitHub)
        - Author count and collaboration
        - Top venues and conferences
        - Recency sweet spot (6-18 months)
        - Title and abstract quality
        
        Returns:
            Float between 0.0 and 1.0
        """
        score = 0.0
        
        # Prepare text for analysis
        title = getattr(paper, 'title', '').lower()
        abstract = getattr(paper, 'abstract', '').lower()
        text = f"{title} {abstract}"
        
        # 1. High-value keywords (weight: 0.30)
        keyword_scores = {
            # Meta/review papers (very high impact)
            'survey': 0.15, 'review': 0.15, 'systematic review': 0.18,
            # Benchmark/evaluation papers
            'benchmark': 0.12, 'evaluation': 0.08, 'comparison': 0.06,
            # Novel contributions
            'state-of-the-art': 0.10, 'sota': 0.10, 'novel': 0.07,
            'breakthrough': 0.09, 'first': 0.06,
            # Resources
            'dataset': 0.10, 'corpus': 0.08, 'toolkit': 0.07, 'framework': 0.06,
            # Performance indicators
            'outperform': 0.05, 'achieves': 0.04, 'improves': 0.04, 'advances': 0.05,
        }
        
        keyword_contribution = 0.0
        for keyword, weight in keyword_scores.items():
            if keyword in text:
                keyword_contribution = max(keyword_contribution, weight)
        score += min(keyword_contribution, 0.30)
        
        # 2. Code availability (weight: 0.20)
        code_indicators = ['github', 'code available', 'code is available', 
                          'gitlab', 'bitbucket', 'open source', 'open-source']
        if any(indicator in text for indicator in code_indicators):
            score += 0.20
        
        # 3. Author signals (weight: 0.15)
        authors = getattr(paper, 'authors', [])
        num_authors = len(authors) if authors else 0
        if num_authors >= 5:
            score += 0.15
        elif num_authors >= 3:
            score += 0.12
        elif num_authors >= 2:
            score += 0.08
        
        # 4. Venue/Conference signals (weight: 0.15)
        top_venues = {
            'neurips': 0.15, 'nips': 0.15, 'icml': 0.15, 'iclr': 0.15,
            'cvpr': 0.15, 'iccv': 0.15, 'eccv': 0.14,
            'acl': 0.15, 'emnlp': 0.14, 'naacl': 0.13, 'coling': 0.12,
            'aaai': 0.13, 'ijcai': 0.13,
        }
        
        venue_contribution = 0.0
        for venue, weight in top_venues.items():
            if venue in text:
                venue_contribution = max(venue_contribution, weight)
        score += venue_contribution
        
        # 5. Recency (weight: 0.10)
        published_date = getattr(paper, 'published_date', None)
        if published_date:
            try:
                now = datetime.now(timezone.utc)
                if published_date.tzinfo is None:
                    published_date = published_date.replace(tzinfo=timezone.utc)
                else:
                    published_date = published_date.astimezone(timezone.utc)
                
                days_old = (now - published_date).days
                
                if 180 <= days_old <= 540:  # 6-18 months
                    score += 0.10
                elif 90 <= days_old < 180:  # 3-6 months
                    score += 0.08
                elif 30 <= days_old < 90:  # 1-3 months
                    score += 0.06
                elif days_old < 30:  # Very new
                    score += 0.03
                elif 540 < days_old <= 730:  # 18-24 months
                    score += 0.05
            except:
                pass
        
        # 6. Title quality (weight: 0.05)
        title_words = len(title.split())
        if 5 <= title_words <= 12:
            score += 0.05
        elif 13 <= title_words <= 15:
            score += 0.03
        
        # 7. Abstract quality (weight: 0.05)
        abstract_words = len(abstract.split())
        if 100 <= abstract_words <= 300:
            score += 0.05
        elif 50 <= abstract_words < 100 or 300 < abstract_words <= 400:
            score += 0.03
        
        return min(score, 1.0)
    
    @staticmethod
    def explain_score(paper) -> str:
        """
        Generate human-readable explanation of heuristic score
        Useful for debugging and understanding why papers rank high/low
        """
        lines = []
        lines.append(f"Paper: {getattr(paper, 'title', 'Unknown')[:60]}...")
        lines.append(f"Heuristic Impact Score: {Recommender.calculate_impact_score(paper):.3f}")
        lines.append("")
        lines.append("Score Breakdown:")
        
        text = f"{getattr(paper, 'title', '')} {getattr(paper, 'abstract', '')}".lower()
        
        # Keywords
        keywords_found = []
        for kw in ['survey', 'benchmark', 'dataset', 'sota', 'novel', 'github']:
            if kw in text:
                keywords_found.append(kw)
        if keywords_found:
            lines.append(f"  + Keywords: {', '.join(keywords_found)}")
        
        # Code
        if 'github' in text or 'code' in text:
            lines.append("  + Code available")
        
        # Authors
        num_authors = len(getattr(paper, 'authors', []))
        if num_authors >= 3:
            lines.append(f"  + {num_authors} authors (multi-author)")
        
        # Venue
        for venue in ['neurips', 'icml', 'iclr', 'cvpr', 'acl']:
            if venue in text:
                lines.append(f"  + Top venue ({venue.upper()})")
                break
        
        # Recency
        published_date = getattr(paper, 'published_date', None)
        if published_date:
            try:
                days_old = (datetime.now(timezone.utc) - published_date).days
                months_old = days_old // 30
                lines.append(f"  - Age: ~{months_old} months")
            except:
                pass
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the unified ranker
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from dataclasses import dataclass
    from datetime import timedelta
    
    @dataclass
    class MockPaper:
        arxiv_id: str
        title: str
        abstract: str
        authors: List[str]
        published_date: datetime
    
    # Create test papers
    papers = [
        MockPaper(
            arxiv_id="1",
            title="A Survey of Deep Learning Methods for Natural Language Processing",
            abstract="This survey reviews recent advances in deep learning for NLP. "
                     "Code is available on GitHub.",
            authors=["Author A", "Author B", "Author C"],
            published_date=datetime.now(timezone.utc) - timedelta(days=200)
        ),
        MockPaper(
            arxiv_id="2",
            title="Novel Approach to Image Classification",
            abstract="We propose a novel method that achieves state-of-the-art results.",
            authors=["Author X", "Author Y"],
            published_date=datetime.now(timezone.utc) - timedelta(days=30)
        ),
    ]
    
    print("=" * 80)
    print("UNIFIED RECOMMENDER TEST")
    print("=" * 80)
    
    recommender = Recommender()
    
    for paper in papers:
        impact = recommender.calculate_impact_score(paper)
        print(f"\n{paper.title[:60]}")
        print(f"Impact Score: {impact:.3f}")
        print(recommender.explain_score(paper))
    
    print("\n" + "=" * 80)
    print("ML model loaded and ready for training!")
    print(f"Feature names: {recommender.feature_names}")
    print("=" * 80)

