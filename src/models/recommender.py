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

try:
    import lightgbm as lgb
except Exception:
    lgb = None

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
        self.model_type = None
        self._load_or_create_model()
    
    # ============================================================================
    # ML MODEL (Learns from user interactions)
    # ============================================================================
    
    def _load_or_create_model(self):
        """Load existing ML model or create a new one"""
        expected_features = [
            "similarity",
            "recency",
            "impact",
            "category",
            "source",
            "title_length",
            "content_length",
            "readability",
            "has_code",
            "is_survey",
            "novelty",
            "venue"
        ]

        # Expected hyperparameters
        expected_n_estimators = 50
        expected_num_leaves = 15

        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.feature_names = data['feature_names']
                    self.model_type = data.get('model_type', 'regressor')
                    self.is_trained = data.get('is_trained', True)  # Assume old models are trained

                # Check if features changed
                if self.feature_names != expected_features:
                    logger.warning("Feature names changed; recreating model with new feature set.")
                    self._create_new_model()
                    return

                # Check if LTR setting changed
                if settings.USE_LTR and self.model_type != "ltr":
                    logger.warning("LTR enabled but stored model is not a ranker; recreating model.")
                    self._create_new_model()
                    return

                # Check if hyperparameters changed (for LightGBM models)
                if self.model_type == "ltr" and lgb is not None:
                    try:
                        model_n_estimators = self.model.n_estimators
                        model_num_leaves = self.model.num_leaves

                        if model_n_estimators != expected_n_estimators or model_num_leaves != expected_num_leaves:
                            logger.warning(
                                f"Model hyperparameters changed (n_estimators: {model_n_estimators}->{expected_n_estimators}, "
                                f"num_leaves: {model_num_leaves}->{expected_num_leaves}). Recreating model."
                            )
                            self._create_new_model()
                            return
                    except AttributeError:
                        # If we can't check hyperparameters, recreate to be safe
                        logger.warning("Cannot verify model hyperparameters; recreating model.")
                        self._create_new_model()
                        return

                status = "trained" if self.is_trained else "untrained"
                logger.info(f"Loaded existing ML ranking model ({status})")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Creating new model.")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new ML ranking model"""
        use_ltr = settings.USE_LTR and lgb is not None
        if settings.USE_LTR and lgb is None:
            logger.warning("LightGBM not installed; falling back to regressor model.")

        if use_ltr:
            self.model_type = "ltr"
            self.model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=50,
                learning_rate=0.05,
                num_leaves=15,
                min_data_in_leaf=5,
                label_gain=[0, 1, 2],
                random_state=42
            )
        else:
            self.model_type = "regressor"
            self.model = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
        # Feature names (should match FeatureExtractor output)
        self.feature_names = [
            "similarity",
            "recency",
            "impact",
            "category",
            "source",
            "title_length",
            "content_length",
            "readability",
            "has_code",
            "is_survey",
            "novelty",
            "venue"
        ]

        # Mark as untrained (will be trained on first real data)
        self.is_trained = False
        logger.info("Created new ML ranking model (untrained - will use heuristics until trained on real data)")
    
    def _save_model(self):
        """Save the ML model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'model_type': self.model_type,
                    'is_trained': self.is_trained
                }, f)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def rank_items(self, items: List, features: List[Dict]) -> List[Tuple]:
        """
        Rank items using ML model (learns from user interactions)
        Falls back to heuristic scoring if model is not yet trained.

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

        # If model is not trained, use heuristic scoring
        if not self.is_trained:
            logger.debug("Model not yet trained, using heuristic scoring")
            scores = []
            for item, feat in zip(items, features):
                # Use impact feature as the heuristic score
                score = feat.get('impact', 0.0)
                scores.append(score)
            scores = np.array(scores)
        else:
            # Predict scores using ML model
            scores = self.model.predict(X)

        # Sort by score (descending)
        ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)

        return ranked
    
    def update_model(self, X: np.ndarray, y: np.ndarray, group: Optional[List[int]] = None):
        """
        Update ML model with new training data

        Args:
            X: Feature matrix
            y: Target scores (e.g., from user interactions: saved=1.0, viewed=0.6, dismissed=0.0)
            group: Group sizes for ranking (required for LTR)
        """
        if self.model_type == "ltr":
            if not group:
                logger.error("LTR training requires group sizes; skipping update.")
                return
            y = np.array(y, dtype=int)
            self.model.fit(X, y, group=group)
        else:
            self.model.fit(X, y)

        # Mark model as trained
        self.is_trained = True
        self._save_model()
        logger.info("Updated ML ranking model with new training data")
    
    # ============================================================================
    # HEURISTIC SCORING (Quality baseline, no API needed)
    # ============================================================================
    
    @staticmethod
    def calculate_impact_score(paper, include_venue: bool = True) -> float:
        """
        Calculate heuristic impact score (used as a quality proxy for ML)
        
        This provides a quality baseline even before any user interactions.
        Signals that correlate with high-impact papers:
        - High-value keywords (survey, benchmark, SOTA)
        - Code availability (GitHub)
        - Author count and collaboration
        - Top venues and conferences
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
        if include_venue:
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
        
        # 5. Recency (weight: 0.10, continuous decay)
        published_date = getattr(paper, 'published_date', None)
        if published_date:
            try:
                now = datetime.now(timezone.utc)
                if published_date.tzinfo is None:
                    published_date = published_date.replace(tzinfo=timezone.utc)
                else:
                    published_date = published_date.astimezone(timezone.utc)
                
                days_old = (now - published_date).days
                score += 0.10 * np.exp(-days_old / 180.0)
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
        Generate human-readable explanation of heuristic score for debugging
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

