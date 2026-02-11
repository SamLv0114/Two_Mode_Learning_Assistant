"""
Per-user recommender with model persistence
"""
import pickle
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
import numpy as np
from sqlalchemy.orm import Session

from src.database.models import UserModelState
from src.utils.config import settings

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.ensemble import GradientBoostingRegressor


class UserRecommender:
    """
    Per-user recommender that stores model state in the database.

    Each user gets their own personalized ML model that learns from
    their interactions.
    """

    FEATURE_NAMES = [
        "similarity", "recency", "impact", "category", "source",
        "title_length", "content_length", "readability", "has_code",
        "is_survey", "novelty", "venue", "author_reputation",
    ]

    def __init__(self, user_id: int, db: Session):
        """
        Initialize recommender for a specific user.

        Args:
            user_id: The user's database ID
            db: Database session
        """
        self.user_id = user_id
        self.db = db
        self.model = None
        self.model_type = None
        self.heuristic_weights = None
        self.is_trained = False
        self.feature_names = list(self.FEATURE_NAMES)

        self._load_state()

    def _load_state(self):
        """Load model state from database"""
        state = self.db.query(UserModelState).filter(
            UserModelState.user_id == self.user_id
        ).first()

        if state:
            # Load model if available
            if state.model_blob:
                try:
                    self.model = pickle.loads(state.model_blob)
                    self.is_trained = state.is_trained
                    self.model_type = state.model_type
                    logger.debug(f"Loaded model for user {self.user_id}")
                except Exception as e:
                    logger.warning(f"Failed to load model for user {self.user_id}: {e}")
                    self._create_new_model()
            else:
                self._create_new_model()

            # Load heuristic weights if available
            if state.heuristic_weights_blob:
                try:
                    self.heuristic_weights = pickle.loads(state.heuristic_weights_blob)
                except Exception as e:
                    logger.warning(f"Failed to load heuristic weights: {e}")
        else:
            self._create_new_model()

    def _create_new_model(self):
        """Create a new ML ranking model"""
        use_ltr = settings.USE_LTR and lgb is not None

        if settings.USE_LTR and lgb is None:
            logger.warning("LightGBM not installed; using regressor model")

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

        self.is_trained = False
        logger.info(f"Created new {self.model_type} model for user {self.user_id}")

    def save_state(self):
        """Persist model state to database"""
        state = self.db.query(UserModelState).filter(
            UserModelState.user_id == self.user_id
        ).first()

        if not state:
            state = UserModelState(user_id=self.user_id)
            self.db.add(state)

        # Serialize and save
        try:
            state.model_blob = pickle.dumps(self.model)
            state.model_type = self.model_type
            state.is_trained = self.is_trained
            state.updated_at = datetime.now(timezone.utc)

            if self.heuristic_weights is not None:
                state.heuristic_weights_blob = pickle.dumps(self.heuristic_weights)

            self.db.commit()
            logger.debug(f"Saved model state for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")
            self.db.rollback()
            raise

    def rank_items(self, items: List, features: List[Dict]) -> List[Tuple]:
        """
        Rank items by ML score (or heuristic fallback).

        Args:
            items: List of Paper/Article objects
            features: List of feature dictionaries

        Returns:
            List of (item, score) tuples, sorted by score descending
        """
        if len(items) != len(features):
            raise ValueError("Items and features must have same length")

        if not items:
            return []

        # Convert features to array
        X = np.array([
            [f.get(name, 0.0) for name in self.feature_names]
            for f in features
        ])

        if not self.is_trained:
            # Use heuristic scoring
            logger.debug("Model not trained, using heuristic scoring")
            scores = np.array([
                self.calculate_weighted_heuristic_score(f)
                for f in features
            ])
        else:
            # Use ML model
            try:
                scores = self.model.predict(X)
            except Exception as e:
                logger.warning(f"ML prediction failed, falling back to heuristics: {e}")
                scores = np.array([
                    self.calculate_weighted_heuristic_score(f)
                    for f in features
                ])

        # Sort by score descending
        ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        return ranked

    def update_model(self, X: np.ndarray, y: np.ndarray, group: Optional[List[int]] = None):
        """
        Train the ML model on interaction data.

        Args:
            X: Feature matrix
            y: Labels (interaction scores)
            group: Group sizes for LTR (required if model_type is "ltr")
        """
        if self.model_type == "ltr":
            if not group:
                logger.error("LTR training requires group sizes")
                return
            y = np.array(y, dtype=int)
            self.model.fit(X, y, group=group)
        else:
            self.model.fit(X, y)

        self.is_trained = True
        self.save_state()
        logger.info(f"Updated ML model for user {self.user_id}")

    def update_heuristic_weights(self, X: np.ndarray, y: np.ndarray):
        """
        Learn heuristic feature weights via logistic regression.

        Args:
            X: Feature matrix
            y: Labels
        """
        try:
            from sklearn.linear_model import LogisticRegression

            # Convert to binary classification
            y_binary = (np.array(y) > 0.5).astype(int)

            # Check if we have both classes
            if len(np.unique(y_binary)) < 2:
                logger.warning("Need both positive and negative examples for weight learning")
                return

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X, y_binary)

            self.heuristic_weights = lr.coef_[0]
            self.save_state()
            logger.info(f"Updated heuristic weights for user {self.user_id}")
        except Exception as e:
            logger.warning(f"Could not update heuristic weights: {e}")

    def calculate_weighted_heuristic_score(self, features: Dict) -> float:
        """
        Score using learned weights if available, else fallback to impact.

        Args:
            features: Feature dictionary

        Returns:
            Score in [0, 1]
        """
        if self.heuristic_weights is None or len(self.heuristic_weights) != len(self.feature_names):
            # Fallback to impact score
            return features.get('impact', 0.0)

        # Compute weighted sum
        feature_vec = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ])
        score = np.dot(self.heuristic_weights, feature_vec)

        # Apply sigmoid to bound to [0, 1]
        score = 1.0 / (1.0 + np.exp(-score))
        return float(score)

    @staticmethod
    def calculate_impact_score(paper, include_venue: bool = True) -> float:
        """
        Heuristic quality proxy for papers.

        Static method so it can be used without user context.
        """
        score = 0.0

        title = getattr(paper, 'title', '').lower()
        abstract = getattr(paper, 'abstract', '').lower()
        text = f"{title} {abstract}"

        # Keywords (weight: 0.30)
        keyword_scores = {
            'survey': 0.15, 'review': 0.15, 'systematic review': 0.18,
            'benchmark': 0.12, 'evaluation': 0.08, 'comparison': 0.06,
            'state-of-the-art': 0.10, 'sota': 0.10, 'novel': 0.07,
            'breakthrough': 0.09, 'first': 0.06,
            'dataset': 0.10, 'corpus': 0.08, 'toolkit': 0.07, 'framework': 0.06,
            'outperform': 0.05, 'achieves': 0.04, 'improves': 0.04, 'advances': 0.05,
        }

        keyword_contribution = 0.0
        for keyword, weight in keyword_scores.items():
            if keyword in text:
                keyword_contribution = max(keyword_contribution, weight)
        score += min(keyword_contribution, 0.30)

        # Code availability (weight: 0.20)
        code_indicators = ['github', 'code available', 'gitlab', 'bitbucket', 'open source']
        if any(indicator in text for indicator in code_indicators):
            score += 0.20

        # Author signals (weight: 0.15)
        authors = getattr(paper, 'authors', [])
        num_authors = len(authors) if authors else 0
        if num_authors >= 5:
            score += 0.15
        elif num_authors >= 3:
            score += 0.12
        elif num_authors >= 2:
            score += 0.08

        # Venue signals (weight: 0.15)
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

        # Title quality (weight: 0.05)
        title_words = len(title.split())
        if 5 <= title_words <= 12:
            score += 0.05
        elif 13 <= title_words <= 15:
            score += 0.03

        # Abstract quality (weight: 0.05)
        abstract_words = len(abstract.split())
        if 100 <= abstract_words <= 300:
            score += 0.05
        elif 50 <= abstract_words < 100 or 300 < abstract_words <= 400:
            score += 0.03

        return min(score, 1.0)

    def save_training_metrics(
        self,
        train_ndcg: float,
        train_mrr: float,
        val_ndcg: Optional[float] = None,
        val_mrr: Optional[float] = None,
        interaction_count: int = 0
    ):
        """Save training metrics to the database"""
        state = self.db.query(UserModelState).filter(
            UserModelState.user_id == self.user_id
        ).first()

        if state:
            state.train_ndcg = train_ndcg
            state.train_mrr = train_mrr
            state.val_ndcg = val_ndcg
            state.val_mrr = val_mrr
            state.interaction_count_at_training = interaction_count
            state.last_trained_at = datetime.now(timezone.utc)
            self.db.commit()
