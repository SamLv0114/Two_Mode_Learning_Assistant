"""
Unified ranking system combining heuristic scoring and ML personalization
"""
import pickle
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
    """Combines heuristic scoring (quality baseline) with ML personalization."""

    FEATURE_NAMES = [
        "similarity", "recency", "impact", "category", "source",
        "title_length", "content_length", "readability", "has_code",
        "is_survey", "novelty", "venue", "author_reputation",
    ]

    def __init__(self):
        self.model_path = settings.MODELS_DIR / "ranker_model.pkl"
        self.heuristic_weights_path = settings.MODELS_DIR / "heuristic_weights.pkl"
        self.model = None
        self.feature_names = None
        self.model_type = None
        self.heuristic_weights = None  # Learned weights for heuristic scoring
        self._load_or_create_model()
        self._load_heuristic_weights()
    
    # ============================================================================
    # ML MODEL (Learns from user interactions)
    # ============================================================================
    
    def _load_or_create_model(self):
        """Load existing ML model or create a new one"""
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
                if self.feature_names != self.FEATURE_NAMES:
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
        
        self.feature_names = list(self.FEATURE_NAMES)

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
        """Rank items by ML score (or heuristic fallback). Returns (item, score) pairs descending."""
        if len(items) != len(features):
            raise ValueError("Items and features must have same length")

        # Convert features to array
        X = np.array([[f.get(name, 0.0) for name in self.feature_names] for f in features])

        # If model is not trained, use heuristic scoring
        if not self.is_trained:
            logger.debug("Model not yet trained, using heuristic scoring")
            scores = []
            for item, feat in zip(items, features):
                # Use learned heuristic weights if available, else fallback to impact
                score = self.calculate_weighted_heuristic_score(feat)
                scores.append(score)
            scores = np.array(scores)
        else:
            # Predict scores using ML model
            scores = self.model.predict(X)

        # Sort by score (descending)
        ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)

        return ranked
    
    def update_model(self, X: np.ndarray, y: np.ndarray, group: Optional[List[int]] = None):
        """Train the ML model on interaction data. group is required for LTR."""
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
        """Heuristic quality proxy based on keywords, code, authors, venue, recency, and text quality."""
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
            except Exception:
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
    
    def _load_heuristic_weights(self):
        """Load learned heuristic weights from disk"""
        if self.heuristic_weights_path.exists():
            try:
                with open(self.heuristic_weights_path, 'rb') as f:
                    self.heuristic_weights = pickle.load(f)
                logger.info(f"Loaded learned heuristic weights: {len(self.heuristic_weights)} features")
            except Exception as e:
                logger.warning(f"Could not load heuristic weights: {e}")
                self.heuristic_weights = None
        else:
            self.heuristic_weights = None

    def _save_heuristic_weights(self):
        """Save learned heuristic weights to disk"""
        try:
            with open(self.heuristic_weights_path, 'wb') as f:
                pickle.dump(self.heuristic_weights, f)
            logger.info("Saved learned heuristic weights")
        except Exception as e:
            logger.error(f"Error saving heuristic weights: {e}")

    def update_heuristic_weights(self, X: np.ndarray, y: np.ndarray):
        """Learn heuristic feature weights via logistic regression on interaction data."""
        try:
            from sklearn.linear_model import LogisticRegression

            # Convert to binary classification
            y_binary = (y > 0.5).astype(int)

            # Train logistic regression to learn feature weights
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X, y_binary)

            # Store coefficients as weights
            self.heuristic_weights = lr.coef_[0]
            self._save_heuristic_weights()

            logger.info(f"Updated heuristic weights from {len(X)} examples")
        except Exception as e:
            logger.warning(f"Could not update heuristic weights: {e}")

    def calculate_weighted_heuristic_score(self, features: Dict) -> float:
        """Score using learned weights if available, else fallback to impact."""
        if self.heuristic_weights is None or len(self.heuristic_weights) != len(self.feature_names):
            # Fallback to impact score
            return features.get('impact', 0.0)

        # Compute weighted sum of features
        feature_vec = np.array([features.get(name, 0.0) for name in self.feature_names])
        score = np.dot(self.heuristic_weights, feature_vec)

        # Apply sigmoid to bound to [0, 1]
        score = 1.0 / (1.0 + np.exp(-score))

        return float(score)


