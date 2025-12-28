"""
XGBoost recommendation model for ranking content
"""
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from src.utils.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recommender:
    """Ranking model using gradient boosting"""
    
    def __init__(self):
        self.model_path = settings.MODELS_DIR / "ranker_model.pkl"
        self.model = None
        self.feature_names = None
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create a new one"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.feature_names = data['feature_names']
                logger.info("Loaded existing ranking model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Creating new model.")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new ranking model"""
        # Simple gradient boosting regressor
        # In production, this would be trained on user interaction data
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Default feature names (should match FeatureExtractor output)
        self.feature_names = [
            "similarity", "recency", "citations", "category", "title_length"
        ]
        
        # Train on dummy data to initialize
        # In production, use real user interaction data
        X_dummy = np.random.rand(100, len(self.feature_names))
        y_dummy = np.random.rand(100)
        self.model.fit(X_dummy, y_dummy)
        
        self._save_model()
        logger.info("Created new ranking model")
    
    def _save_model(self):
        """Save the model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names
                }, f)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def rank_items(self, items: List, features: List[Dict]) -> List[tuple]:
        """
        Rank items by their features
        
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
        
        # Predict scores
        scores = self.model.predict(X)
        
        # Sort by score (descending)
        ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def update_model(self, X: np.ndarray, y: np.ndarray):
        """
        Update model with new training data
        
        Args:
            X: Feature matrix
            y: Target scores (e.g., from user interactions)
        """
        self.model.fit(X, y)
        self._save_model()
        logger.info("Updated ranking model with new data")

