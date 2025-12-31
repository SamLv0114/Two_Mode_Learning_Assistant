"""
Model trainer for collecting user interactions and retraining the recommender
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from src.database.models import UserInteraction, Paper, Article
from src.models.feature_extractor import FeatureExtractor
from src.models.embeddings import EmbeddingManager
from src.utils.config import settings
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Collects training data from user interactions and retrains the model"""

    def __init__(self, db: Session, embedding_manager: EmbeddingManager):
        self.db = db
        self.embedding_manager = embedding_manager
        self.feature_extractor = FeatureExtractor()

    def record_interaction(self, item_type: str, item_id: int, interaction_type: str):
        """
        Record or update a user interaction.
        Keeps a single row per item, updating type/timestamp instead of inserting duplicates.
        """
        now = datetime.now(timezone.utc)

        existing = (
            self.db.query(UserInteraction)
            .filter(
                UserInteraction.item_type == item_type,
                UserInteraction.item_id == item_id,
            )
            .order_by(UserInteraction.timestamp.desc())
            .first()
        )

        if existing:
            existing.interaction_type = interaction_type
            existing.timestamp = now
            logger.debug(
                f"Updated interaction for {item_type} {item_id} to {interaction_type}"
            )
        else:
            interaction = UserInteraction(
                item_type=item_type,
                item_id=item_id,
                interaction_type=interaction_type,
                timestamp=now,
            )
            self.db.add(interaction)
            logger.debug(f"Recorded {interaction_type} for {item_type} {item_id}")

        self.db.commit()

    def generate_training_data(self, min_interactions: int = 50) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate training data from user interactions
        
        Args:
            min_interactions: Minimum number of interactions needed to generate training data
            
        Returns:
            Tuple of (X, y) numpy arrays, or (None, None) if not enough data
        """
        interactions = self.db.query(UserInteraction).all()

        if len(interactions) < min_interactions:
            logger.info(f"Not enough interactions ({len(interactions)} < {min_interactions}). Need more user feedback.")
            return None, None

        # Get all recommended items (shown to user)
        recommended_papers = self.db.query(Paper).filter(Paper.recommended == True).all()
        recommended_articles = self.db.query(Article).filter(Article.recommended == True).all()

        # Build interaction map: (item_type, item_id) -> best interaction score
        interaction_scores = {}
        for interaction in interactions:
            key = (interaction.item_type, interaction.item_id)
            
            # Score interactions: saved > viewed > dismissed
            if interaction.interaction_type == "saved":
                score = 1.0
            elif interaction.interaction_type == "viewed":
                score = 0.6
            elif interaction.interaction_type == "dismissed":
                score = 0.0
            else:
                score = 0.5

            # Keep highest score for each item
            if key not in interaction_scores or score > interaction_scores[key]:
                interaction_scores[key] = score

        # Generate features and labels
        X_list = []
        y_list = []

        # Process papers
        # Use recommender's feature names for papers
        paper_feature_names = ["similarity", "recency", "citations", "category", "title_length"]
        for paper in recommended_papers:
            features = self.feature_extractor.extract_paper_features(
                paper, self.embedding_manager, settings.USER_INTERESTS
            )
            
            # Convert to array matching feature_names
            X_list.append([features.get(name, 0.0) for name in paper_feature_names])

            # Get label from interactions
            key = ("paper", paper.id)
            if key in interaction_scores:
                y_list.append(interaction_scores[key])
            else:
                # No interaction = implicit dismiss (user didn't view it)
                # If recommended recently (< 1 day), treat as neutral, otherwise as dismiss
                if paper.recommended_date:
                    recommended_time = paper.recommended_date
                    if isinstance(recommended_time, datetime):
                        if recommended_time.tzinfo is None:
                            recommended_time = recommended_time.replace(tzinfo=timezone.utc)
                        else:
                            recommended_time = recommended_time.astimezone(timezone.utc)
                        
                        days_since_recommendation = (datetime.now(timezone.utc) - recommended_time).days
                        if days_since_recommendation > 1:
                            y_list.append(0.0)  # Implicit dismiss (never viewed)
                        else:
                            y_list.append(0.2)  # Weak negative (recently shown but not viewed)
                else:
                    y_list.append(0.0)  # Implicit dismiss

        # Process articles
        # Note: Articles have different features, but we'll map them to paper features for now
        # will considereparate models or unified features in future
        article_feature_names = ["similarity", "recency", "engagement", "source", "content_length"]
        for article in recommended_articles:
            features = self.feature_extractor.extract_article_features(
                article, self.embedding_manager, settings.USER_INTERESTS
            )
            
            # Map article features to paper feature format for training
            # This is a simplification - in production, consider separate models
            mapped_features = {
                "similarity": features.get("similarity", 0.0),
                "recency": features.get("recency", 0.0),
                "citations": features.get("engagement", 0.0),  # Map engagement to citations
                "category": features.get("source", 0.0),  # Map source to category
                "title_length": features.get("content_length", 0.0)  # Map content_length to title_length
            }
            X_list.append([mapped_features.get(name, 0.0) for name in paper_feature_names])

            key = ("article", article.id)
            if key in interaction_scores:
                y_list.append(interaction_scores[key])
            else:
                # No interaction = implicit dismiss (user didn't view it)
                # If recommended recently (< 1 day), treat as neutral, otherwise as dismiss
                if article.recommended_date:
                    recommended_time = article.recommended_date
                    if isinstance(recommended_time, datetime):
                        if recommended_time.tzinfo is None:
                            recommended_time = recommended_time.replace(tzinfo=timezone.utc)
                        else:
                            recommended_time = recommended_time.astimezone(timezone.utc)
                        
                        days_since_recommendation = (datetime.now(timezone.utc) - recommended_time).days
                        if days_since_recommendation > 1:
                            y_list.append(0.0)  # Implicit dismiss (never viewed)
                        else:
                            y_list.append(0.2)  # Weak negative (recently shown but not viewed)
                else:
                    y_list.append(0.0)  # Implicit dismiss

        if len(X_list) == 0:
            logger.warning("No training examples generated")
            return None, None

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Generated {len(X)} training examples from {len(interactions)} interactions")
        return X, y

    def retrain_model(self, recommender, min_interactions: int = 50):
        """
        Retrain the recommender model with collected user interactions
        
        Args:
            recommender: Recommender instance to retrain
            min_interactions: Minimum interactions needed to retrain
            
        Returns:
            True if retraining succeeded, False otherwise
        """
        X, y = self.generate_training_data(min_interactions)

        if X is None or y is None:
            logger.info("Skipping retraining - not enough training data")
            return False

        # Retrain the model
        recommender.update_model(X, y)
        logger.info(f"Successfully retrained model with {len(X)} examples")
        return True

    def get_interaction_count(self) -> int:
        """Get total number of interactions recorded"""
        return self.db.query(UserInteraction).count()

