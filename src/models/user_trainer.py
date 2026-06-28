"""
Per-user model trainer for collecting interactions and retraining
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

from src.database.models import UserInteraction, Paper, Article
from src.models.feature_extractor import FeatureExtractor
from src.models.embeddings import EmbeddingManager
from src.models.evaluator import compute_ndcg, compute_mrr
from src.utils.config import settings

logger = logging.getLogger(__name__)


class UserModelTrainer:
    """
    Trainer for per-user models.

    Collects training data from user interactions and retrains
    the user's personalized recommender model.
    """

    FEATURE_NAMES = [
        "similarity", "recency", "impact", "category", "source",
        "title_length", "content_length", "readability", "has_code",
        "is_survey", "novelty", "venue", "author_reputation",
    ]

    def __init__(self, user_id: int, db: Session, embedding_manager: EmbeddingManager):
        """
        Initialize trainer for a specific user.

        Args:
            user_id: The user's database ID
            db: Database session
            embedding_manager: Embedding manager for similarity computation
        """
        self.user_id = user_id
        self.db = db
        self.embedding_manager = embedding_manager
        self.feature_extractor = FeatureExtractor()

    def record_interaction(self, item_type: str, item_id: int, interaction_type: str):
        """
        Record or update a user interaction.

        Args:
            item_type: "paper" or "article"
            item_id: Database ID of the item
            interaction_type: "saved", "viewed", or "dismissed"
        """
        now = datetime.now(timezone.utc)

        # Check for existing interaction
        existing = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == self.user_id,
            UserInteraction.item_type == item_type,
            UserInteraction.item_id == item_id,
        ).first()

        if existing:
            existing.interaction_type = interaction_type
            existing.timestamp = now
            logger.debug(f"Updated interaction for user {self.user_id}: {item_type} {item_id} -> {interaction_type}")
        else:
            interaction = UserInteraction(
                user_id=self.user_id,
                item_type=item_type,
                item_id=item_id,
                interaction_type=interaction_type,
                timestamp=now,
            )
            self.db.add(interaction)
            logger.debug(f"Recorded interaction for user {self.user_id}: {item_type} {item_id} -> {interaction_type}")

        self.db.commit()

    def get_interaction_count(self) -> int:
        """Get total number of interactions for this user"""
        return self.db.query(UserInteraction).filter(
            UserInteraction.user_id == self.user_id
        ).count()

    def _get_user_interests(self) -> List[str]:
        """Get user interests from database or use defaults"""
        from src.database.models import User
        user = self.db.query(User).filter(User.id == self.user_id).first()
        if user:
            interests = user.get_interests_list()
            if interests:
                return interests
        return settings.USER_INTERESTS

    def _get_recent_texts(self, item_type: str) -> List[str]:
        """Get recently recommended item texts for novelty calculation"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.NOVELTY_LOOKBACK_DAYS)
        texts = []

        if item_type == "paper":
            items = (
                self.db.query(Paper)
                .filter(Paper.recommended == True)
                .filter(Paper.recommended_date >= cutoff)
                .order_by(Paper.recommended_date.desc())
                .limit(settings.NOVELTY_MAX_ITEMS)
                .all()
            )
            for item in items:
                texts.append(f"{item.title} {item.abstract or ''}")
        else:
            items = (
                self.db.query(Article)
                .filter(Article.recommended == True)
                .filter(Article.recommended_date >= cutoff)
                .order_by(Article.recommended_date.desc())
                .limit(settings.NOVELTY_MAX_ITEMS)
                .all()
            )
            for item in items:
                texts.append(f"{item.title} {item.content or ''}")

        return texts

    def generate_training_data(
        self,
        min_interactions: int = 50
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate (X, y) arrays from user interactions.

        Returns (None, None) if insufficient data.
        """
        # Get user's interactions
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == self.user_id
        ).all()

        if len(interactions) < min_interactions:
            logger.info(f"User {self.user_id}: Not enough interactions ({len(interactions)} < {min_interactions})")
            return None, None

        # Get user interests
        user_interests = self._get_user_interests()

        # Get all recommended items
        recommended_papers = self.db.query(Paper).filter(Paper.recommended == True).all()
        recommended_articles = self.db.query(Article).filter(Article.recommended == True).all()

        # Build interaction score map
        interaction_scores = {}
        for interaction in interactions:
            key = (interaction.item_type, interaction.item_id)

            if interaction.interaction_type == "saved":
                score = 1.0
            elif interaction.interaction_type == "viewed":
                score = 0.6
            elif interaction.interaction_type == "dismissed":
                score = 0.0
            else:
                score = 0.5

            if key not in interaction_scores or score > interaction_scores[key]:
                interaction_scores[key] = score

        X_list = []
        y_list = []

        # Process papers
        recent_paper_texts = self._get_recent_texts("paper")
        for paper in recommended_papers:
            features = self.feature_extractor.extract_features(
                paper, "paper", self.embedding_manager, user_interests,
                recent_texts=recent_paper_texts,
            )

            X_list.append([features.get(name, 0.0) for name in self.FEATURE_NAMES])

            key = ("paper", paper.id)
            if key in interaction_scores:
                y_list.append(interaction_scores[key])
            elif settings.INCLUDE_IMPLICIT_NEGATIVES:
                if np.random.rand() <= settings.IMPLICIT_NEGATIVE_SAMPLE_RATE:
                    y_list.append(0.0)
                else:
                    X_list.pop()
            else:
                X_list.pop()

        # Process articles
        recent_article_texts = self._get_recent_texts("article")
        for article in recommended_articles:
            features = self.feature_extractor.extract_features(
                article, "article", self.embedding_manager, user_interests,
                recent_texts=recent_article_texts,
            )

            X_list.append([features.get(name, 0.0) for name in self.FEATURE_NAMES])

            key = ("article", article.id)
            if key in interaction_scores:
                y_list.append(interaction_scores[key])
            elif settings.INCLUDE_IMPLICIT_NEGATIVES:
                if np.random.rand() <= settings.IMPLICIT_NEGATIVE_SAMPLE_RATE:
                    y_list.append(0.0)
                else:
                    X_list.pop()
            else:
                X_list.pop()

        if len(X_list) == 0:
            logger.warning(f"User {self.user_id}: No training examples generated")
            return None, None

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"User {self.user_id}: Generated {len(X)} training examples")
        return X, y

    def generate_ranking_data(
        self,
        min_interactions: int = 50,
        min_group_size: int = 10
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
        """
        Generate (X, y, group_sizes) for LTR training, grouped by week.
        """
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == self.user_id
        ).all()

        if len(interactions) < min_interactions:
            logger.info(f"User {self.user_id}: Not enough interactions for LTR")
            return None, None, None

        user_interests = self._get_user_interests()

        recommended_papers = self.db.query(Paper).filter(Paper.recommended == True).all()
        recommended_articles = self.db.query(Article).filter(Article.recommended == True).all()

        interaction_scores = {}
        for interaction in interactions:
            key = (interaction.item_type, interaction.item_id)
            if interaction.interaction_type == "saved":
                score = 2
            elif interaction.interaction_type == "viewed":
                score = 1
            elif interaction.interaction_type == "dismissed":
                score = 0
            else:
                score = 1
            if key not in interaction_scores or score > interaction_scores[key]:
                interaction_scores[key] = score

        recent_paper_texts = self._get_recent_texts("paper")
        recent_article_texts = self._get_recent_texts("article")

        # Group by week
        grouped_items = {}
        for paper in recommended_papers:
            if not paper.recommended_date:
                continue
            group_key = (paper.recommended_date.isocalendar()[0], paper.recommended_date.isocalendar()[1])
            grouped_items.setdefault(group_key, []).append(("paper", paper))

        for article in recommended_articles:
            if not article.recommended_date:
                continue
            group_key = (article.recommended_date.isocalendar()[0], article.recommended_date.isocalendar()[1])
            grouped_items.setdefault(group_key, []).append(("article", article))

        if not grouped_items:
            logger.warning(f"User {self.user_id}: No grouped recommendations")
            return None, None, None

        X_list = []
        y_list = []
        group_sizes = []

        sorted_group_keys = sorted(grouped_items.keys())
        current_group_x = []
        current_group_y = []

        for group_key in sorted_group_keys:
            items = grouped_items[group_key]

            for item_type, item in items:
                recent = recent_paper_texts if item_type == "paper" else recent_article_texts
                features = self.feature_extractor.extract_features(
                    item, item_type, self.embedding_manager, user_interests,
                    recent_texts=recent,
                )

                feature_vec = [features.get(name, 0.0) for name in self.FEATURE_NAMES]

                key = (item_type, item.id)
                if key in interaction_scores:
                    current_group_x.append(feature_vec)
                    current_group_y.append(interaction_scores[key])
                elif settings.INCLUDE_IMPLICIT_NEGATIVES:
                    if np.random.rand() <= settings.IMPLICIT_NEGATIVE_SAMPLE_RATE:
                        current_group_x.append(feature_vec)
                        current_group_y.append(0)

            if len(current_group_x) >= min_group_size:
                X_list.extend(current_group_x)
                y_list.extend(current_group_y)
                group_sizes.append(len(current_group_x))
                current_group_x = []
                current_group_y = []

        # Add remaining
        if len(current_group_x) > 0:
            X_list.extend(current_group_x)
            y_list.extend(current_group_y)
            group_sizes.append(len(current_group_x))

        if len(X_list) == 0:
            logger.warning(f"User {self.user_id}: No ranking examples generated")
            return None, None, None

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"User {self.user_id}: Generated {len(X)} ranking examples in {len(group_sizes)} groups")
        return X, y, group_sizes

    def retrain_model(self, recommender, min_interactions: int = 50, use_validation: bool = True) -> bool:
        """
        Retrain the user's recommender from interactions.

        Args:
            recommender: UserRecommender instance
            min_interactions: Minimum interactions required
            use_validation: Whether to use validation split

        Returns:
            True on success, False on failure
        """
        if getattr(recommender, "model_type", "regressor") == "ltr":
            X, y, group_sizes = self.generate_ranking_data(min_interactions)
        else:
            X, y = self.generate_training_data(min_interactions)
            group_sizes = None

        if X is None or y is None:
            logger.info(f"User {self.user_id}: Skipping retraining - not enough data")
            return False

        interaction_count = self.get_interaction_count()

        # Train with validation if enough data
        if use_validation and len(X) >= 100:
            train_metrics, val_metrics = self._train_with_validation(
                recommender, X, y, group_sizes
            )
            logger.info(f"User {self.user_id}: Train NDCG@10={train_metrics['ndcg']:.3f}, MRR={train_metrics['mrr']:.3f}")
            logger.info(f"User {self.user_id}: Val NDCG@10={val_metrics['ndcg']:.3f}, MRR={val_metrics['mrr']:.3f}")

            recommender.save_training_metrics(
                train_ndcg=train_metrics['ndcg'],
                train_mrr=train_metrics['mrr'],
                val_ndcg=val_metrics['ndcg'],
                val_mrr=val_metrics['mrr'],
                interaction_count=interaction_count
            )
        else:
            # Train on all data
            recommender.update_model(X, y, group=group_sizes)
            try:
                preds = recommender.model.predict(X)
                ndcg = compute_ndcg(y, preds, k=10, group_sizes=group_sizes)
                mrr = compute_mrr(y, preds, group_sizes=group_sizes)
                logger.info(f"User {self.user_id}: Train NDCG@10={ndcg:.3f}, MRR={mrr:.3f}")

                recommender.save_training_metrics(
                    train_ndcg=ndcg,
                    train_mrr=mrr,
                    interaction_count=interaction_count
                )
            except Exception as e:
                logger.warning(f"Metric computation failed: {e}")

        # Update heuristic weights
        try:
            recommender.update_heuristic_weights(X, y)
        except Exception as e:
            logger.warning(f"Could not update heuristic weights: {e}")

        logger.info(f"User {self.user_id}: Retrained model with {len(X)} examples")
        return True

    def _train_with_validation(
        self,
        recommender,
        X: np.ndarray,
        y: np.ndarray,
        group_sizes: Optional[List[int]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """80/20 temporal split training"""
        if group_sizes:
            train_size = max(1, int(len(group_sizes) * 0.8))
            if train_size >= len(group_sizes):
                recommender.update_model(X, y, group=group_sizes)
                preds = recommender.model.predict(X)
                metrics = {
                    'ndcg': compute_ndcg(y, preds, k=10, group_sizes=group_sizes),
                    'mrr': compute_mrr(y, preds, group_sizes=group_sizes)
                }
                return metrics, metrics

            train_groups = group_sizes[:train_size]
            val_groups = group_sizes[train_size:]
            split_idx = sum(train_groups)

            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            recommender.update_model(X_train, y_train, group=train_groups)

            train_preds = recommender.model.predict(X_train)
            val_preds = recommender.model.predict(X_val)

            train_metrics = {
                'ndcg': compute_ndcg(y_train, train_preds, k=10, group_sizes=train_groups),
                'mrr': compute_mrr(y_train, train_preds, group_sizes=train_groups)
            }
            val_metrics = {
                'ndcg': compute_ndcg(y_val, val_preds, k=10, group_sizes=val_groups),
                'mrr': compute_mrr(y_val, val_preds, group_sizes=val_groups)
            }
        else:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            recommender.update_model(X_train, y_train)

            train_preds = recommender.model.predict(X_train)
            val_preds = recommender.model.predict(X_val)

            train_metrics = {
                'ndcg': compute_ndcg(y_train, train_preds, k=10),
                'mrr': compute_mrr(y_train, train_preds)
            }
            val_metrics = {
                'ndcg': compute_ndcg(y_val, val_preds, k=10),
                'mrr': compute_mrr(y_val, val_preds)
            }

        return train_metrics, val_metrics
