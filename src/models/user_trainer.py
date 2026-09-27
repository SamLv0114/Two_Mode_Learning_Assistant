"""
Per-user model trainer for collecting interactions and retraining
"""
import json
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

from src.database.models import UserInteraction, UserModelState, Paper, UserPaperRecommendation, upsert_user_interaction, count_labeled_paper_interactions
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
        "similarity", "recency", "impact", "category",
        "title_length", "content_length", "readability", "has_code",
        "is_survey", "novelty", "citation_velocity",
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
        upsert_user_interaction(self.db, self.user_id, item_type, item_id, interaction_type)

    def get_interaction_count(self) -> int:
        """Count paper feedback eligible for the paper-only ranker."""
        return count_labeled_paper_interactions(self.db, self.user_id)

    def _get_user_interests(self) -> List[str]:
        """Get user interests from database or use defaults"""
        from src.database.models import User
        user = self.db.query(User).filter(User.id == self.user_id).first()
        if user:
            interests = user.get_interests_list()
            if interests:
                return interests
        return settings.USER_INTERESTS

    def _get_recent_texts(self) -> List[str]:
        """Get recently recommended paper texts for novelty calculation."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.NOVELTY_LOOKBACK_DAYS)
        texts = []
        items = (
            self.db.query(Paper)
            .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
            .filter(
                UserPaperRecommendation.user_id == self.user_id,
                UserPaperRecommendation.recommended_date >= cutoff,
            )
            .order_by(UserPaperRecommendation.recommended_date.desc())
            .limit(settings.NOVELTY_MAX_ITEMS)
            .all()
        )
        for item in items:
            texts.append(f"{item.title} {item.abstract or ''}")

        return texts

    def _labeled_papers(self):
        """One real, explicit feedback label per paper, including older feeds.

        UserPaperRecommendation is only the latest five items, so it cannot be
        used as the training population. Cold-start cards also count because
        their Save/Dismiss events are explicit feedback on real Paper rows.
        """
        rows = (
            self.db.query(Paper, UserInteraction)
            .join(UserInteraction, Paper.id == UserInteraction.item_id)
            .filter(UserInteraction.user_id == self.user_id,
                    UserInteraction.item_type == "paper",
                    UserInteraction.interaction_type.in_(["saved", "viewed", "dismissed"]))
            .all()
        )
        by_paper = {}
        for paper, interaction in rows:
            previous = by_paper.get(paper.id)
            when = (interaction.timestamp or datetime.min).replace(tzinfo=None)
            previous_when = (previous[1].timestamp or datetime.min).replace(tzinfo=None) if previous else datetime.min
            if previous is None or when > previous_when:
                by_paper[paper.id] = (paper, interaction)
        return sorted(by_paper.values(), key=lambda row: (
            (row[1].timestamp or datetime.min).replace(tzinfo=None), row[0].id))

    def generate_training_data(
        self,
        min_interactions: int = 50
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate (X, y) arrays from user interactions.

        Returns (None, None) if insufficient data.
        """
        labeled = self._labeled_papers()
        if len(labeled) < min_interactions:
            logger.info(f"User {self.user_id}: Only {len(labeled)} labeled papers; need {min_interactions}")
            return None, None
        user_interests = self._get_user_interests()
        X_list = []
        y_list = []
        recent_paper_texts = self._get_recent_texts()
        labels = {"saved": 1.0, "viewed": 0.6, "dismissed": 0.0}
        for paper, interaction in labeled:
            features = self.feature_extractor.extract_features(
                paper, self.embedding_manager, user_interests,
                recent_texts=recent_paper_texts,
            )
            X_list.append([features.get(name, 0.0) for name in self.FEATURE_NAMES])
            y_list.append(labels[interaction.interaction_type])
        if len(set(y_list)) < 2:
            logger.warning("User %s: Training labels have no variation", self.user_id)
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
        Generate LTR examples from actual paper feedback, grouped by feedback week.
        """
        labeled = self._labeled_papers()
        if len(labeled) < min_interactions:
            logger.info("User %s: Only %s labeled papers; need %s", self.user_id, len(labeled), min_interactions)
            return None, None, None
        user_interests = self._get_user_interests()
        recent_paper_texts = self._get_recent_texts()
        grouped_items = {}
        labels = {"saved": 2, "viewed": 1, "dismissed": 0}
        for paper, interaction in labeled:
            when = interaction.timestamp or datetime.now(timezone.utc)
            week = when.isocalendar()
            grouped_items.setdefault((week.year, week.week), []).append((paper, interaction))
        X_list = []
        y_list = []
        group_sizes = []
        current_group_x = []
        current_group_y = []
        for group_key in sorted(grouped_items):
            for item, interaction in grouped_items[group_key]:
                features = self.feature_extractor.extract_features(
                    item, self.embedding_manager, user_interests,
                    recent_texts=recent_paper_texts,
                )
                feature_vec = [features.get(name, 0.0) for name in self.FEATURE_NAMES]
                current_group_x.append(feature_vec)
                current_group_y.append(labels[interaction.interaction_type])
            if len(current_group_x) >= min_group_size:
                X_list.extend(current_group_x)
                y_list.extend(current_group_y)
                group_sizes.append(len(current_group_x))
                current_group_x = []
                current_group_y = []

        if len(current_group_x) > 0:
            X_list.extend(current_group_x)
            y_list.extend(current_group_y)
            if group_sizes:
                group_sizes[-1] += len(current_group_x)
            else:
                group_sizes.append(len(current_group_x))
        if len(set(y_list)) < 2:
            logger.warning("User %s: LTR labels have no variation", self.user_id)
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
            if val_metrics is not None:
                logger.info(f"User {self.user_id}: Val NDCG@10={val_metrics['ndcg']:.3f}, MRR={val_metrics['mrr']:.3f}")

            recommender.save_training_metrics(
                train_ndcg=train_metrics['ndcg'],
                train_mrr=train_metrics['mrr'],
                val_ndcg=val_metrics['ndcg'] if val_metrics else None,
                val_mrr=val_metrics['mrr'] if val_metrics else None,
                interaction_count=interaction_count
            )
            # Validation scores describe the held-out split. Fit the model
            # serving the feed on all eligible feedback after measuring them.
            if val_metrics is not None:
                recommender.update_model(X, y, group=group_sizes)
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

        state = self.db.query(UserModelState).filter(UserModelState.user_id == self.user_id).first()
        if state:
            counts = {"dismissed": int(np.sum(y == 0)),
                      "viewed": int(np.sum(y == (1 if group_sizes else 0.6))),
                      "saved": int(np.sum(y == (2 if group_sizes else 1.0)))}
            state.training_sample_count = len(X)
            state.training_label_counts = json.dumps(counts)
            state.interaction_count_at_training = interaction_count
            state.last_trained_at = datetime.now(timezone.utc)
            self.db.commit()

        logger.info(f"User {self.user_id}: Retrained model with {len(X)} examples")
        return True

    def _train_with_validation(
        self,
        recommender,
        X: np.ndarray,
        y: np.ndarray,
        group_sizes: Optional[List[int]] = None
    ) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
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
                # One temporal group cannot provide a held-out validation set.
                return metrics, None

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
