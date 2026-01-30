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
from src.models.evaluator import compute_ndcg, compute_mrr
from src.utils.config import settings
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Collects training data from user interactions and retrains the model."""

    FEATURE_NAMES = [
        "similarity", "recency", "impact", "category", "source",
        "title_length", "content_length", "readability", "has_code",
        "is_survey", "novelty", "venue", "author_reputation",
    ]

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

    def _get_recent_texts(self, item_type: str) -> List[str]:
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

    def generate_training_data(self, min_interactions: int = 50) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate (X, y) arrays from interactions, or (None, None) if insufficient data."""
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
        recent_paper_texts = self._get_recent_texts("paper")
        for paper in recommended_papers:
            features = self.feature_extractor.extract_features(
                paper, "paper", self.embedding_manager, settings.USER_INTERESTS,
                recent_texts=recent_paper_texts,
            )
            
            # Convert to array matching feature_names
            X_list.append([features.get(name, 0.0) for name in self.FEATURE_NAMES])

            # Get label from interactions
            key = ("paper", paper.id)
            if key in interaction_scores:
                y_list.append(interaction_scores[key])
            elif settings.INCLUDE_IMPLICIT_NEGATIVES:
                # Position bias correction
                position = len(y_list)
                position_factor = 0.5 + 0.5 * min(position / 20.0, 1.0)
                adjusted_sample_rate = settings.IMPLICIT_NEGATIVE_SAMPLE_RATE * position_factor

                if np.random.rand() <= adjusted_sample_rate:
                    y_list.append(0.0)
                else:
                    X_list.pop()
            else:
                X_list.pop()

        # Process articles
        recent_article_texts = self._get_recent_texts("article")
        for article in recommended_articles:
            features = self.feature_extractor.extract_features(
                article, "article", self.embedding_manager, settings.USER_INTERESTS,
                recent_texts=recent_article_texts,
            )

            X_list.append([features.get(name, 0.0) for name in self.FEATURE_NAMES])

            key = ("article", article.id)
            if key in interaction_scores:
                y_list.append(interaction_scores[key])
            elif settings.INCLUDE_IMPLICIT_NEGATIVES:
                # Position bias correction
                position = len(y_list)
                position_factor = 0.5 + 0.5 * min(position / 20.0, 1.0)
                adjusted_sample_rate = settings.IMPLICIT_NEGATIVE_SAMPLE_RATE * position_factor

                if np.random.rand() <= adjusted_sample_rate:
                    y_list.append(0.0)
                else:
                    X_list.pop()
            else:
                X_list.pop()

        if len(X_list) == 0:
            logger.warning("No training examples generated")
            return None, None

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Generated {len(X)} training examples from {len(interactions)} interactions")
        return X, y

    def generate_ranking_data(
        self,
        min_interactions: int = 50,
        min_group_size: int = 10
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
        """Generate (X, y, group_sizes) for LTR training, grouped by week."""
        interactions = self.db.query(UserInteraction).all()

        if len(interactions) < min_interactions:
            logger.info(f"Not enough interactions ({len(interactions)} < {min_interactions}). Need more user feedback.")
            return None, None, None

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

        # Group by week instead of day for larger groups
        grouped_items = {}
        for paper in recommended_papers:
            if not paper.recommended_date:
                continue
            # Group by ISO week (year, week_number)
            group_key = (paper.recommended_date.isocalendar()[0], paper.recommended_date.isocalendar()[1])
            grouped_items.setdefault(group_key, []).append(("paper", paper))
        for article in recommended_articles:
            if not article.recommended_date:
                continue
            group_key = (article.recommended_date.isocalendar()[0], article.recommended_date.isocalendar()[1])
            grouped_items.setdefault(group_key, []).append(("article", article))

        if not grouped_items:
            logger.warning("No grouped recommendations available for ranking data")
            return None, None, None

        X_list = []
        y_list = []
        group_sizes = []

        # Process groups in chronological order
        sorted_group_keys = sorted(grouped_items.keys())

        current_group_x = []
        current_group_y = []

        for group_key in sorted_group_keys:
            items = grouped_items[group_key]

            for item_type, item in items:
                recent = recent_paper_texts if item_type == "paper" else recent_article_texts
                features = self.feature_extractor.extract_features(
                    item, item_type, self.embedding_manager, settings.USER_INTERESTS,
                    recent_texts=recent,
                )

                feature_vec = [features.get(name, 0.0) for name in self.FEATURE_NAMES]

                key = (item_type, item.id)
                if key in interaction_scores:
                    current_group_x.append(feature_vec)
                    current_group_y.append(interaction_scores[key])
                elif settings.INCLUDE_IMPLICIT_NEGATIVES:
                    # Position bias correction: items at top positions are sampled less often
                    # to avoid learning "top position = good quality"
                    position_in_group = len(current_group_x)
                    # Decay sample rate: top items sampled at 50% rate, bottom at 100%
                    position_factor = 0.5 + 0.5 * min(position_in_group / 20.0, 1.0)
                    adjusted_sample_rate = settings.IMPLICIT_NEGATIVE_SAMPLE_RATE * position_factor

                    if np.random.rand() <= adjusted_sample_rate:
                        current_group_x.append(feature_vec)
                        current_group_y.append(0)

            # Check if current group meets minimum size
            if len(current_group_x) >= min_group_size:
                # Finalize this group
                X_list.extend(current_group_x)
                y_list.extend(current_group_y)
                group_sizes.append(len(current_group_x))
                current_group_x = []
                current_group_y = []
            # Otherwise, continue accumulating for next group

        # Add remaining items as final group if any exist
        if len(current_group_x) > 0:
            X_list.extend(current_group_x)
            y_list.extend(current_group_y)
            group_sizes.append(len(current_group_x))

        if len(X_list) == 0:
            logger.warning("No ranking examples generated")
            return None, None, None

        X = np.array(X_list)
        y = np.array(y_list)

        # Log group size statistics
        if group_sizes:
            avg_group_size = sum(group_sizes) / len(group_sizes)
            min_size = min(group_sizes)
            max_size = max(group_sizes)
            logger.info(f"Generated {len(X)} ranking examples across {len(group_sizes)} groups")
            logger.info(f"Group size stats: min={min_size}, max={max_size}, avg={avg_group_size:.1f}")

        return X, y, group_sizes

    def retrain_model(self, recommender, min_interactions: int = 50, use_validation: bool = True):
        """Retrain recommender from interactions. Returns True on success."""
        if getattr(recommender, "model_type", "regressor") == "ltr":
            X, y, group_sizes = self.generate_ranking_data(min_interactions)
        else:
            X, y = self.generate_training_data(min_interactions)
            group_sizes = None

        if X is None or y is None:
            logger.info("Skipping retraining - not enough training data")
            return False

        # Use temporal validation if enabled and we have enough data
        if use_validation and len(X) >= 100:
            train_metrics, val_metrics = self._train_with_validation(
                recommender, X, y, group_sizes
            )
            logger.info(f"Train metrics: NDCG@10={train_metrics['ndcg']:.3f}, MRR={train_metrics['mrr']:.3f}")
            logger.info(f"Val metrics: NDCG@10={val_metrics['ndcg']:.3f}, MRR={val_metrics['mrr']:.3f}")
        else:
            # Train on all data
            recommender.update_model(X, y, group=group_sizes)
            try:
                preds = recommender.model.predict(X)
                ndcg = compute_ndcg(y, preds, k=10, group_sizes=group_sizes)
                mrr = compute_mrr(y, preds, group_sizes=group_sizes)
                logger.info(f"Train metrics (no validation): NDCG@10={ndcg:.3f}, MRR={mrr:.3f}")
            except Exception as e:
                logger.warning(f"Metric computation failed: {e}")

        # Learn heuristic weights from the same interaction data
        # This improves cold-start recommendations
        try:
            recommender.update_heuristic_weights(X, y)
        except Exception as e:
            logger.warning(f"Could not update heuristic weights: {e}")

        logger.info(f"Successfully retrained model with {len(X)} examples")
        return True

    def get_interaction_count(self) -> int:
        """Get total number of interactions recorded"""
        return self.db.query(UserInteraction).count()

    def _train_with_validation(
        self,
        recommender,
        X: np.ndarray,
        y: np.ndarray,
        group_sizes: Optional[List[int]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """80/20 temporal split; returns (train_metrics, val_metrics) dicts."""
        if group_sizes:
            # For LTR, split by groups to maintain group integrity
            train_size = max(1, int(len(group_sizes) * 0.8))
            if train_size >= len(group_sizes):
                # Not enough groups to split — train on all, report train metrics as both
                recommender.update_model(X, y, group=group_sizes)
                preds = recommender.model.predict(X)
                metrics = {
                    'ndcg': compute_ndcg(y, preds, k=10, group_sizes=group_sizes),
                    'mrr': compute_mrr(y, preds, group_sizes=group_sizes)
                }
                return metrics, metrics
            train_groups = group_sizes[:train_size]
            val_groups = group_sizes[train_size:]

            # Calculate split index
            split_idx = sum(train_groups)

            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train on training set
            recommender.update_model(X_train, y_train, group=train_groups)

            # Evaluate on both sets
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
            # For regression, simple temporal split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train on training set
            recommender.update_model(X_train, y_train)

            # Evaluate on both sets
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

