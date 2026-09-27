"""SQLite regression for Feed training and serving contracts."""
import tempfile
import sqlite3
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    Base, Paper, User, UserInteraction, UserModelState,
    UserPaperRecommendation, upsert_user_interaction,
)
from src.models.user_recommender import UserRecommender
from src.models.user_trainer import UserModelTrainer
from src.pipelines.daily_feed import DailyFeedPipeline


class DeterministicRanker:
    """Portable estimator for testing the training/serving contract."""
    def fit(self, X, y, group):
        self.sample_count = len(X)
        self.labels = tuple(y)
        self.groups = tuple(group)
        return self

    def predict(self, X):
        return X[:, 0]


class TrackingRanker(DeterministicRanker):
    def __init__(self):
        self.fit_sizes = []

    def fit(self, X, y, group):
        self.fit_sizes.append(len(X))
        return super().fit(X, y, group)


class MLFeedEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'feed.db'}",
                                    connect_args={"check_same_thread": False, "timeout": 10})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        user = User(email="reader@example.com", hashed_password="test")
        self.db.add(user)
        self.db.flush()
        self.user_id = user.id
        papers = [Paper(arxiv_id=f"2609.{i:05d}", title=f"Paper {i} about ranking",
                        abstract="Ranking methods", authors="A. Author", citation_count=i,
                        published_date=datetime.now(timezone.utc)) for i in range(55)]
        self.db.add_all(papers)
        self.db.flush()
        self.papers = papers
        now = datetime.now(timezone.utc)
        for i, paper in enumerate(papers[:50]):
            self.db.add(UserInteraction(user_id=self.user_id, item_type="paper", item_id=paper.id,
                                        interaction_type=("saved", "viewed", "dismissed")[i % 3],
                                        timestamp=now - timedelta(minutes=50-i)))
        # The live recommendation table intentionally holds only five papers.
        for i, paper in enumerate(papers[45:50]):
            self.db.add(UserPaperRecommendation(user_id=self.user_id, paper_id=paper.id,
                                                recommended_date=now, rank=i+1))
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        self.temp.cleanup()

    @staticmethod
    def _features(paper, *_args, **_kwargs):
        return {"impact": paper.citation_count / 100, "similarity": paper.citation_count / 100,
                "recency": 0.5, "category": 0.5, "title_length": 0.5,
                "content_length": 0.5, "readability": 0.5, "has_code": 0,
                "is_survey": 0, "novelty": 0.5, "citation_velocity": 0}

    def test_history_trains_model_and_feed_records_actual_serving_path(self):
        trainer = UserModelTrainer(self.user_id, self.db, SimpleNamespace())
        trainer._get_recent_texts = lambda: []
        trainer.feature_extractor = SimpleNamespace(extract_features=self._features)
        X, y, groups = trainer.generate_ranking_data(min_interactions=50)
        self.assertEqual(len(X), 50)
        self.assertEqual(sum(groups), 50)
        self.assertEqual({int(label): int((y == label).sum()) for label in set(y)},
                         {0: 16, 1: 17, 2: 17})

        recommender = UserRecommender(self.user_id, self.db)
        recommender.model = DeterministicRanker()
        recommender.model_type = "ltr"
        self.assertTrue(trainer.retrain_model(recommender, min_interactions=50, use_validation=False))
        state = self.db.query(UserModelState).filter_by(user_id=self.user_id).one()
        self.assertTrue(state.is_trained)
        self.assertEqual(state.training_sample_count, 50)
        self.assertEqual(state.interaction_count_at_training, 50)

        pipeline = DailyFeedPipeline.__new__(DailyFeedPipeline)
        pipeline.user_id = self.user_id
        pipeline.db = self.db
        pipeline.recommender = recommender
        pipeline.trainer = trainer
        pipeline.feature_extractor = SimpleNamespace(extract_features=self._features)
        pipeline.embedding_manager = SimpleNamespace(add_paper=lambda *_args: None)
        pipeline._get_recent_item_texts = lambda _kind: []
        pipeline._ranking_source = "heuristic"
        pipeline._ranking_fallback_reason = None
        candidates = [pipeline._db_paper_to_paperdata(p) for p in self.papers[50:55]]
        selected = pipeline._rank_and_select(candidates, 5, "paper", [], use_ml=True)
        self.assertEqual(pipeline._ranking_source, "ml")
        pipeline._store_results(selected, [], mode="latest")
        output = pipeline._format_output(selected, [])
        self.assertTrue(output["used_ml_ranking"])
        self.assertTrue(pipeline._todays_feed()["used_ml_ranking"])

        class BrokenModel:
            def predict(self, _features):
                raise RuntimeError("predict unavailable")

        recommender.model = BrokenModel()
        pipeline._rank_and_select(candidates, 5, "paper", [], use_ml=True)
        self.assertEqual(pipeline._ranking_source, "heuristic")
        self.assertIn("prediction_failed", pipeline._ranking_fallback_reason)

        # No new interaction: the same dataset cannot be retrained on every Feed.
        calls = []
        trainer.retrain_model = lambda *_args, **_kwargs: calls.append(1)
        pipeline._maybe_retrain()
        self.assertEqual(calls, [])
        self.db.add(UserInteraction(user_id=self.user_id, item_type="paper",
                                    item_id=self.papers[50].id, interaction_type="saved"))
        self.db.commit()
        pipeline._maybe_retrain()  # New data, but still inside cooldown.
        self.assertEqual(calls, [])
        state.last_trained_at = datetime.now(timezone.utc) - timedelta(days=2)
        self.db.commit()
        pipeline._maybe_retrain()
        self.assertEqual(calls, [1])

    def test_atomic_upsert_keeps_one_row_under_concurrent_clicks(self):
        item_id = self.papers[0].id
        barrier = threading.Barrier(2)
        errors = []

        def click(label):
            db = self.Session()
            try:
                barrier.wait(timeout=5)
                upsert_user_interaction(db, self.user_id, "paper", item_id, label)
            except Exception as exc:
                errors.append(exc)
            finally:
                db.close()

        threads = [threading.Thread(target=click, args=(label,)) for label in ("saved", "dismissed")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        rows = self.db.query(UserInteraction).filter_by(user_id=self.user_id,
                                                        item_type="paper", item_id=item_id).all()
        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0].interaction_type, ("saved", "dismissed"))

    def test_failed_training_is_not_repeated_without_new_feedback(self):
        trainer = UserModelTrainer(self.user_id, self.db, SimpleNamespace())
        recommender = UserRecommender(self.user_id, self.db)
        pipeline = DailyFeedPipeline.__new__(DailyFeedPipeline)
        pipeline.user_id = self.user_id
        pipeline.db = self.db
        pipeline.trainer = trainer
        pipeline.recommender = recommender
        calls = []
        trainer.retrain_model = lambda *_args, **_kwargs: calls.append(1) or False

        pipeline._maybe_retrain()
        pipeline._maybe_retrain()
        self.assertEqual(calls, [1])
        state = self.db.query(UserModelState).filter_by(user_id=self.user_id).one()
        self.assertEqual(state.interaction_count_at_attempt, 50)

        self.db.add(UserInteraction(user_id=self.user_id, item_type="paper",
                                    item_id=self.papers[50].id, interaction_type="saved"))
        self.db.commit()
        pipeline._maybe_retrain()
        self.assertEqual(calls, [1, 1])

    def test_validation_refits_served_model_on_all_historical_feedback(self):
        older = datetime.now(timezone.utc) - timedelta(days=15)
        extra = self.papers[50:55] + [
            Paper(arxiv_id=f"2608.{i:05d}", title=f"Older paper {i}",
                  abstract="Ranking methods", authors="A. Author", citation_count=i,
                  published_date=older) for i in range(45)
        ]
        self.db.add_all(extra)
        self.db.flush()
        for i, paper in enumerate(extra):
            self.db.add(UserInteraction(user_id=self.user_id, item_type="paper", item_id=paper.id,
                                        interaction_type=("saved", "viewed", "dismissed")[i % 3],
                                        timestamp=older + timedelta(minutes=i)))
        self.db.commit()

        trainer = UserModelTrainer(self.user_id, self.db, SimpleNamespace())
        trainer._get_recent_texts = lambda: []
        trainer.feature_extractor = SimpleNamespace(extract_features=self._features)
        recommender = UserRecommender(self.user_id, self.db)
        ranker = TrackingRanker()
        recommender.model = ranker
        recommender.model_type = "ltr"
        self.assertTrue(trainer.retrain_model(recommender, min_interactions=50))
        self.assertEqual(ranker.fit_sizes, [50, 100])
        self.assertEqual(ranker.sample_count, 100)
        state = self.db.query(UserModelState).filter_by(user_id=self.user_id).one()
        self.assertEqual(state.training_sample_count, 100)
        self.assertIsNotNone(state.val_ndcg)

    def test_single_temporal_group_does_not_fake_validation(self):
        now = datetime(2026, 9, 26, 13, tzinfo=timezone.utc)
        for interaction in self.db.query(UserInteraction).filter_by(user_id=self.user_id).all():
            interaction.timestamp = now - timedelta(minutes=50)
        extra = self.papers[50:55] + [
            Paper(arxiv_id=f"2607.{i:05d}", title=f"Same week paper {i}",
                  abstract="Ranking methods", authors="A. Author", citation_count=i,
                  published_date=now) for i in range(45)
        ]
        self.db.add_all(extra)
        self.db.flush()
        for i, paper in enumerate(extra):
            self.db.add(UserInteraction(user_id=self.user_id, item_type="paper", item_id=paper.id,
                                        interaction_type=("saved", "viewed", "dismissed")[i % 3],
                                        timestamp=now - timedelta(minutes=i)))
        self.db.commit()

        trainer = UserModelTrainer(self.user_id, self.db, SimpleNamespace())
        trainer._get_recent_texts = lambda: []
        trainer.feature_extractor = SimpleNamespace(extract_features=self._features)
        recommender = UserRecommender(self.user_id, self.db)
        ranker = TrackingRanker()
        recommender.model = ranker
        recommender.model_type = "ltr"
        self.assertTrue(trainer.retrain_model(recommender, min_interactions=50))
        self.assertEqual(ranker.fit_sizes, [100])
        state = self.db.query(UserModelState).filter_by(user_id=self.user_id).one()
        self.assertIsNone(state.val_ndcg)

    def test_existing_duplicate_rows_are_migrated_to_unique_index(self):
        legacy_path = Path(self.temp.name) / "legacy.db"
        with sqlite3.connect(legacy_path) as conn:
            conn.execute("CREATE TABLE user_interactions (id INTEGER PRIMARY KEY, user_id INTEGER, "
                         "item_type VARCHAR(20), item_id INTEGER, interaction_type VARCHAR(20), timestamp DATETIME)")
            conn.execute("CREATE INDEX ix_user_item ON user_interactions (user_id, item_type, item_id)")
            conn.execute("INSERT INTO user_interactions VALUES (1, 7, 'paper', 9, 'viewed', '2026-09-27 00:00:00')")
            conn.execute("INSERT INTO user_interactions VALUES (2, 7, 'paper', 9, 'saved', '2026-09-26 00:00:00')")
        legacy_engine = create_engine(f"sqlite:///{legacy_path}")
        try:
            with patch("src.database.models.engine", legacy_engine):
                from src.database.models import init_db
                init_db()
            with sqlite3.connect(legacy_path) as conn:
                rows = conn.execute("SELECT id, interaction_type FROM user_interactions").fetchall()
                self.assertEqual(rows, [(1, "viewed")])
                self.assertEqual(conn.execute("PRAGMA index_list(user_interactions)").fetchone()[2], 1)
        finally:
            legacy_engine.dispose()


if __name__ == "__main__":
    unittest.main()
