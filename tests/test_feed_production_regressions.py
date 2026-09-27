"""Regressions for failures observed in the deployed Feed API."""

import unittest
from types import SimpleNamespace

import numpy as np
from psycopg2.extensions import adapt

from src.database.models import Paper, UserPaperRecommendation
from src.api.routers.feed import _materialize_refined_papers
from src.pipelines.daily_feed import DailyFeedPipeline


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *args):
        return self

    def order_by(self, *args):
        return self

    def limit(self, *args):
        return self

    def join(self, *args):
        return self

    def all(self):
        return self.rows

    def delete(self, **kwargs):
        return 0

    def first(self):
        return self.rows[0] if self.rows else None


class FeedProductionRegressionTests(unittest.TestCase):
    def test_refined_paper_keeps_fields_required_by_the_card(self):
        paper = SimpleNamespace(
            id=42, arxiv_id="2609.00042", title="Test paper", abstract="Full abstract",
            authors="A. Author", categories="cs.LG", published_date=None,
            arxiv_url="https://arxiv.org/abs/2609.00042",
            pdf_url="https://arxiv.org/pdf/2609.00042",
            citation_count=10, heuristic_impact_score=0.7,
        )
        db = SimpleNamespace(query=lambda model: _Query([paper]))

        result = _materialize_refined_papers(db, [{
            "arxiv_id": paper.arxiv_id, "title": paper.title,
            "relevance_score": np.float64(0.42), "summary": "Why it matters",
        }])

        self.assertEqual(result[0].id, 42)
        self.assertEqual(result[0].arxiv_url, paper.arxiv_url)
        self.assertEqual(result[0].summary, "Why it matters")
        self.assertIs(type(result[0].relevance_score), float)

    def test_numpy_ranking_score_is_native_float_before_database_storage(self):
        item = SimpleNamespace(relevance_score=None)
        pipeline = DailyFeedPipeline.__new__(DailyFeedPipeline)
        pipeline.feature_extractor = SimpleNamespace(extract_features=lambda *args, **kwargs: {})
        pipeline.embedding_manager = object()
        pipeline.recommender = SimpleNamespace(rank_items=lambda items, features: [(item, np.float64(0.42))])
        pipeline._get_recent_item_texts = lambda item_type: []

        selected = pipeline._rank_and_select([item], 1, "paper", [], use_ml=True)

        self.assertEqual(selected, [item])
        self.assertIs(type(item.relevance_score), float)
        self.assertNotIn(b"np.float64", adapt(item.relevance_score).getquoted())

    def test_bulk_insert_receives_native_score_even_if_source_is_numpy(self):
        paper = SimpleNamespace(id=42, arxiv_id="2609.00042", citation_count=10)
        added = []
        pipeline = DailyFeedPipeline.__new__(DailyFeedPipeline)
        pipeline.db = SimpleNamespace(
            query=lambda model: _Query([paper] if model is Paper else []),
            add=added.append,
            commit=lambda: None,
        )
        pipeline.user_id = 1
        pipeline.recommender = SimpleNamespace(calculate_impact_score=lambda item: 0.5)
        pipeline.embedding_manager = SimpleNamespace(add_paper=lambda *args: None)
        item = SimpleNamespace(
            arxiv_id=paper.arxiv_id, title="Test paper", abstract="abstract",
            arxiv_url="https://arxiv.org/abs/2609.00042", published_date=None,
            citation_count=10, personalized_summary="summary", relevance_score=np.float64(0.42),
        )

        pipeline._store_results([item], [], mode="recommended", time_window_days=7)

        recommendation = next(row for row in added if isinstance(row, UserPaperRecommendation))
        self.assertIs(type(recommendation.relevance_score), float)
        self.assertNotIn(b"np.float64", adapt(recommendation.relevance_score).getquoted())

    def test_coldstart_mmr_pairs_are_unwrapped_before_response(self):
        paper = SimpleNamespace(
            id=42, arxiv_id="2609.00042", title="Test paper", abstract="abstract",
            categories="cs.LG", citation_count=10, arxiv_url="https://arxiv.org/abs/2609.00042",
            heuristic_impact_score=0.7,
        )
        queries = iter([_Query([paper]), _Query([])])
        pipeline = DailyFeedPipeline.__new__(DailyFeedPipeline)
        pipeline.db = SimpleNamespace(query=lambda *args: next(queries))
        pipeline.user_id = 1
        pipeline._apply_mmr = lambda scored, top_k, item_type: scored[:top_k]

        seeds = pipeline.get_coldstart_seeds(n=1)

        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["db_id"], 42)
        self.assertEqual(seeds[0]["arxiv_id"], "2609.00042")


if __name__ == "__main__":
    unittest.main()
