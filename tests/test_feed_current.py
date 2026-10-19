"""GET /feed/current: the dashboard shows the last feed on load instead of an empty page."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.deps import get_current_user, get_db_session
from src.api.routers import feed, interactions
from src.database.models import Base, Paper, User, UserInteraction, UserPaperRecommendation


class _FeedApiCase(unittest.TestCase):
    """An app with the feed and interaction routers on a throwaway SQLite database."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'feed.db'}",
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.reader = User(email="reader@example.com", hashed_password="x")
        self.other = User(email="other@example.com", hashed_password="x")
        self.db.add_all([self.reader, self.other])
        self.db.commit()
        app = FastAPI()
        app.include_router(feed.router)
        app.include_router(interactions.router)
        app.dependency_overrides[get_db_session] = lambda: self.db
        self.user_id = self.reader.id
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=self.user_id)
        self.client = TestClient(app)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()


class CurrentFeedTests(_FeedApiCase):
    def test_user_without_a_feed_gets_none(self):
        body = self.client.get("/feed/current").json()
        self.assertEqual(body, {"papers": [], "generated_at": None})

    def test_returns_own_feed_in_rank_order_with_how_it_was_built(self):
        when = datetime(2026, 10, 9, 8, 0, tzinfo=timezone.utc)
        papers = [Paper(arxiv_id=f"2610.0000{i}", title=f"P{i}", citation_count=0) for i in range(3)]
        self.db.add_all(papers)
        self.db.commit()
        for rank, paper in zip([2, 1], papers[:2]):
            self.db.add(UserPaperRecommendation(user_id=self.reader.id, paper_id=paper.id, rank=rank,
                                                recommended_date=when, feed_mode="latest",
                                                feed_time_window_days=7, ranking_source="ml"))
        self.db.add(UserPaperRecommendation(user_id=self.other.id, paper_id=papers[2].id, rank=1,
                                            recommended_date=when))
        self.db.commit()
        body = self.client.get("/feed/current").json()
        self.assertEqual([p["arxiv_id"] for p in body["papers"]], ["2610.00001", "2610.00000"])
        self.assertEqual((body["feed_mode"], body["time_window_days"], body["ranking_source"]), ("latest", 7, "ml"))
        self.assertTrue(body["generated_at"].startswith("2026-10-09T08:00"))

    def test_unscored_row_does_not_break_the_feed(self):
        paper = Paper(arxiv_id="2610.00009", title="Unscored", citation_count=None)
        self.db.add(paper)
        self.db.commit()
        self.db.add(UserPaperRecommendation(user_id=self.reader.id, paper_id=paper.id, rank=1,
                                            relevance_score=None))
        self.db.commit()
        for path in ("/feed/current", "/feed/papers"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)


class SavedPapersTests(_FeedApiCase):
    def test_saved_lists_papers_and_ignores_retired_article_rows(self):
        paper = Paper(arxiv_id="2610.00010", title="Kept", citation_count=None)
        self.db.add(paper)
        self.db.commit()
        self.db.add_all([UserInteraction(user_id=self.reader.id, item_type="paper", item_id=paper.id,
                                         interaction_type="saved"),
                         UserInteraction(user_id=self.reader.id, item_type="article", item_id=paper.id,
                                         interaction_type="saved")])
        self.db.commit()
        body = self.client.get("/feed/saved").json()
        self.assertEqual(([p["title"] for p in body["papers"]], body["total_saved"]), (["Kept"], 1))
        self.assertEqual(self.client.get("/interactions/stats").json()["saved"], 1)

    def test_interactions_are_for_papers_only(self):
        paper = Paper(arxiv_id="2610.00011", title="P")
        self.db.add(paper)
        self.db.commit()
        ok = self.client.post("/interactions", json={"item_id": paper.id, "interaction_type": "saved"})
        self.assertEqual((ok.status_code, ok.json()["item_type"]), (201, "paper"))
        retired = self.client.post("/interactions", json={"item_type": "article", "item_id": paper.id,
                                                          "interaction_type": "saved"})
        self.assertEqual(retired.status_code, 422)
        missing = self.client.post("/interactions", json={"item_id": 99999, "interaction_type": "saved"})
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
