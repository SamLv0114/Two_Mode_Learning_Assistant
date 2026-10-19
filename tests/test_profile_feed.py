"""
The profile-driven feed end to end on SQLite: recall from three channels, exclusions,
the model's judgements, stored reasons, regeneration on a profile edit, and the ML gate.
"""
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from _fakes import TaskClient, WordEmbeddings, user_text
from src.agents import research_profile
from src.collectors.semantic_scholar import S2Paper
from src.database.models import (Base, Paper, User, UserInteraction, UserModelState, UserPaperImpression,
                                 UserPaperRecommendation)
from src.pipelines import daily_feed
from src.pipelines.feed import channels, profile_feed
from src.pipelines.daily_feed import DailyFeedPipeline

NOW = datetime.now(timezone.utc)


def judge_reply(fits):
    """A feed_judge handler giving each listed paper the goal fit its title asks for."""
    def reply(kwargs):
        text = user_text(kwargs)
        rows = re.findall(r"^\[(\d+)\] (.+)$", text, re.M)
        return {"papers": [{"index": int(i), "goal_fit": fits(title), "readiness": "right_level",
                            "paper_type": "survey" if "survey" in title.lower() else "method",
                            "reason": f"Helps with diffusion: {title}."} for i, title in rows]}
    return reply


class ProfileFeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'pf.db'}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.user = User(email="reader@example.com", hashed_password="x")
        self.db.add(self.user)
        self.db.commit()
        self.em = WordEmbeddings()
        corpus = [("2501.00001", "Diffusion sampler", "diffusion diffusion sampling"),
                  ("2501.00002", "Diffusion survey", "diffusion survey of everything"),
                  ("2501.00003", "Diffusion for speech", "diffusion speech"),
                  ("2501.00004", "Shown yesterday diffusion", "diffusion"),
                  ("2501.00005", "Saved diffusion paper", "diffusion"),
                  ("2501.00006", "Language models", "language language"),
                  ("2501.00007", "Robot control", "robot")]
        for i, (arxiv_id, title, abstract) in enumerate(corpus):
            self.db.add(Paper(arxiv_id=arxiv_id, title=title, abstract=abstract, citation_count=i,
                              published_date=NOW - timedelta(days=i)))
            self.em.index(arxiv_id, title, abstract)
        self.db.commit()
        saved = self.db.query(Paper).filter(Paper.arxiv_id == "2501.00005").one()
        self.db.add(UserInteraction(user_id=self.user.id, item_type="paper", item_id=saved.id,
                                    interaction_type="saved", timestamp=NOW))
        self.db.add(UserPaperImpression(user_id=self.user.id, arxiv_id="2501.00004", shown_date=NOW))
        self.db.commit()
        research_profile.save(self.db, self.user, {
            "goal": "implement diffusion for speech", "level": "intermediate", "interests": ["diffusion"],
            "avoid_types": ["survey"], "seed_papers": [{"id": "2006.11239", "title": "DDPM diffusion"}],
            "summary": "Wants to implement diffusion models for speech."})

        self.graph = [S2Paper(paper_id="s1", title="Graph-found diffusion vocoder", abstract="diffusion speech vocoder",
                              arxiv_id="2602.00001", publication_date="2026-02-01", citation_count=3)]
        self.trending = [{"arxiv_id": "2610.00001", "title": "Trending diffusion speech", "upvotes": 50,
                          "abstract": "diffusion speech", "authors": [], "published_at": ""},
                         {"arxiv_id": "2610.00002", "title": "Trending robots", "upvotes": 99,
                          "abstract": "robot robot", "authors": [], "published_at": ""}]
        self.client = TaskClient(feed_judge=judge_reply(lambda t: 3 if "speech" in t.lower() or "vocoder" in t.lower()
                                                        else 0 if "robot" in t.lower() or "language" in t.lower()
                                                        else 2))
        for p in (patch.object(channels.semantic_scholar, "recommend", return_value=self.graph),
                  patch.object(channels, "trending_records", return_value=self.trending),
                  patch.object(profile_feed, "openai_client", return_value=self.client),
                  # Summaries are not under test, and CI has no OpenAI key for the real Generator.
                  patch.object(daily_feed, "Generator", lambda: SimpleNamespace(generate_summary=lambda **kw: "summary")),
                  patch("src.agents.tools._expand_query", side_effect=lambda q: [q]),
                  patch.object(daily_feed.settings, "TOP_PAPERS_COUNT", 3)):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def pipeline(self):
        pipe = DailyFeedPipeline(self.user.id, self.db, embedding_manager=self.em)
        pipe._maybe_retrain = lambda: None
        return pipe

    def test_feed_draws_on_every_channel_and_excludes_seen_and_saved(self):
        result = self.pipeline().run(mode="recommended")
        ids = [p["arxiv_id"] for p in result["papers"]]
        self.assertEqual(result["selection"], "llm_judge")
        self.assertEqual(len(ids), 3)
        self.assertIn("2602.00001", ids)                 # graph channel, added to the corpus on the way
        self.assertIn("2610.00001", ids)                 # trending, close to the profile
        self.assertNotIn("2610.00002", ids)              # trending but off-profile, never recalled
        self.assertNotIn("2501.00004", ids)              # shown yesterday
        self.assertNotIn("2501.00005", ids)              # already saved
        self.assertNotIn("2501.00002", ids)              # a survey, which the profile avoids
        graph = next(p for p in result["papers"] if p["arxiv_id"] == "2602.00001")
        self.assertEqual(graph["reasons"]["channels"], ["graph"])
        self.assertIn("DDPM diffusion", graph["reasons"]["evidence"][0])
        self.assertTrue(graph["reasons"]["reason"].startswith("Helps with diffusion"))

    def test_reasons_are_stored_and_the_same_feed_is_reused_until_the_profile_changes(self):
        first = self.pipeline().run(mode="recommended")
        row = self.db.query(UserPaperRecommendation).filter(UserPaperRecommendation.rank == 1).one()
        self.assertIn("goal_fit", row.reasons_json)
        again = self.pipeline().run(mode="recommended")
        self.assertTrue(again.get("reused"))
        self.assertEqual(again["papers"][0]["reasons"], first["papers"][0]["reasons"])
        research_profile.save(self.db, self.user, {"interests": ["diffusion"], "summary": "Diffusion for speech."})
        self.assertFalse(self.pipeline().run(mode="recommended").get("reused"))

    def test_without_the_model_the_prerank_order_stands_with_channel_reasons(self):
        self.client.handlers["feed_judge"] = lambda kw: "not json"
        result = self.pipeline().run(mode="recommended")
        self.assertEqual(result["selection"], "prerank")
        self.assertEqual(len(result["papers"]), 3)
        self.assertTrue(all(p["reasons"]["reason"] is None and p["reasons"]["evidence"] for p in result["papers"]))

    def test_latest_mode_judges_the_date_window(self):
        result = self.pipeline().run(mode="latest", time_window_days=3)
        ids = [p["arxiv_id"] for p in result["papers"]]
        # In the window: 00001-00003 (00004 was shown yesterday). No graph or trending recall in Latest.
        self.assertEqual(ids, ["2501.00003", "2501.00001", "2501.00002"])  # best fit first, avoided survey last
        self.assertTrue(all(p["reasons"]["channels"] == ["latest"] for p in result["papers"]))
        self.assertEqual(result["papers"][0]["reasons"]["evidence"], ["Published in the last 3 days"])

    def run_for_new_user(self, implemented: bool):
        other = User(email="other@example.com", hashed_password="x")
        self.db.add(other)
        self.db.commit()
        if implemented:  # the Coding Agent creates a profile row holding only this
            research_profile.record_studied(self.db, other.id, "2106.09685", "LoRA")
        pipe = DailyFeedPipeline(other.id, self.db, embedding_manager=self.em)
        pipe._maybe_retrain = lambda: None
        with patch.object(pipe, "_fetch_papers_recommended", return_value=[]) as earlier:
            pipe.run(mode="recommended")
        return earlier

    def test_users_without_a_profile_keep_the_earlier_path(self):
        self.run_for_new_user(implemented=False).assert_called_once()
        self.assertEqual(self.client.requests, [])

    def test_a_profile_with_only_implemented_papers_is_not_enough_for_the_profile_path(self):
        self.run_for_new_user(implemented=True).assert_called_once()


class TrainedRankerGateTests(unittest.TestCase):
    def gate(self, state, interactions=60, allow=True):
        pipe = DailyFeedPipeline.__new__(DailyFeedPipeline)
        pipe.user_id, pipe._allow_ml, pipe._ranking_fallback_reason, pipe._trained_ranker_ok = 1, allow, None, None
        pipe.trainer = SimpleNamespace(get_interaction_count=lambda: interactions)
        pipe.db = SimpleNamespace(query=lambda model: SimpleNamespace(
            filter=lambda *a: SimpleNamespace(first=lambda: state)))
        return pipe._use_trained_ranker(), pipe._ranking_fallback_reason

    def test_a_model_must_beat_the_heuristic_on_held_out_feedback(self):
        def state(val, base):
            return UserModelState(is_trained=True, val_ndcg=val, val_baseline_ndcg=base)
        self.assertEqual(self.gate(state(0.7, 0.6)), (True, None))
        self.assertEqual(self.gate(state(0.5, 0.6)), (False, "model_below_heuristic_on_validation"))
        self.assertEqual(self.gate(state(None, None)), (False, "model_not_validated"))
        self.assertEqual(self.gate(state(0.9, 0.1), interactions=10), (False, None))
        self.assertEqual(self.gate(state(0.9, 0.1), allow=False), (False, None))


if __name__ == "__main__":
    unittest.main()
