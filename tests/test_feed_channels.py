"""Recall channels: quota merge, adding outside papers to the corpus, graph and trending recall."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from _fakes import WordEmbeddings
from src.collectors.semantic_scholar import S2Paper
from src.database.models import Base, Paper
from src.pipelines.feed import channels


class MergeTests(unittest.TestCase):
    def test_each_channel_gets_its_quota_before_any_overflow(self):
        ranked = {"content": [f"c{i}" for i in range(10)], "graph": ["g0", "g1", "g2"], "trending": ["t0"]}
        merged = channels.merge(ranked, {"content": 3, "graph": 2, "trending": 1})
        self.assertEqual(merged, ["c0", "c1", "c2", "g0", "g1", "t0"])

    def test_unfilled_slots_go_to_other_channels_and_duplicates_count_once(self):
        ranked = {"content": ["a", "b", "c", "d"], "graph": ["a"], "trending": []}
        merged = channels.merge(ranked, {"content": 2, "graph": 2, "trending": 2})
        self.assertEqual(merged, ["a", "b", "c", "d"])


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'c.db'}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.db.add(Paper(arxiv_id="2001.00001", title="Known diffusion paper", abstract="diffusion"))
        self.db.commit()
        self.em = WordEmbeddings()
        self.em.index("2001.00001", "Known diffusion paper", "diffusion")

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def test_ingest_adds_only_new_complete_papers_to_both_stores(self):
        present = channels.ingest(self.db, self.em, [
            channels.OutsidePaper("2001.00001", "Known diffusion paper", "diffusion"),
            channels.OutsidePaper("2401.00002", "New graph paper", "graph networks", ["A"], citation_count=4),
            channels.OutsidePaper("2401.00003", "No abstract", ""),
            channels.OutsidePaper("2401.00002", "Duplicate", "graph")])
        self.assertEqual(present, ["2001.00001", "2401.00002"])
        new = self.db.query(Paper).filter(Paper.arxiv_id == "2401.00002").one()
        self.assertEqual((new.title, new.citation_count), ("New graph paper", 4))
        self.assertIn("paper_2401.00002", self.em.store)
        self.assertEqual(self.db.query(Paper).count(), 2)

    def test_graph_recall_keeps_arxiv_papers_in_semantic_scholar_order(self):
        recommended = [S2Paper(paper_id="s1", title="Graph diffusion", abstract="graph diffusion", arxiv_id="2402.00001"),
                       S2Paper(paper_id="s2", title="No arXiv id", abstract="x"),
                       S2Paper(paper_id="s3", title="Known diffusion paper", abstract="diffusion", arxiv_id="2001.00001")]
        with patch.object(channels.semantic_scholar, "recommend", return_value=recommended) as recommend:
            ids = channels.graph_recall(self.db, self.em, ["2006.11239", "2006.11239", ""], ["2101.0"])
        self.assertEqual(ids, ["2402.00001", "2001.00001"])
        recommend.assert_called_once_with(["2006.11239"], ["2101.0"], limit=channels.GRAPH_RECALL_LIMIT)

    def test_graph_recall_needs_a_liked_paper(self):
        with patch.object(channels.semantic_scholar, "recommend") as recommend:
            self.assertEqual(channels.graph_recall(self.db, self.em, [], []), [])
        recommend.assert_not_called()

    def test_trending_keeps_papers_close_to_the_profile_most_upvoted_first(self):
        records = [{"arxiv_id": "2610.1", "title": "Diffusion for speech", "abstract": "diffusion speech",
                    "authors": [], "published_at": "2026-10-15T07:47:04.000Z", "upvotes": 5, "has_code": True},
                   {"arxiv_id": "2610.2", "title": "Robot arms", "abstract": "robot robot", "authors": [],
                    "published_at": "", "upvotes": 90},
                   {"arxiv_id": "2610.3", "title": "Diffusion again", "abstract": "diffusion", "authors": [],
                    "published_at": "", "upvotes": 40}]
        profile_vec = np.asarray(self.em.generate_embedding("diffusion"))
        with patch.object(channels, "trending_records", return_value=records):
            found = channels.trending_recall(self.db, self.em, profile_vec, min_similarity=0.5)
        self.assertEqual(list(found), ["2610.3", "2610.1"])
        added = self.db.query(Paper).filter(Paper.arxiv_id == "2610.1").one()
        self.assertEqual(added.published_date.year, 2026)


if __name__ == "__main__":
    unittest.main()
