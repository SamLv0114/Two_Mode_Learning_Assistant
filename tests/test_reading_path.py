"""Reading path agent: plan searches, find foundations through references, order, and fall back."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from _fakes import TaskClient, WordEmbeddings, user_text
from src.agents import reading_path_agent as agent
from src.collectors.semantic_scholar import S2Paper
from src.database.models import Base, Paper

DDPM = S2Paper(paper_id="ddpm", title="Denoising Diffusion Probabilistic Models", abstract="diffusion",
               year=2020, arxiv_id="2006.11239", citation_count=9000, publication_date="2020-06-19")
SCORE = S2Paper(paper_id="score", title="Score-Based Generative Modeling", abstract="diffusion score",
                year=2021, arxiv_id="2011.13456", citation_count=4000)
NCSN = S2Paper(paper_id="ncsn", title="Generative Modeling by Estimating Gradients", abstract="score matching",
               year=2019, arxiv_id="1907.05600", citation_count=3000)
VAE = S2Paper(paper_id="vae", title="Auto-Encoding Variational Bayes", abstract="vae", year=2013,
              arxiv_id="1312.6114", citation_count=30000)
NO_ABSTRACT = S2Paper(paper_id="old", title="A 1990s paper", abstract="", year=1995, citation_count=500)


class ReadingPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'rp.db'}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.db.add(Paper(arxiv_id="2207.00001", title="Diffusion for audio", abstract="diffusion speech"))
        self.db.commit()
        self.em = WordEmbeddings()
        self.em.index("2207.00001", "Diffusion for audio", "diffusion speech")
        search = {"diffusion models": [DDPM, SCORE], "score matching": [NCSN]}
        refs = {"ddpm": [NCSN, VAE, NO_ABSTRACT], "ncsn": [VAE]}
        for p in (patch.object(agent.semantic_scholar, "search", side_effect=lambda q, limit=6: search.get(q, [])),
                  patch.object(agent.semantic_scholar, "references", side_effect=lambda pid: refs.get(pid, []))):
            p.start()
            self.addCleanup(p.stop)
        self.profile = {"goal": "implement DDPM", "level": "intermediate", "known": ["VAEs"], "interests": [],
                        "preferences": [], "studied": []}

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def order(self, kwargs):
        listing = user_text(kwargs)
        index = {title: i for i, title in enumerate(t.split("] ", 1)[1].split(" (")[0]
                                                     for t in listing.splitlines() if t.startswith("["))}
        return {"steps": [{"index": index["Generative Modeling by Estimating Gradients"], "stage": "foundation",
                           "why": "Score matching comes first."},
                          {"index": index["Denoising Diffusion Probabilistic Models"], "stage": "core",
                           "why": "The model you want to implement."},
                          {"index": index["Denoising Diffusion Probabilistic Models"], "stage": "core", "why": "dup"},
                          {"index": 999, "stage": "frontier", "why": "out of range"}],
                "skipped": "VAE basics, which you already know."}

    def test_path_is_ordered_with_foundations_found_through_references(self):
        client = TaskClient(reading_path_plan=lambda kw: {"queries": ["diffusion models", "score matching"]},
                            reading_path_order=self.order)
        with patch.object(agent, "MIN_POOL", 5):
            path = agent.build_reading_path(self.db, self.em, client, "Understand diffusion", self.profile)
        self.assertEqual([s["paper"]["title"] for s in path["steps"]],
                         ["Generative Modeling by Estimating Gradients", "Denoising Diffusion Probabilistic Models"])
        self.assertEqual(path["skipped"], "VAE basics, which you already know.")
        listing = user_text(client.requests[-1][1])
        self.assertIn("Cited by 2 of the central papers", listing)   # VAE: cited by both anchors
        self.assertIn("Already understands: VAEs", listing)
        # Chosen papers with an arXiv id join the corpus, so they can be saved.
        self.assertIsNotNone(path["steps"][1]["paper"]["db_id"])
        self.assertEqual(client.tasks().count("reading_path_plan"), 1)  # the pool was big enough
        self.assertFalse(path["library_only"])

    def test_thin_pool_triggers_one_more_planning_round(self):
        queries = iter([{"queries": ["nothing matches"]}, {"queries": ["diffusion models"]}])
        client = TaskClient(reading_path_plan=lambda kw: next(queries), reading_path_order=self.order)
        path = agent.build_reading_path(self.db, self.em, client, "Understand diffusion", self.profile)
        self.assertEqual(client.tasks().count("reading_path_plan"), 2)
        self.assertEqual(path["queries"], ["nothing matches", "diffusion models"])
        self.assertIn("A first round of searches found too little", user_text(client.requests[1][1]))

    def test_a_path_built_without_semantic_scholar_says_so(self):
        client = TaskClient(reading_path_plan=lambda kw: {"queries": ["diffusion"]},
                            reading_path_order=lambda kw: {"steps": [{"index": 0, "stage": "core", "why": "x"}],
                                                           "skipped": ""})
        with patch.object(agent.semantic_scholar, "search", return_value=None):  # no answer
            path = agent.build_reading_path(self.db, self.em, client, "Understand diffusion", self.profile)
        self.assertTrue(path["library_only"])
        self.assertEqual(path["steps"][0]["paper"]["source"], "library")

    def test_ordering_failure_falls_back_to_publication_order(self):
        client = TaskClient(reading_path_plan=lambda kw: {"queries": ["diffusion models"]},
                            reading_path_order=lambda kw: "not json")
        path = agent.build_reading_path(self.db, self.em, client, "Understand diffusion", self.profile)
        years = [s["paper"]["year"] for s in path["steps"] if s["paper"]["year"]]
        self.assertEqual(path["ordered_by"], "year")
        self.assertEqual(years, sorted(years))


if __name__ == "__main__":
    unittest.main()
