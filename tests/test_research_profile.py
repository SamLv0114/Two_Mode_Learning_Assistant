"""The research profile: drafted from the user's words, edited, fed back into, and kept by the Coding Agent."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from _fakes import TaskClient, WordEmbeddings, user_text
from src.agents import research_profile
from src.api.deps import get_current_user, get_db_session, get_embedding_manager
from src.api.routers import profile as profile_router
from src.database.models import Base, Paper, User

DRAFT = {"goal": "Implement a diffusion model for audio", "level": "intermediate", "known": ["VAEs", "VAEs"],
         "interests": ["diffusion models", "speech synthesis"], "preferences": ["papers with code"],
         "avoid_types": ["survey"], "seed_search_terms": ["denoising diffusion"],
         "summary": "An ML student who wants to implement diffusion for audio."}
CHANGE = {"add_preferences": [], "remove_preferences": [], "add_avoid_types": ["theory"], "remove_avoid_types": [],
          "add_known": [], "add_interests": [], "remove_interests": [],
          "note": "Fewer theory papers from now on."}


class ResearchProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'profile.db'}",
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.user = User(email="reader@example.com", hashed_password="x")
        self.db.add(self.user)
        self.db.add(Paper(arxiv_id="2006.11239", title="Denoising Diffusion Probabilistic Models",
                          abstract="diffusion models for image generation"))
        self.db.commit()
        self.em = WordEmbeddings()
        self.em.index("2006.11239", "Denoising Diffusion Probabilistic Models", "diffusion")
        self.client_fake = TaskClient(profile_draft=lambda kw: DRAFT, profile_feedback=lambda kw: CHANGE)
        app = FastAPI()
        app.include_router(profile_router.router)
        app.dependency_overrides[get_db_session] = lambda: self.db
        app.dependency_overrides[get_current_user] = lambda: self.user
        app.dependency_overrides[get_embedding_manager] = lambda: self.em
        for p in (patch.object(profile_router, "openai_client", return_value=self.client_fake),
                  patch.object(profile_router.semantic_scholar, "search", return_value=[])):
            p.start()
            self.addCleanup(p.stop)
        self.http = TestClient(app)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def test_no_profile_before_onboarding(self):
        self.assertEqual(self.http.get("/profile/research").json(), {"profile": None})

    def test_draft_extracts_fields_and_suggests_library_papers(self):
        body = self.http.post("/profile/research/draft", json={"description": "I study audio ML and want "
                                                                              "to implement diffusion."}).json()
        self.assertEqual(body["draft"]["known"], ["VAEs"])  # de-duplicated
        self.assertNotIn("seed_search_terms", body["draft"])
        self.assertEqual([p["id"] for p in body["suggested_papers"]], ["2006.11239"])
        self.assertIn("<untrusted", user_text(self.client_fake.requests[0][1]))  # the user's words are fenced

    def test_save_cleans_fields_bumps_version_and_syncs_interests(self):
        payload = {**DRAFT, "avoid_types": ["survey", "not-a-type"], "level": "expert",
                   "seed_papers": [{"id": "2006.11239", "title": "DDPM", "year": 2020}, {"id": "2006.11239"}],
                   "summary": ""}
        payload.pop("seed_search_terms")
        first = self.http.put("/profile/research", json=payload).json()["profile"]
        self.assertEqual(first["avoid_types"], ["survey"])
        self.assertEqual(first["level"], "")              # unknown level dropped
        self.assertEqual(len(first["seed_papers"]), 1)
        self.assertIn("diffusion models", first["summary"])  # composed when left empty
        self.assertEqual(self.user.get_interests_list(), ["diffusion models", "speech synthesis"])
        second = self.http.put("/profile/research", json=payload).json()["profile"]
        self.assertEqual(second["version"], first["version"] + 1)
        self.assertTrue(second["set_up"])

    def test_a_profile_holding_only_implemented_papers_is_not_set_up(self):
        research_profile.record_studied(self.db, self.user.id, "2106.09685", "LoRA")
        profile = self.http.get("/profile/research").json()["profile"]
        self.assertEqual(len(profile["studied"]), 1)
        self.assertFalse(profile["set_up"])

    def test_feedback_becomes_a_profile_edit_and_is_logged(self):
        self.http.put("/profile/research", json={"interests": ["diffusion models"], "summary": "Diffusion."})
        body = self.http.post("/profile/research/feedback", json={"text": "too theoretical",
                                                                  "paper_id": 1}).json()
        self.assertEqual(body["note"], "Fewer theory papers from now on.")
        self.assertEqual(body["profile"]["avoid_types"], ["theory"])
        self.assertEqual(body["profile"]["summary"], "Diffusion.")  # untouched
        self.assertEqual(body["profile"]["feedback"][0]["paper"], "Denoising Diffusion Probabilistic Models")

    def test_model_failure_is_a_503_not_a_500(self):
        self.client_fake.handlers["profile_draft"] = lambda kw: "not json"
        response = self.http.post("/profile/research/draft", json={"description": "I study things a lot."})
        self.assertEqual(response.status_code, 503)

    def test_paper_search_finds_library_papers(self):
        papers = self.http.get("/profile/research/papers", params={"q": "Denoising"}).json()["papers"]
        self.assertEqual(papers[0]["source"], "library")

    def test_studied_papers_are_kept_once_and_survive_edits(self):
        research_profile.record_studied(self.db, self.user.id, "2106.09685", "LoRA", "low-rank adapters")
        research_profile.record_studied(self.db, self.user.id, "2106.09685", "LoRA", "again")
        research_profile.save(self.db, self.user, {"interests": ["fine-tuning"]})
        data = research_profile.to_dict(research_profile.load(self.db, self.user.id))
        self.assertEqual([s["objective"] for s in data["studied"]], ["again"])
        self.assertIn("Has implemented: LoRA", research_profile.prompt_block(data))


if __name__ == "__main__":
    unittest.main()
