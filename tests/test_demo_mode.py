"""Shared demo account: sign-in, account-changing actions refused, model work metered."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api import demo
from src.api.deps import get_current_user, get_db_session
from src.api.middleware import client_ip
from src.api.routers import auth, public
from src.database.models import Base, User
from src.utils.config import settings

DEMO_EMAIL = "demo@researchmate.site"


def _request(headers=None, host="10.0.0.1"):
    return SimpleNamespace(headers=headers or {}, client=SimpleNamespace(host=host))


class ClientIpTests(unittest.TestCase):
    def test_nginx_real_ip_wins_over_a_spoofable_forwarded_for(self):
        request = _request({"x-real-ip": "203.0.113.7", "x-forwarded-for": "1.2.3.4, 203.0.113.7"})
        self.assertEqual(client_ip(request), "203.0.113.7")

    def test_falls_back_to_the_socket_address(self):
        self.assertEqual(client_ip(_request({"x-forwarded-for": "1.2.3.4"})), "10.0.0.1")


class DemoGuardTests(unittest.TestCase):
    def setUp(self):
        demo._memory_counts.clear()
        self.stack = [
            patch.object(settings, "DEMO_ENABLED", True),
            patch.object(settings, "DEMO_USER_EMAIL", DEMO_EMAIL),
            patch.object(settings, "DEMO_ACTIONS_PER_VISITOR_PER_DAY", 2),
            patch.object(settings, "DEMO_ACTIONS_TOTAL_PER_DAY", 3),
            patch.object(demo, "get_redis", return_value=None),
        ]
        for p in self.stack:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.stack])
        self.app = FastAPI()
        self.user = SimpleNamespace(id=1, email=DEMO_EMAIL)
        self.app.dependency_overrides[get_current_user] = lambda: self.user

        @self.app.post("/costly")
        def costly(user=Depends(demo.demo_quota)):
            return {"ok": True}

        @self.app.post("/account")
        def account(user=Depends(demo.forbid_demo)):
            return {"ok": True}

        self.client = TestClient(self.app)

    def post(self, path, ip):
        return self.client.post(path, headers={"x-real-ip": ip}).status_code

    def test_account_changes_are_refused_for_the_demo_only(self):
        self.assertEqual(self.post("/account", "1.1.1.1"), 403)
        self.user = SimpleNamespace(id=2, email="someone@example.com")
        self.assertEqual(self.post("/account", "1.1.1.1"), 200)

    def test_each_visitor_and_the_whole_demo_have_a_daily_allowance(self):
        self.assertEqual([self.post("/costly", "1.1.1.1") for _ in range(3)], [200, 200, 429])
        # A visitor past their own allowance did not use up the shared one: 2 of 3 spent.
        self.assertEqual([self.post("/costly", "2.2.2.2") for _ in range(2)], [200, 429])

    def test_other_users_are_never_metered(self):
        self.user = SimpleNamespace(id=2, email="someone@example.com")
        self.assertEqual({self.post("/costly", "1.1.1.1") for _ in range(10)}, {200})

    def test_nothing_applies_when_the_demo_is_off(self):
        with patch.object(settings, "DEMO_ENABLED", False):
            self.assertEqual(self.post("/account", "1.1.1.1"), 200)
            self.assertEqual({self.post("/costly", "1.1.1.1") for _ in range(5)}, {200})


class DemoLoginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        engine = create_engine(f"sqlite:///{Path(self.temp.name) / 'demo.db'}",
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        app = FastAPI()
        app.include_router(auth.router)
        app.include_router(public.router)
        app.dependency_overrides[get_db_session] = lambda: self.db
        app.dependency_overrides[auth.get_redis] = lambda: None
        self.client = TestClient(app)
        public._cache["value"] = None

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def test_demo_sign_in_needs_the_flag_and_the_account(self):
        with patch.object(settings, "DEMO_ENABLED", False):
            self.assertEqual(self.client.post("/auth/demo").status_code, 404)
        with patch.object(settings, "DEMO_ENABLED", True), patch.object(settings, "DEMO_USER_EMAIL", DEMO_EMAIL):
            self.assertEqual(self.client.post("/auth/demo").status_code, 404)  # not seeded yet
            self.db.add(User(email=DEMO_EMAIL, hashed_password="unusable"))
            self.db.commit()
            body = self.client.post("/auth/demo").json()
            self.assertTrue(body["access_token"])
            stats = self.client.get("/public/stats").json()
            self.assertEqual(stats, {"papers_indexed": 0, "demo_available": True})


if __name__ == "__main__":
    unittest.main()
