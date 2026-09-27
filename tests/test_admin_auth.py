"""Admin jobs must require a superuser access token."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.deps import get_db_session
from src.api.main import app
from src.api.security import create_access_token, create_refresh_token


class _Query:
    def __init__(self, user):
        self.user = user

    def filter(self, *_args):
        return self

    def first(self):
        return self.user


class _DB:
    def __init__(self, user):
        self.user = user

    def query(self, *_args):
        return _Query(self.user)


class AdminAuthTests(unittest.TestCase):
    def test_both_admin_jobs_require_superuser_access_token(self):
        user = SimpleNamespace(id=7, is_active=True, is_superuser=False)
        app.dependency_overrides[get_db_session] = lambda: _DB(user)
        access = create_access_token(7, "test@example.com")
        refresh, _ = create_refresh_token(7)
        paths = ["/api/v1/admin/index-now", "/api/v1/admin/refresh-citations-now"]

        try:
            with patch("src.api.main.run_nightly_index", return_value={"ok": True}) as index_job, \
                    patch("src.api.main.run_citation_refresh", return_value={"ok": True}) as citation_job:
                client = TestClient(app)
                for path in paths:
                    self.assertEqual(client.post(path).status_code, 401)
                    self.assertEqual(client.post(path, headers={"Authorization": f"Bearer {refresh}"}).status_code, 401)
                    self.assertEqual(client.post(path, headers={"Authorization": f"Bearer {access}"}).status_code, 403)
                index_job.assert_not_called()
                citation_job.assert_not_called()

                user.is_superuser = True
                for path in paths:
                    self.assertEqual(client.post(path, headers={"Authorization": f"Bearer {access}"}).status_code, 200)
                index_job.assert_called_once()
                citation_job.assert_called_once()
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
