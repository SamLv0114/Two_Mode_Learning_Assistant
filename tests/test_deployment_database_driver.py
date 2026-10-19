"""The production Compose URL must select a DBAPI installed by requirements."""

import re
import unittest
from pathlib import Path

from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]


class DeploymentDatabaseDriverTests(unittest.TestCase):
    def test_compose_database_url_uses_available_driver(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        match = re.search(r"^\s*- DATABASE_URL=(postgresql\+[^:]+)://", compose, re.MULTILINE)
        self.assertIsNotNone(match, "Compose must select an explicit PostgreSQL driver")

        dialect = make_url(f"{match.group(1)}://user:pass@db:5432/test").get_dialect()
        dialect.import_dbapi()

    def test_practice_worker_waits_for_healthy_api_and_is_deployable(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("practice-worker:", compose)
        self.assertIn('command: ["python", "-m", "src.jobs.practice_worker"]', compose)
        self.assertRegex(compose, r"(?s)api:.*?healthcheck:.*?practice-worker:")
        self.assertRegex(compose, r"(?s)practice-worker:.*?depends_on:\s+api:\s+condition: service_healthy")

        deploy = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        self.assertIn("docker compose build --build-arg GIT_SHA=\"$AFTER\" api practice-worker", deploy)
        self.assertIn("learning_assistant_practice_worker", deploy)


if __name__ == "__main__":
    unittest.main()
