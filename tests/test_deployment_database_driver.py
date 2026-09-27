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


if __name__ == "__main__":
    unittest.main()
