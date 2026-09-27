"""Feed polling must not expose another user's job or retain expired fallbacks."""
import time
import unittest
from unittest.mock import patch

from src.api.routers import feed


class FeedJobScopeTests(unittest.TestCase):
    def test_status_is_owner_scoped(self):
        with patch.object(feed, "_get_redis", return_value=None):
            feed._set_status(10, "known", {"status": "done", "papers_count": 5})
            self.assertEqual(feed._get_status(10, "known")["papers_count"], 5)
            self.assertEqual(feed._get_status(11, "known"), {"status": "not_found"})

    def test_memory_fallback_expires(self):
        with patch.object(feed, "_get_redis", return_value=None):
            key = feed._status_key(10, "expired")
            feed._job_store[key] = (time.monotonic() - 1, {"status": "done"})
            self.assertEqual(feed._get_status(10, "expired"), {"status": "not_found"})
            self.assertNotIn(key, feed._job_store)


if __name__ == "__main__":
    unittest.main()
