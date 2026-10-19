"""Feed polling must not expose another user's job or retain expired fallbacks."""
import time
import unittest
from unittest.mock import patch

from src.api.routers import feed
from src.utils import redis_client


class FeedJobScopeTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(redis_client, "get_redis", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        feed._jobs.local.clear()

    def test_status_is_owner_scoped(self):
        feed._set_status(10, "known", {"status": "done", "papers_count": 5})
        self.assertEqual(feed._get_status(10, "known")["papers_count"], 5)
        self.assertEqual(feed._get_status(11, "known"), {"status": "not_found"})

    def test_memory_fallback_expires(self):
        feed._set_status(10, "expired", {"status": "done"})
        key = next(iter(feed._jobs.local))
        feed._jobs.local[key] = (time.monotonic() - 1, {"status": "done"})
        self.assertEqual(feed._get_status(10, "expired"), {"status": "not_found"})
        self.assertNotIn(key, feed._jobs.local)


if __name__ == "__main__":
    unittest.main()
