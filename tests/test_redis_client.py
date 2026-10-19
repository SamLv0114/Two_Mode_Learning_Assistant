"""Shared Redis client: one pool per process, None while Redis is down."""
import time
import unittest
from unittest.mock import MagicMock, patch

from src.utils import redis_client


class GetRedisTests(unittest.TestCase):
    def setUp(self):
        redis_client._client = None
        redis_client._down_since = 0.0

    tearDown = setUp

    def test_none_without_url(self):
        with patch.object(redis_client.settings, "REDIS_URL", None):
            self.assertIsNone(redis_client.get_redis())

    def test_reuses_one_client(self):
        fake = MagicMock()
        with patch.object(redis_client.settings, "REDIS_URL", "redis://x"), \
                patch("redis.from_url", return_value=fake) as from_url:
            self.assertIs(redis_client.get_redis(), fake)
            self.assertIs(redis_client.get_redis(), fake)
        from_url.assert_called_once()
        self.assertEqual(fake.ping.call_count, 2)

    def test_down_redis_is_not_retried_until_cooldown_ends(self):
        fake = MagicMock()
        fake.ping.side_effect = ConnectionError("down")
        with patch.object(redis_client.settings, "REDIS_URL", "redis://x"), \
                patch("redis.from_url", return_value=fake), \
                patch.object(redis_client.time, "monotonic", side_effect=[100.0, 110.0, 200.0, 200.0]):
            self.assertIsNone(redis_client.get_redis())   # ping fails at t=100
            self.assertIsNone(redis_client.get_redis())   # t=110: cooling down, no ping
            fake.ping.side_effect = None
            self.assertIs(redis_client.get_redis(), fake)  # t=200: retried and back
        self.assertEqual(fake.ping.call_count, 2)


class JsonCacheTests(unittest.TestCase):
    def test_uses_redis_when_it_is_up(self):
        fake = MagicMock()
        fake.get.return_value = '{"a": 1}'
        cache = redis_client.JsonCache("t", ttl_seconds=60)
        with patch.object(redis_client, "get_redis", return_value=fake):
            cache.set("k", {"a": 1})
            self.assertEqual(cache.get("k"), {"a": 1})
        fake.setex.assert_called_once_with("t:k", 60, '{"a": 1}')
        self.assertEqual(cache.local, {})

    def test_falls_back_to_this_process_and_expires(self):
        cache = redis_client.JsonCache("t", ttl_seconds=60)
        with patch.object(redis_client, "get_redis", return_value=None):
            cache.set("k", [1, 2])
            self.assertEqual(cache.get("k"), [1, 2])
            cache.local["t:k"] = (time.monotonic() - 1, [1, 2])
            self.assertIsNone(cache.get("k"))
            self.assertNotIn("t:k", cache.local)

    def test_local_copy_is_bounded(self):
        cache = redis_client.JsonCache("t", ttl_seconds=60, max_local=3)
        with patch.object(redis_client, "get_redis", return_value=None):
            for i in range(10):
                cache.set(str(i), i)
        self.assertLessEqual(len(cache.local), 3)


if __name__ == "__main__":
    unittest.main()
