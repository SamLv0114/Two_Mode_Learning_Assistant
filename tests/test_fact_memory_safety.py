"""A failed read of stored facts must never be written back as 'no facts'."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.agents.memory import UserFactMemory


def _model_returning(facts):
    message = SimpleNamespace(content=json.dumps({"facts": facts}))
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)])
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


class FactMemoryReadFailureTests(unittest.TestCase):
    def test_merge_aborts_instead_of_overwriting_when_the_read_fails(self):
        memory = UserFactMemory()
        memory._client = _model_returning(["user prefers PyTorch"])
        redis = MagicMock()
        redis.get.side_effect = TimeoutError("redis timeout")
        with self.assertRaises(TimeoutError):
            memory.extract_and_store(7, "I use PyTorch", "Noted", redis)
        redis.setex.assert_not_called()

    def test_readers_still_degrade_to_no_facts(self):
        redis = MagicMock()
        redis.get.side_effect = TimeoutError("redis timeout")
        with self.assertLogs("src.agents.memory", level="WARNING"):
            self.assertEqual(UserFactMemory().get_context_string(7, redis), "")


if __name__ == "__main__":
    unittest.main()
