import unittest
from types import SimpleNamespace

from src.agents.reflection_agent import ReflectionAgent


class ReflectionStreamTests(unittest.TestCase):
    def test_draft_arrives_before_critic_and_refinement_replaces_it(self):
        agent = object.__new__(ReflectionAgent)
        prompts = []

        def inner_stream(prompt, _history, _context):
            prompts.append(prompt)
            text = "initial answer" if len(prompts) == 1 else "checked answer"
            for word in text.split():
                yield {"type": "token", "value": word + " "}
            yield {"type": "done", "tools_called": [], "citations": [{"title": "paper", "evidence": "proof"}]}

        agent._inner = SimpleNamespace(stream=inner_stream)
        scores = iter([
            SimpleNamespace(aggregate=0.4, should_retry=True, critique="Needs evidence", to_dict=lambda: {"aggregate": 0.4}),
            SimpleNamespace(aggregate=0.8, should_retry=False, critique="", to_dict=lambda: {"aggregate": 0.8}),
        ])
        agent._critic = SimpleNamespace(evaluate=lambda *_: next(scores))
        events = list(agent.stream("question", [], {}))
        types = [event["type"] for event in events]
        self.assertLess(types.index("draft_token"), types.index("critiquing"))
        self.assertEqual("".join(e["value"] for e in events if e["type"] == "draft_token"), "initial answer ")
        self.assertEqual(next(e["value"] for e in events if e["type"] == "replace"), "checked answer ")
        self.assertEqual(types[-1], "done")


if __name__ == "__main__":
    unittest.main()
