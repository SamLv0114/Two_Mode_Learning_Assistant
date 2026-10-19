"""PlanAndSolveAgent's plan: schema-checked, capped, and a clean single-step fallback."""
import json
import unittest
from types import SimpleNamespace

from src.agents.plan_and_solve_agent import PlanAndSolveAgent


def _agent(*replies):
    agent = PlanAndSolveAgent.__new__(PlanAndSolveAgent)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        content = replies[len(requests) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, requests


class PlanTests(unittest.TestCase):
    def test_plan_is_requested_as_a_strict_schema_and_capped_at_five_steps(self):
        steps = [{"step": f"Compare aspect {i}", "focus": "cost"} for i in range(7)]
        agent, requests = _agent(json.dumps({"steps": steps}))
        plan = agent._generate_plan("LoRA vs QLoRA?")
        self.assertEqual([s.number for s in plan], [1, 2, 3, 4, 5])
        self.assertEqual(requests[0]["response_format"]["type"], "json_schema")

    def test_unusable_reply_falls_back_to_answering_directly(self):
        agent, requests = _agent("```json\n[not valid]```", "still not json")
        plan = agent._generate_plan("LoRA vs QLoRA?")
        self.assertEqual([(s.step, s.focus) for s in plan], [("LoRA vs QLoRA?", "direct answer")])
        self.assertEqual(len(requests), 2)  # one repair turn, then the fallback


if __name__ == "__main__":
    unittest.main()
