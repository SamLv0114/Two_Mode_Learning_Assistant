import unittest
from types import SimpleNamespace

from src.agents.base_agent import AgentResult
from src.agents.supervisor_experiment import SupervisorExperiment


class SupervisorExperimentTests(unittest.TestCase):
    def test_handoff_and_failure_fallback(self):
        first = AgentResult("Finding A", ["search_knowledge_base"], [{"title": "A", "url": "u", "evidence": "proof"}])
        observed = []
        primary = SimpleNamespace(name="ResearchAgent", run=lambda *_: first)

        def second_run(message, _history, _context):
            observed.append(message)
            return AgentResult("Checked answer", ["search_web"], [{"title": "A", "url": "u"}])

        secondary = SimpleNamespace(name="AnalysisAgent", run=second_run)
        context = {}
        result = SupervisorExperiment(primary, secondary).run("Question", [], context)
        self.assertEqual(result.reply, "Checked answer")
        self.assertEqual(len(result.citations), 1)
        self.assertIn("<untrusted_source", observed[0])
        self.assertEqual(context["_supervisor_handoffs"][0]["to"], "AnalysisAgent")

        secondary.run = lambda *_: (_ for _ in ()).throw(RuntimeError("specialist unavailable"))
        self.assertEqual(SupervisorExperiment(primary, secondary).run("Question", [], {}).reply, "Finding A")


if __name__ == "__main__":
    unittest.main()
