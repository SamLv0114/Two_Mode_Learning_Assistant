import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.agents.deep_research_agent import (
    DeepResearchAgent, PLANNER_MODEL, SUMMARIZER_MODEL, WRITER_MODEL,
    ResearchNote, TodoItem, research_stop_reason,
)
from src.agents.research_run_store import ResearchRunStore
from src.database.models import Base


class ResearchResumeTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.store = ResearchRunStore(self.db, 1)
        self.user = SimpleNamespace(id=1)
        self.tasks = [TodoItem(id=1, title="First", query="first"), TodoItem(id=2, title="Second", query="second")]

    def tearDown(self):
        self.db.close()

    def agent(self, calls):
        agent = object.__new__(DeepResearchAgent)
        agent.planner = SimpleNamespace(plan=lambda _: self.tasks)

        def summarize(task, context):
            calls.append(task.id)
            return ResearchNote(task_id=task.id, title=task.title, summary=f"Finding {task.id}")

        agent.summarizer = SimpleNamespace(summarize=summarize)
        def write_stream(question, notes):
            yield "report "
            yield str(len(notes))

        agent.writer = SimpleNamespace(stream=write_stream)
        return agent

    def test_owner_scoped_resume_skips_completed_tasks_and_is_idempotent(self):
        versions = {"schema": 1, "planner": PLANNER_MODEL, "summarizer": SUMMARIZER_MODEL, "writer": WRITER_MODEL}
        state = self.store.create("Question", [vars(task) for task in self.tasks], versions)
        state["notes"]["1"] = vars(ResearchNote(task_id=1, title="First", summary="saved"))
        state["status"] = "paused"
        self.store.save(state)
        self.assertIsNone(ResearchRunStore(self.db, 2).load(state["run_id"]))
        self.assertIsNone(ResearchRunStore(self.db, 2).claim(state["run_id"]))

        claimed = self.store.claim(state["run_id"])
        self.assertEqual(claimed["status"], "running")
        self.assertIsNone(self.store.claim(state["run_id"]))
        claimed["status"] = "paused"
        self.store.save(claimed)

        calls = []
        agent = self.agent(calls)
        context = {"db": self.db, "user": self.user, "resume_run_id": state["run_id"], "_cancel_event": threading.Event()}
        events = list(agent.stream("ignored", [], context))
        self.assertEqual(calls, [2])
        self.assertEqual("report 2", "".join(e["value"] for e in events if e["type"] == "token"))
        self.assertEqual(self.store.load(state["run_id"])["status"], "completed")

        events_again = list(agent.stream("ignored", [], context))
        self.assertEqual(calls, [2])
        self.assertIn("done", [e["type"] for e in events_again])

    def test_interrupted_stream_marks_checkpoint_paused(self):
        calls = []
        agent = self.agent(calls)
        context = {"db": self.db, "user": self.user, "_cancel_event": threading.Event()}
        stream = agent.stream("Question", [], context)
        run_event = next(stream)
        self.assertEqual(run_event["type"], "run")
        stream.close()
        saved = self.store.load(run_event["run_id"])
        self.assertEqual(saved["status"], "paused")
        self.assertEqual(saved["stop_reason"], "interrupted")

    def test_explicit_stop_rules_and_partial_report(self):
        notes = {
            1: ResearchNote(1, "First", "finding", citations=[{"url": f"source-{i}"} for i in range(2)]),
            2: ResearchNote(2, "Second", "finding", citations=[{"url": f"source-{i}"} for i in range(2, 4)]),
        }
        self.assertEqual(research_stop_reason(notes, 3, 0, 100, True), "evidence_sufficient")
        self.assertEqual(research_stop_reason(notes, 3, 101, 100), "time_budget")
        self.assertIsNone(research_stop_reason(notes, 3, 0, 100))

        agent = self.agent([])
        context = {"db": self.db, "user": self.user, "_cancel_event": threading.Event()}
        with patch("src.agents.deep_research_agent.settings.RESEARCH_MAX_SECONDS", 0):
            events = list(agent.stream("Question", [], context))
        reply = "".join(e["value"] for e in events if e["type"] == "token")
        self.assertIn("Unfinished research tasks", reply)
        self.assertEqual(events[-1]["stop_reason"], "time_budget")
        self.assertTrue(events[-1]["resumable"])
        run_id = next(e["run_id"] for e in events if e["type"] == "run")
        self.assertEqual(self.store.load(run_id)["status"], "paused")
        resumed = list(agent.stream("", [], {**context, "resume_run_id": run_id}))
        self.assertEqual(resumed[-1]["stop_reason"], "plan_complete")
        self.assertEqual(self.store.load(run_id)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
