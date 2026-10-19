"""DeepResearch LangGraph loop: gap replanning, checkpoint resume, budgets, cancellation."""
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.agents import deep_research_agent as dr
from src.agents.deep_research_agent import (
    DeepResearchAgent, GapAssessor, ReportWriter, ResearchNote, TodoItem, TodoPlanner,
)
from src.agents.research_run_store import ResearchRunStore
from src.database.models import Base, ResearchRun


def _completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class DeepResearchGraphTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.store = ResearchRunStore(self.db, 1)
        self.saver = InMemorySaver()
        self.calls, self.assessed, self.written, self.planned_with, self.profiles = [], [], [], [], []
        self.gaps = [[{"title": "Benchmarks", "query": "benchmark results", "reason": "no numbers yet"}]]
        self.concurrency = patch.object(dr, "MAX_CONCURRENCY", 1)
        self.concurrency.start()

    def tearDown(self):
        self.concurrency.stop()
        self.db.close()

    def agent(self, fail_task=None):
        agent = object.__new__(DeepResearchAgent)
        agent.checkpointer = self.saver
        tasks = [TodoItem(1, "First", "first"), TodoItem(2, "Second", "second")]

        def plan(question, history, profile=""):
            self.planned_with.append((question, history))
            self.profiles.append(("plan", profile))
            return tasks

        def summarize(task, context):
            if task.id == fail_task and not context.get("allow"):
                context["_cancel_event"].set()  # client goes away while this task runs
            self.calls.append(task.id)
            return ResearchNote(task.id, task.title, f"Finding {task.id}",
                                citations=[{"title": f"Src {task.id}", "url": f"https://s/{task.id}", "evidence": "e"}],
                                tools_called=["search_knowledge_base"])

        def assess(question, plan, notes):
            self.assessed.append(sorted(notes))
            return self.gaps.pop(0) if self.gaps else []

        def write_stream(question, notes, sources, profile=""):
            self.written.append([n["task_id"] for n in notes])
            self.profiles.append(("write", profile))
            yield "report "
            yield str(len(notes))

        agent.planner = SimpleNamespace(plan=plan)
        agent.summarizer = SimpleNamespace(summarize=summarize)
        agent.assessor = SimpleNamespace(assess=assess)
        agent.writer = SimpleNamespace(stream=write_stream)
        return agent

    def context(self, **extra):
        return {"db": self.db, "user": SimpleNamespace(id=1), "_cancel_event": threading.Event(), **extra}

    @staticmethod
    def text(events):
        return "".join(e["value"] for e in events if e["type"] == "token")

    def test_gap_review_adds_one_research_round_before_writing(self):
        events = list(self.agent().stream("Question", [], self.context()))
        types = [e["type"] for e in events]
        self.assertEqual(sorted(self.calls), [1, 2, 3])
        self.assertEqual(self.assessed, [["1", "2"]])  # no second review: replan budget spent
        self.assertEqual(self.written, [[1, 2, 3]])
        self.assertLess(types.index("gaps"), types.index("generating"))
        plans = [e["tasks"] for e in events if e["type"] == "plan"]
        self.assertEqual([len(p) for p in plans], [2, 3])
        self.assertEqual(plans[1][2]["title"], "Follow-up: Benchmarks")
        self.assertEqual([e["id"] for e in events if e["type"] == "task_started"], [1, 2, 3])
        done = events[-1]
        self.assertEqual((done["type"], done["stop_reason"], done["resumable"]), ("done", "plan_complete", False))
        self.assertEqual([c["url"] for c in done["citations"]], ["https://s/1", "https://s/2", "https://s/3"])
        self.assertEqual(self.store.load(done["run_id"])["status"], "completed")
        self.assertEqual(self.text(events), "report 3")

    def test_sufficient_evidence_writes_without_replanning(self):
        self.gaps = [[]]
        events = list(self.agent().stream("Question", [], self.context()))
        self.assertEqual(sorted(self.calls), [1, 2])
        self.assertEqual(len([e for e in events if e["type"] == "plan"]), 1)
        self.assertNotIn("gaps", [e["type"] for e in events])

    def test_sub_tasks_really_run_in_parallel_up_to_the_limit(self):
        self.gaps = [[]]
        barrier = threading.Barrier(2, timeout=5)
        agent = self.agent()
        original = agent.summarizer.summarize

        def meet_then_summarize(task, context):
            barrier.wait()  # raises BrokenBarrierError unless both tasks are in flight together
            return original(task, context)

        agent.summarizer = SimpleNamespace(summarize=meet_then_summarize)
        with patch.object(dr, "MAX_CONCURRENCY", 2):
            events = list(agent.stream("Question", [], self.context()))
        self.assertEqual(sorted(self.calls), [1, 2])
        self.assertEqual([e["citations"] for e in events if e["type"] == "task_done"], [1, 1])

    def test_planner_receives_recent_conversation(self):
        history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
        list(self.agent().stream("Compare it with the second one", history, self.context()))
        question, seen = self.planned_with[0]
        self.assertEqual(question, "Compare it with the second one")
        self.assertEqual([m["content"] for m in seen], [f"turn {i}" for i in range(4, 10)])

    def test_long_term_user_facts_reach_the_planner_and_the_writer(self):
        self.gaps = [[]]
        facts = "\n\n## Personalisation context (from prior sessions)\n- user is new to reinforcement learning\n"
        list(self.agent().stream("Explain PPO", [], self.context(user_facts_context=facts)))
        self.assertEqual(self.profiles, [("plan", "- user is new to reinforcement learning"),
                                         ("write", "- user is new to reinforcement learning")])

    def test_writer_and_planner_prompts_carry_the_profile_as_data(self):
        prompt = ReportWriter(Mock())._build_prompt("Q", [], [], profile="- prefers PyTorch")
        self.assertIn('<untrusted_source origin="learner_profile">', prompt)
        self.assertNotIn("learner_profile", ReportWriter(Mock())._build_prompt("Q", [], []))

    def test_owner_scoped_resume_after_disconnect_skips_finished_tasks(self):
        events = list(self.agent(fail_task=2).stream("Question", [], self.context()))
        self.assertEqual(events[-1]["type"], "cancelled")
        run_id = next(e["run_id"] for e in events if e["type"] == "run")
        self.assertEqual(self.store.load(run_id)["status"], "paused")
        self.assertIsNone(ResearchRunStore(self.db, 2).load(run_id))
        self.assertIsNone(ResearchRunStore(self.db, 2).claim(run_id))

        self.calls.clear()
        resumed = list(self.agent(fail_task=2).stream("", [], self.context(resume_run_id=run_id, allow=True)))
        self.assertEqual(sorted(self.calls), [2, 3])  # task 1 came from the checkpoint
        self.assertEqual(self.text(resumed), "report 3")
        self.assertEqual(self.store.load(run_id)["status"], "completed")

        self.calls.clear()
        replay = list(self.agent().stream("", [], self.context(resume_run_id=run_id)))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.text(replay), "report 3")
        self.assertEqual(replay[-1]["type"], "done")

    def test_time_budget_pauses_with_partial_report_then_resume_finishes(self):
        with patch.object(dr.settings, "RESEARCH_MAX_SECONDS", 0):
            events = list(self.agent().stream("Question", [], self.context()))
        self.assertEqual(self.calls, [])
        self.assertIn("Unfinished research tasks", self.text(events))
        self.assertEqual((events[-1]["stop_reason"], events[-1]["resumable"]), ("time_budget", True))
        run_id = events[-1]["run_id"]
        self.assertEqual(self.store.load(run_id)["status"], "paused")

        resumed = list(self.agent().stream("", [], self.context(resume_run_id=run_id)))
        self.assertEqual(sorted(self.calls), [1, 2, 3])
        self.assertEqual(resumed[-1]["stop_reason"], "plan_complete")
        self.assertNotIn("Unfinished", self.text(resumed))
        self.assertEqual(self.store.load(run_id)["status"], "completed")

    def test_spent_token_budget_writes_from_what_it_has_instead_of_looping(self):
        from src.utils.llm_usage import request_budget
        # Only the writer's reserve is left: research and gap review must stop, and pending
        # sub-tasks must not be re-sent forever (they would come back empty every time).
        with request_budget("test", max_tokens=dr.WRITER_RESERVE):
            events = list(self.agent().stream("Question", [], self.context()))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.assessed, [])
        self.assertEqual((events[-1]["stop_reason"], events[-1]["resumable"]), ("token_budget", True))
        self.assertIn("Unfinished research tasks", self.text(events))

    def test_changed_model_versions_refuse_to_resume(self):
        meta = self.store.create("Question", {**dr.model_versions(), "schema": 1})
        self.store.set_status(meta["run_id"], "paused")
        events = list(self.agent().stream("", [], self.context(resume_run_id=meta["run_id"])))
        self.assertEqual(events, [{"type": "error", "value": "Research run model version changed; restart this run"}])

    def test_non_streaming_run_returns_report_without_a_run_record(self):
        result = self.agent().run("Question", [], self.context())
        self.assertEqual(result.reply, "report 3")
        self.assertEqual(len(result.citations), 3)
        self.assertEqual(self.db.query(ResearchRun).count(), 0)


class ComponentTests(unittest.TestCase):
    def test_planner_uses_structured_output_and_caps_tasks(self):
        requests = []
        tasks = [{"title": f"T{i}", "query": f"q{i}", "intent": "i"} for i in range(8)]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: (requests.append(kw), _completion(json.dumps({"tasks": tasks})))[1])))
        plan = TodoPlanner(client).plan("What changed?", [{"role": "user", "content": "about LoRA"}])
        self.assertEqual([t.id for t in plan], [1, 2, 3, 4, 5])
        self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
        self.assertIn("about LoRA", requests[0]["messages"][1]["content"])

    def test_planner_failure_falls_back_to_one_task(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: _completion("```json\nnot json```"))))
        plan = TodoPlanner(client).plan("What changed?")
        self.assertEqual([(t.id, t.query) for t in plan], [(1, "What changed?")])

    def test_gap_assessor_drops_gaps_that_repeat_a_tried_query(self):
        reply = {"sufficient": False, "gaps": [
            {"title": "Same", "query": "First ", "reason": "r"},
            {"title": "New", "query": "ablation results", "reason": "r"}]}
        requests = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: (requests.append(kw), _completion(json.dumps(reply)))[1])))
        plan = [{"id": 1, "title": "One", "query": "first"}]
        notes = {"1": {"summary": "nothing", "citations": []}}
        gaps = GapAssessor(client).assess("Q", plan, notes)
        self.assertEqual([g["title"] for g in gaps], ["New"])
        self.assertIn("NO EVIDENCE FOUND", requests[0]["messages"][1]["content"])

    def test_writer_failure_after_output_is_reported_once_not_rewritten(self):
        def broken_stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Partial "))])
            raise RuntimeError("connection reset")

        calls = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: (calls.append(kw), broken_stream() if kw.get("stream") else _completion("FULL"))[1])))
        notes = [{"task_id": 1, "title": "One", "summary": "s",
                  "citations": [{"title": "A", "url": "https://a", "evidence": "x"}]}]
        tokens = list(ReportWriter(client).stream("Q", notes, notes[0]["citations"]))
        self.assertEqual(tokens[0], "Partial ")
        self.assertIn("cut short", tokens[1])
        self.assertNotIn("FULL", "".join(tokens))
        self.assertEqual(len(calls), 1)
        self.assertIn("[S1] A https://a", calls[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
