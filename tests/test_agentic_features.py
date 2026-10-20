"""Plan confirmation, evidence gathering across agents, call-graph tools, live events and exercises."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents.coding_practice import (build_paper_graph, code_map, evidence_gaps,
                                        fill_exercise, make_exercise, plan_from_decision)
from src.agents.coding_services import CodingServices
from src.agents.repo_agent import RepoAgent
from src.database.models import Base, PracticeEvent, PracticeMessage, PracticeRun, User

try:  # unittest discover -s tests puts tests/ itself on the path
    from tests.test_coding_practice import LATEX_PAPER, PaperServices
    from tests.test_repo_agent import SHA, FakeClient, call, fixture_snapshot, reply
except ImportError:  # pragma: no cover
    from test_coding_practice import LATEX_PAPER, PaperServices
    from test_repo_agent import SHA, FakeClient, call, fixture_snapshot, reply

CHOICE = {"id": "q1", "detail": "Initialisation of A", "question": "How should A be initialised?",
          "options": ["Gaussian", "Uniform"], "recommended": "Gaussian"}


class PlannedServices(PaperServices):
    """Fake services whose method check leaves one choice open and records what generation saw."""

    def __init__(self):
        super().__init__()
        self.generated_with = None
        self.gather_calls = 0

    def assess_paper_method(self, evidence, objective):
        return {"feasible": True, "method": "LoRA", "reason": "", "open_choices": [CHOICE],
                "known_details": [{"detail": "B is zero", "evidence_ids": ["E1"]}], "unknown_details": [],
                "simplifications": ["Tiny random W0"]}

    def generate_paper(self, state):
        self.generated_with = state.get("plan")
        return super().generate_paper(state)

    def gather_evidence(self, state, on_event=None):
        self.gather_calls += 1
        return super().gather_evidence(state, on_event)


class PlanConfirmationTests(unittest.TestCase):
    def test_run_pauses_for_the_learner_and_resumes_with_their_decisions(self):
        services = PlannedServices()
        graph = build_paper_graph(services, InMemorySaver())
        config = {"configurable": {"thread_id": "plan"}}
        paused = graph.invoke({"paper_id": "2106.09685", "attempts": 0, "execution_enabled": False,
                               "confirm_plan": True}, config)
        self.assertIn("__interrupt__", paused)
        self.assertNotIn("code", paused)                           # nothing written before approval
        self.assertEqual(graph.get_state(config).next, ("confirm",))
        final = graph.invoke(Command(resume={"choices": {"q1": "Uniform"}, "simplifications": ["Tiny W0"],
                                             "instructions": "Keep it under 40 lines"}), config)
        self.assertEqual(final["stop_reason"], "generated_unexecuted")
        self.assertEqual(services.generated_with["choices"][0]["answer"], "Uniform")
        self.assertEqual(services.generated_with["instructions"], "Keep it under 40 lines")
        self.assertEqual(final["manifest"]["simplifications"], ["Tiny W0"])
        self.assertEqual(services.gather_calls, 1)                 # the open choice was searched for first

    def test_without_confirmation_the_graph_runs_straight_through(self):
        services = PlannedServices()
        state = build_paper_graph(services, InMemorySaver()).invoke(
            {"paper_id": "2106.09685", "attempts": 0, "execution_enabled": False},
            {"configurable": {"thread_id": "auto"}})
        self.assertEqual(state["stop_reason"], "generated_unexecuted")
        self.assertNotIn("plan", state)

    def test_plan_defaults_to_the_recommended_option_and_is_bounded(self):
        manifest = {"open_choices": [CHOICE], "simplifications": ["keep me"]}
        plan = plan_from_decision({"choices": {}, "instructions": "x" * 900}, manifest)
        self.assertEqual(plan["choices"][0]["answer"], "Gaussian")
        self.assertTrue(plan["choices"][0]["recommended"])
        self.assertEqual(plan["simplifications"], ["keep me"])
        self.assertEqual(len(plan["instructions"]), 500)

    def test_gaps_include_open_details_choices_and_refusal_reason(self):
        gaps = evidence_gaps({"feasible": False, "reason": "No loss given", "unknown_details": ["Init of A"],
                              "open_choices": [CHOICE]})
        self.assertEqual(gaps, ["Init of A", "Initialisation of A", "No loss given"])
        # later rounds leave open choices to the learner
        self.assertEqual(evidence_gaps({"feasible": True, "unknown_details": [], "open_choices": [CHOICE]},
                                       include_choices=False), [])


class GatherLoopTests(unittest.TestCase):
    def run_with(self, found_per_round, feasible_after=None):
        services = PaperServices()
        services.assess_paper_method = lambda evidence, objective: {
            "feasible": False, "method": "", "reason": "Loss missing", "known_details": [],
            "unknown_details": ["Loss"]}
        rounds = iter(found_per_round)

        def gather(state, on_event=None):
            n = state.get("gather_rounds", 0) + 1
            found = next(rounds)
            update = {"gather_rounds": n, "gathered": list(state.get("gathered") or []) + [{"new_pieces": found}]}
            if feasible_after == n:
                update.update(manifest={"feasible": True, "method": "M", "reason": "", "unknown_details": [],
                                        "known_details": [{"detail": "loss", "evidence_ids": ["E1"]}] * 2},
                              stop_reason="")
            return update

        services.gather_evidence = gather
        return build_paper_graph(services, InMemorySaver()).invoke(
            {"paper_id": "2106.09685", "attempts": 0, "execution_enabled": False},
            {"configurable": {"thread_id": str(found_per_round) + str(feasible_after)}})

    def test_a_round_that_finds_nothing_ends_the_search(self):
        state = self.run_with([0, 5])
        self.assertEqual(state["gather_rounds"], 1)
        self.assertEqual(state["stop_reason"], "method_detail_insufficient")

    def test_search_stops_at_the_round_limit(self):
        state = self.run_with([2, 2, 2])
        self.assertEqual(state["gather_rounds"], 2)

    def test_found_evidence_can_turn_a_refusal_into_a_draft(self):
        state = self.run_with([2, 0], feasible_after=1)
        self.assertEqual(state["stop_reason"], "generated_unexecuted")
        self.assertTrue(state["code"])


class GatherEvidenceServiceTests(unittest.TestCase):
    def services(self):
        db = Mock()
        db.query.return_value.filter.return_value.first.return_value = None
        services = CodingServices(db, client=Mock())
        services._public_page = lambda url, limit=800_000: ""
        return services

    def test_paper_search_adds_a_new_excerpt_and_rechecks_the_method(self):
        services = self.services()
        paper = LATEX_PAPER.replace("\\section{Related Work}",
                                    "\\section{Setup}\nThe rank r is 4 and alpha is 8 for every layer.\n\\section{Related Work}")
        state = {"paper_id": "2106.09685", "title": "LRA", "source_url": "https://arxiv.org/abs/2106.09685",
                 "evidence": [{"id": "E1", "text": "abstract", "kind": "abstract"},
                              {"id": "E2", "text": "LRA freezes the pretrained weight", "kind": "latex"}],
                 "manifest": {"feasible": True, "unknown_details": ["rank r and alpha values"], "open_choices": []},
                 "linked_repo": {}}
        rechecked = []
        events = []
        with patch.object(services, "_latex_source", return_value=paper), \
             patch.object(services, "assess_paper_method",
                          side_effect=lambda result, objective: rechecked.append(result) or
                          {"feasible": True, "method": "LRA", "known_details": [], "unknown_details": []}):
            update = services.gather_evidence(state, on_event=lambda kind, **data: events.append(kind))
        added = update["evidence"][2:]
        self.assertEqual(added[0]["id"], "E3")
        self.assertIn("rank r is 4", added[0]["text"])
        self.assertEqual(len(rechecked), 1)
        self.assertEqual(update["gathered"][0]["new_pieces"], 1)
        self.assertEqual(events[:2], ["evidence_search", "evidence_result"])

    def test_repository_agent_lines_become_code_evidence_only_when_verified(self):
        services = self.services()
        services.client = FakeClient([
            reply(calls=[call("c1", "find_symbol", {"name": "LoraLayer.update_layer"}),
                         call("c2", "read_file", {"path": "src/demo/layers.py", "start_line": 3, "end_line": 6})]),
            reply("Scaling is lora_alpha / r [src/demo/layers.py:4-5]; see also [src/demo/core.py:1-2]."),
        ])
        state = {"paper_id": "2106.09685", "title": "LRA", "evidence": [{"id": "E1", "text": "abs", "kind": "abstract"}],
                 "manifest": {"method": "LoRA", "feasible": True, "unknown_details": ["scaling factor"],
                              "open_choices": []},
                 "linked_repo": {"url": "https://github.com/example/project", "label": "Direct GitHub URL"}}
        steps = []
        with patch.object(services, "_latex_source", return_value=""), \
             patch.object(services, "_github_json", side_effect=[{"private": False, "default_branch": "main"},
                                                                  {"sha": SHA}]), \
             patch("src.agents.coding_services.load_snapshot", return_value=fixture_snapshot()), \
             patch.object(services, "assess_paper_method", return_value={"feasible": True, "known_details": [],
                                                                          "unknown_details": []}):
            update = services.gather_evidence(state, on_event=lambda kind, **data: steps.append(kind))
        code = [e for e in update["evidence"] if e["kind"] == "repo_code"]
        self.assertEqual([e["id"] for e in code], ["C1"])            # core.py:1-2 was never read
        self.assertIn("self.scaling = lora_alpha / r", code[0]["text"])
        self.assertTrue(code[0]["source"].endswith("src/demo/layers.py#L4-L5"))
        self.assertEqual(update["repo_evidence"]["used"], 1)
        self.assertIn("repo_step", steps)
        self.assertGreater(services.input_tokens, 0)

    def test_open_choices_are_validated(self):
        manifest = CodingServices._manifest({"feasible": True, "method": "M", "known_details": [], "open_choices": [
            {"detail": "Init", "question": "Which init?", "options": ["zeros", "normal"], "recommended": "other"},
            {"detail": "Size", "question": "Which size?", "options": ["only one"]},
        ]}, [])
        self.assertEqual(len(manifest["open_choices"]), 1)
        self.assertEqual(manifest["open_choices"][0]["recommended"], "zeros")
        self.assertEqual(manifest["open_choices"][0]["id"], "q1")


class CallGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = fixture_snapshot()

    def test_callees_follow_a_call_path_down_and_callers_up(self):
        symbol, calls = self.snapshot.callees("LoraLayer.update_layer")
        self.assertEqual(symbol.qualname, "LoraLayer.update_layer")
        engine = next(c for c in calls if c[0] == "Engine")
        self.assertEqual(engine[2][0].path, "src/demo/core.py")
        callers = self.snapshot.callers("run")
        self.assertEqual([(s.qualname, line) for s, line in callers], [("main", 4), ("test_run", 5)])

    def test_callees_of_a_class_are_its_constructor_calls(self):
        symbol, _ = self.snapshot.callees("Engine")
        self.assertEqual(symbol.qualname, "Engine.__init__")

    def test_references_are_whole_words_in_source_files(self):
        hits, total = self.snapshot.references("Engine")
        self.assertTrue(all(path.endswith(".py") for path, _, _ in hits))
        self.assertNotIn("README.md", {path for path, _, _ in hits})
        self.assertEqual(self.snapshot.references("Eng")[1], 0)
        self.assertGreater(self.snapshot.references("loralayer")[1], 0)  # case-insensitive fallback

    def test_agent_tool_reports_in_repo_calls_and_records_seen_lines(self):
        agent = RepoAgent(Mock(), "gpt-4o-mini", self.snapshot)
        content, summary = agent._find_callees("main")
        self.assertIn("Engine() -> Engine at src/demo/core.py:1-6", content)
        self.assertIn("calls to names not defined in this repository", content)  # print
        self.assertTrue(agent.seen["src/demo/cli.py"])

    def test_memory_from_earlier_turns_counts_as_read_and_steps_stream(self):
        client = FakeClient([reply("As established, scaling is set in [src/demo/layers.py:5-5].")])
        steps = []
        agent = RepoAgent(client, "gpt-4o-mini", self.snapshot, on_step=steps.append)
        history = [{"user": "where is scaling?", "assistant": "...", "citations": [
            {"text": "[src/demo/layers.py:4-5]", "path": "src/demo/layers.py", "start": 4, "end": 5, "verified": True}]}]
        result = agent.answer("remind me where scaling is", history, {})
        self.assertTrue(result["citations"][0]["verified"])
        self.assertIn("[src/demo/layers.py:4-5]", client.requests[0]["messages"][1]["content"])

    def test_guided_answer_is_checked_not_answered(self):
        client = FakeClient([reply("Partly right [src/demo/core.py:5-6].")])
        RepoAgent(client, "gpt-4o-mini", self.snapshot).answer("It multiplies by scale", [], {},
                                                              guided_question="What does Engine.run return?")
        opening = client.requests[0]["messages"][1]["content"]
        self.assertIn("guided-tour question: What does Engine.run return?", opening)
        self.assertIn('<untrusted_source origin="learner_answer">', opening)

    def test_live_steps_are_reported_as_tools_run(self):
        client = FakeClient([reply(calls=[call("c1", "find_callers", {"name": "run"})]), reply("Done.")])
        steps = []
        RepoAgent(client, "gpt-4o-mini", self.snapshot, on_step=steps.append).answer("who calls run?", [], {})
        self.assertEqual(steps[0]["tool"], "find_callers")
        self.assertIn("2 callers", steps[0]["summary"])


class TourTests(unittest.TestCase):
    def test_tour_stops_must_point_inside_inspected_files(self):
        services = CodingServices(Mock(), client=Mock())
        state = {"repo_url": "https://github.com/example/project", "commit_sha": SHA, "file_tree": ["src/core.py"],
                 "readme": "", "files": [{"path": "src/core.py", "content": "40 lines\n...", "lines": 40}]}
        with patch.object(services, "_json", return_value={"architecture": "core", "tour": [
                {"path": "src/core.py", "start_line": 1, "end_line": 20, "focus": "the class", "question": "What?"},
                {"path": "src/core.py", "start_line": 30, "end_line": 90, "focus": "too far", "question": "?"},
                {"path": "src/unread.py", "start_line": 1, "end_line": 5, "focus": "x", "question": "y"},
                {"path": "src/core.py", "start_line": 5, "end_line": 6, "focus": "", "question": "no focus"}]}):
            result = services.explain_repo(state)
        self.assertEqual([(s["start_line"], s["end_line"]) for s in result["tour"]], [(1, 20)])


PROGRAM = '''import numpy as np


class Adapter:
    def __init__(self, r):
        # [E2] B starts at zero
        self.B = np.zeros((4, r))
        self.A = np.random.randn(r, 4)  # [E3]

    def forward(self, x):
        return x + self.B @ self.A @ x  # [E4]


print("CHECK:PASS")
'''
MANIFEST = {"known_details": [{"detail": "B is initialised to zero", "evidence_ids": ["E2"]},
                              {"detail": "A is Gaussian", "evidence_ids": ["E3"]},
                              {"detail": "h = x + BAx", "evidence_ids": ["E4"]}]}


class CodeMapAndExerciseTests(unittest.TestCase):
    def test_markers_map_to_their_lines_and_functions(self):
        links = {link["evidence_id"]: link for link in code_map(PROGRAM)["links"]}
        self.assertEqual(links["E2"]["lines"], [6, 7])   # comment above the statement it describes
        self.assertEqual(links["E2"]["function"], "Adapter.__init__")
        self.assertEqual(links["E4"]["function"], "Adapter.forward")

    def test_exercise_blanks_cited_lines_and_keeps_assignment_targets(self):
        exercise = make_exercise(PROGRAM, MANIFEST)
        self.assertEqual([b["evidence_id"] for b in exercise["blanks"]], ["E2", "E3", "E4"])
        self.assertIn("        self.B = ____  # TODO 1 (E2): B is initialised to zero", exercise["template"])
        self.assertIn("        return ____  # TODO 3 (E4)", exercise["template"])
        self.assertEqual(exercise["answers"]["1"], "np.zeros((4, r))")
        self.assertNotIn("np.zeros((4, r))", exercise["template"])
        filled = fill_exercise(PROGRAM, exercise, exercise["answers"])
        compile(filled, "filled.py", "exec")
        self.assertIn("        self.B = np.zeros((4, r))", filled)

    def test_grading_skips_the_model_for_exact_answers(self):
        services = CodingServices(Mock(), client=Mock())
        exercise = make_exercise(PROGRAM, MANIFEST)
        state = {"evidence": [{"id": "E3", "text": "A is drawn from a Gaussian"}]}
        with patch.object(services, "_json", return_value={"results": [
                {"id": "2", "verdict": "partially", "feedback": "Right idea, wrong shape."}]}) as model:
            results = services.grade_exercise(state, exercise, {"1": "np.zeros( (4,r) )", "2": "np.random.randn(4, r)"})
        self.assertEqual([r["verdict"] for r in results], ["correct", "partially", "missing"])
        self.assertEqual(results[1]["answer"], "np.random.randn(r, 4)")
        self.assertEqual(len(json.loads(model.call_args.args[1].split("\n", 1)[1].rsplit("\n", 1)[0])), 1)


def api_client(sessions, user):
    from src.api import deps
    from src.api.routers import practice
    app = FastAPI()
    app.include_router(practice.router, prefix="/api/v1")

    def session():
        db = sessions()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[deps.get_db_session] = session
    app.dependency_overrides[deps.get_current_user] = lambda: user
    return TestClient(app)


def sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def sqlite_engine(test: unittest.TestCase):
    """
    A throwaway SQLite file where every session gets its own connection, as with Postgres in
    production. An in-memory database needs one shared connection (StaticPool), and the
    threaded endpoints and the worker interleaved their transactions on it, which made these
    tests fail now and then on slower CI machines.
    """
    folder = tempfile.TemporaryDirectory()
    test.addCleanup(folder.cleanup)
    engine = create_engine(f"sqlite:///{Path(folder.name) / 'test.db'}",
                           connect_args={"check_same_thread": False, "timeout": 30})
    test.addCleanup(engine.dispose)  # cleanups run last-in first-out: dispose before deleting the file
    return engine


class ApiTests(unittest.TestCase):
    def setUp(self):
        engine = sqlite_engine(self)
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(bind=engine, expire_on_commit=False)
        with self.sessions() as db:
            self.user = User(id=1, email="learner@example.com", hashed_password="x")
            db.add(self.user)
            db.add(User(id=2, email="other@example.com", hashed_password="x"))
            db.commit()
        from src.api.routers import practice
        self.patches = [patch.object(practice, "SessionLocal", self.sessions),
                        patch.object(practice, "_require_available")]
        for item in self.patches:
            item.start()
        self.client = api_client(self.sessions, self.user)

    def tearDown(self):
        for item in self.patches:
            item.stop()

    def add_run(self, run_id, kind="paper", status="completed", user_id=1):
        with self.sessions() as db:
            db.add(PracticeRun(run_id=run_id, user_id=user_id, kind=kind, source="x", status=status, stage=status))
            db.commit()

    def test_plan_is_accepted_only_while_waiting(self):
        self.add_run("waiting-run", status="waiting")
        response = self.client.post("/api/v1/practice/runs/waiting-run/plan",
                                    json={"choices": {"q1": "Uniform"}, "instructions": "short"})
        self.assertEqual(response.json(), {"status": "queued"})
        with self.sessions() as db:
            row = db.get(PracticeRun, "waiting-run")
            self.assertEqual((row.status, json.loads(row.resume_json)["choices"]), ("queued", {"q1": "Uniform"}))
        self.assertEqual(self.client.post("/api/v1/practice/runs/waiting-run/plan", json={}).status_code, 409)

    def test_event_stream_replays_events_then_ends_with_the_status(self):
        self.add_run("done-run")
        with self.sessions() as db:
            db.add(PracticeEvent(run_id="done-run", kind="evidence", payload_json='{"found": 10}'))
            db.add(PracticeEvent(run_id="done-run", kind="review", payload_json='{"issues": 0}'))
            db.commit()
        events = sse(self.client.get("/api/v1/practice/runs/done-run/events/stream"))
        self.assertEqual([e.get("kind") for e in events if e["type"] == "event"], ["evidence", "review"])
        self.assertEqual(events[-1], {"type": "end", "status": "completed"})
        after = sse(self.client.get(f"/api/v1/practice/runs/done-run/events/stream?after={events[0]['id']}"))
        self.assertEqual([e.get("kind") for e in after if e["type"] == "event"], ["review"])

    def test_run_list_shows_titles_from_events(self):
        self.add_run("titled-run")
        self.add_run("repo-list", kind="repo")
        with self.sessions() as db:
            db.add(PracticeEvent(run_id="titled-run", kind="evidence", payload_json='{"title": "LoRA"}'))
            db.commit()
        runs = {r["run_id"]: r for r in self.client.get("/api/v1/practice/runs").json()["runs"]}
        self.assertEqual(runs["titled-run"]["title"], "LoRA")
        self.assertEqual(runs["repo-list"]["title"], "x")
        self.assertTrue(runs["titled-run"]["created_at"])

    def test_event_stream_is_owner_only(self):
        self.add_run("someone-else", user_id=2)
        self.assertEqual(self.client.get("/api/v1/practice/runs/someone-else/events/stream").status_code, 404)

    def test_message_stream_pushes_agent_steps_then_the_saved_answer(self):
        self.add_run("repo-run", kind="repo")
        state = {"repo_url": "https://github.com/example/project", "commit_sha": SHA, "tour": []}

        def fake_answer(self_, state_, message, history, guided_question="", on_step=None):
            on_step({"tool": "find_symbol", "args": {"name": "Engine"}, "summary": "1 matches"})
            return {"answer": "See [src/demo/core.py:1-6].", "trace": [{"tool": "find_symbol"}], "tool_calls": 1,
                    "citations": [{"text": "[src/demo/core.py:1-6]", "verified": True}]}

        from src.api.routers import practice
        with patch.object(practice, "graph_state", return_value=state), \
             patch.object(CodingServices, "answer_repo_question", fake_answer):
            events = sse(self.client.post("/api/v1/practice/runs/repo-run/messages/stream",
                                          json={"idempotency_key": "stream-123", "message": "Where is Engine?"}))
        self.assertEqual([e["type"] for e in events], ["started", "step", "answer"])
        self.assertEqual(events[1]["tool"], "find_symbol")
        self.assertTrue(events[2]["message"]["citations"][0]["verified"])
        with self.sessions() as db:
            self.assertEqual(db.query(PracticeMessage).one().response, "See [src/demo/core.py:1-6].")

    def test_file_viewer_reads_the_pinned_snapshot_with_a_window(self):
        self.add_run("repo-view", kind="repo")
        from src.api.routers import practice
        state = {"repo_url": "https://github.com/example/project", "commit_sha": SHA, "focus": ""}
        with patch.object(practice, "graph_state", return_value=state), \
             patch("src.agents.repo_snapshot.load_snapshot", return_value=fixture_snapshot()):
            body = self.client.get("/api/v1/practice/runs/repo-view/file",
                                   params={"path": "src/demo/layers.py", "start": 4, "end": 5}).json()
            missing = self.client.get("/api/v1/practice/runs/repo-view/file", params={"path": "nope.py"})
        self.assertEqual((body["start"], body["highlight"], body["total"]), (1, [4, 5], 6))
        self.assertEqual(body["lines"][3], "    def update_layer(self, r, lora_alpha):")
        self.assertEqual(missing.status_code, 404)

    def test_exercise_hides_answers_and_grades(self):
        self.add_run("paper-run")
        from src.api.routers import practice
        state = {"code": PROGRAM, "manifest": MANIFEST, "evidence": []}
        with patch.object(practice, "graph_state", return_value=state), \
             patch.object(practice, "_worker_available", return_value=False):
            exercise = self.client.get("/api/v1/practice/runs/paper-run/exercise").json()
            graded = self.client.post("/api/v1/practice/runs/paper-run/exercise/grade",
                                      json={"answers": {"1": "np.zeros((4, r))", "2": "np.random.randn(r, 4)",
                                                        "3": "x + self.B @ self.A @ x"}}).json()
        self.assertNotIn("answers", exercise)
        self.assertNotIn("np.zeros((4, r))", json.dumps(exercise))
        self.assertEqual([r["verdict"] for r in graded["results"]], ["correct"] * 3)
        self.assertIsNone(graded["execution"])


class WorkerInterruptTests(unittest.TestCase):
    def test_graph_nodes_do_not_share_the_runtime_session(self):
        """Nodes run on LangGraph's threads while the runtime records events; one shared session
        failed in production with "This session is in 'prepared' state"."""
        from src.agents import coding_runtime
        engine = sqlite_engine(self)
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with sessions() as db:
            db.add(User(id=1, email="own@example.com", hashed_password="x"))
            db.add(PracticeRun(run_id="own-run", user_id=1, kind="paper", source="2106.09685",
                               status="running", stage="starting"))
            db.commit()
        opened, given = [], {}

        def open_session():
            opened.append(sessions())
            return opened[-1]

        def services(db, **kwargs):
            given["db"] = db
            raise RuntimeError("stop once the services exist")

        with patch.object(coding_runtime, "SessionLocal", open_session), \
             patch("src.api.deps.get_embedding_manager", return_value=object()), \
             patch("src.agents.coding_services.CodingServices", side_effect=services):
            coding_runtime.run_graph_job("own-run")
        self.assertEqual(len(opened), 2)
        self.assertIs(given["db"], opened[1])  # not opened[0], which records events and status

    def test_worker_marks_the_run_waiting_then_finishes_after_the_plan(self):
        from src.agents import coding_runtime
        engine = sqlite_engine(self)
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with sessions() as db:
            db.add(User(id=1, email="plan@example.com", hashed_password="x"))
            db.add(PracticeRun(run_id="plan-run", user_id=1, kind="paper", source="2106.09685", status="running",
                               stage="starting", request_json=json.dumps({"confirm_plan": True})))
            db.commit()
        saver = InMemorySaver()
        services = PlannedServices()
        with patch.object(coding_runtime, "SessionLocal", sessions), \
             patch("src.api.deps.get_embedding_manager", return_value=object()), \
             patch.object(coding_runtime, "get_checkpointer", return_value=saver), \
             patch("src.agents.coding_services.CodingServices", return_value=services):
            coding_runtime.run_graph_job("plan-run")
            with sessions() as db:
                row = db.get(PracticeRun, "plan-run")
                self.assertEqual((row.status, row.stage), ("waiting", "confirm_plan"))
                kinds = [e.kind for e in db.query(PracticeEvent).order_by(PracticeEvent.id)]
                self.assertEqual(kinds[0], "started")
                self.assertIn("manifest", kinds)
                self.assertEqual(kinds[-1], "awaiting_confirmation")
                row.resume_json = json.dumps({"choices": {"q1": "Uniform"}})
                row.status = "running"
                db.commit()
            coding_runtime.run_graph_job("plan-run")
            with sessions() as db:
                self.assertEqual(db.get(PracticeRun, "plan-run").status, "completed")
                kinds = [e.kind for e in db.query(PracticeEvent).order_by(PracticeEvent.id)]
                self.assertIn("resumed", kinds)
                self.assertIn("plan_confirmed", kinds)
                self.assertEqual(kinds[-1], "finished")
        self.assertEqual(services.generated_with["choices"][0]["answer"], "Uniform")


if __name__ == "__main__":
    unittest.main()
