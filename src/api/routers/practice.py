"""Authenticated Coding Agent API backed by LangGraph checkpoints.

Execution is deliberately unavailable unless an isolated HTTPS worker is
configured. No endpoint executes submitted Python inside the API container.
"""
import asyncio
import json
import logging
import threading
import time as clock
import uuid
from datetime import datetime, time, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from src.agents import method_notes
from src.api.demo import demo_quota, forbid_demo, is_demo_user
from src.api.deps import get_current_user, get_db_session
from src.agents.coding_practice import code_map
from src.agents.coding_runtime import graph_state, record_event
from src.database.models import (PracticeComparison, PracticeEvent, PracticeExecution, PracticeMessage, PracticeRun,
                                 SessionLocal, User)
from src.utils.config import settings

router = APIRouter(prefix="/practice", tags=["Coding Agent"])
logger = logging.getLogger(__name__)
MAX_DAILY_RUNS = 5
MAX_EXERCISE_GRADES = 30
EVENT_STREAM_SECONDS = 900
TERMINAL = ("completed", "failed", "cancelled", "waiting")


class PaperRequest(BaseModel):
    arxiv_id: str = Field(pattern=r"^\d{4}\.\d{4,5}(?:v\d+)?$")
    objective: str = Field(default="", max_length=300)
    confirm_plan: bool = True   # pause after the method check so the learner can approve the plan


class PlanRequest(BaseModel):
    choices: Dict[str, str] = Field(default_factory=dict, max_length=5)
    simplifications: List[str] = Field(default_factory=list, max_length=8)
    instructions: str = Field(default="", max_length=500)


class GradeRequest(BaseModel):
    answers: Dict[str, str] = Field(max_length=5)


class RepoRequest(BaseModel):
    github_url: str = Field(max_length=300)


class ActionRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=80)
    code: Optional[str] = Field(default=None, max_length=20_000)


class MessageRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=80)
    message: str = Field(min_length=1, max_length=2000)
    guided_question: str = Field(default="", max_length=240)


def _available() -> bool:
    return bool(settings.PRACTICE_ENABLED and settings.OPENAI_API_KEY)


def _worker_available() -> bool:
    return bool(_available() and settings.PRACTICE_WORKER_URL
                and settings.PRACTICE_WORKER_URL.startswith("https://")
                and settings.PRACTICE_WORKER_TOKEN)


def _require_available():
    if not _available():
        raise HTTPException(status_code=503, detail="Coding Agent is not enabled")


def _require_execution_available():
    if not _worker_available():
        raise HTTPException(status_code=503, detail="Running code needs a separately configured isolated worker")


def _owned_run(db: Session, run_id: str, user_id: int) -> PracticeRun:
    row = db.query(PracticeRun).filter(PracticeRun.run_id == run_id,
                                       PracticeRun.user_id == user_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Coding Agent run not found")
    return row


def _require_run_quota(db: Session, user: User) -> None:
    # The demo account is shared, so its limits are everyone's: say so rather than
    # telling a visitor to finish a run someone else started.
    demo = is_demo_user(user)
    today = datetime.combine(datetime.now(timezone.utc).date(), time.min, tzinfo=timezone.utc)
    if db.query(PracticeRun).filter(PracticeRun.user_id == user.id,
            PracticeRun.created_at >= today).count() >= MAX_DAILY_RUNS:
        raise HTTPException(status_code=429, detail=(
            "The demo's new Coding Agent runs for today are used up. Open one of the finished runs, "
            "or create a free account to start your own." if demo else "Daily Coding Agent run limit reached"))
    if db.query(PracticeRun).filter(PracticeRun.user_id == user.id,
            PracticeRun.status.in_(["queued", "running"])).count():
        raise HTTPException(status_code=429, detail=(
            "Another demo visitor's run is in progress. Open one of the finished runs meanwhile, "
            "or create a free account to run your own." if demo else "Finish the current Coding Agent run first"))


def _public_state(kind: str, state: Dict[str, Any]) -> Dict[str, Any]:
    if kind == "paper":
        return {key: state.get(key) for key in
                ("paper_id", "objective", "execution_enabled", "title", "source_url", "evidence",
                 "evidence_source", "code",
                 "explanation", "attempts", "execution", "stop_reason", "questions",
                 "manifest", "timeline", "faithfulness", "detail_review", "draft_revision", "linked_repo",
                 "confirm_plan", "plan", "gathered", "repo_evidence", "personalized")} \
            | {"code_map": code_map(state["code"]) if state.get("code") else None,
               "method_notes": [note.name for note in method_notes.by_ids(state.get("method_notes") or [])]}
    return {"repo_url": state.get("repo_url"), "commit_sha": state.get("commit_sha"),
            "focus": state.get("focus", ""), "overview": state.get("overview"),
            "inspected_files": [item["path"] for item in state.get("files", [])],
            "file_tree": state.get("file_tree", []), "tree_truncated": state.get("tree_truncated", False),
            "architecture": state.get("architecture"), "entry_points": state.get("entry_points", []),
            "file_roles": state.get("file_roles", []), "limitations": state.get("limitations"),
            "linked_paper": state.get("linked_paper", {}), "tour": state.get("tour", [])}


def _message_view(row: PracticeMessage) -> Dict[str, Any]:
    return {"id": row.id, "prompt": row.prompt, "response": row.response,
            "guided_question": row.guided_question,
            "trace": json.loads(row.trace_json) if row.trace_json else [],
            "citations": json.loads(row.citations_json) if row.citations_json else []}


def _event_view(row: PracticeEvent) -> Dict[str, Any]:
    return {"id": row.id, "kind": row.kind, "data": json.loads(row.payload_json or "{}"),
            "at": row.created_at.isoformat() if row.created_at else None}


def _snapshot(row: PracticeRun) -> Dict[str, Any]:
    values = graph_state(row.run_id, row.kind)
    return {"run_id": row.run_id, "kind": row.kind, "source": row.source,
            "status": row.status, "stage": row.stage, "error": row.error,
            "input_tokens": row.input_tokens or 0, "output_tokens": row.output_tokens or 0,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "result": _public_state(row.kind, values)}


@router.get("/capabilities")
def capabilities(current_user: User = Depends(get_current_user)):
    return {"enabled": _available(),
            "paper_enabled": _available(), "repo_enabled": _available(),
            "paper_execution_enabled": _worker_available(),
            "modes": ["paper", "repo"],
            "reason": None if _available() else "Coding Agent is not enabled"}


@router.post("/paper", status_code=202)
def create_paper(request: PaperRequest, current_user: User = Depends(demo_quota),
                 db: Session = Depends(get_db_session)):
    _require_available()
    _require_run_quota(db, current_user)
    run_id = str(uuid.uuid4())
    db.add(PracticeRun(run_id=run_id, user_id=current_user.id, kind="paper", source=request.arxiv_id,
                       request_json=json.dumps({"objective": request.objective,
                                                "execution_enabled": _worker_available(),
                                                "confirm_plan": request.confirm_plan}),
                       status="queued", stage="queued"))
    db.commit()
    return {"run_id": run_id}


@router.post("/repo", status_code=202)
def create_repo(request: RepoRequest, current_user: User = Depends(demo_quota),
                db: Session = Depends(get_db_session)):
    _require_available()
    _require_run_quota(db, current_user)
    from src.agents.coding_services import CodingServices
    try:
        CodingServices._repo_target(request.github_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = str(uuid.uuid4())
    db.add(PracticeRun(run_id=run_id, user_id=current_user.id, kind="repo",
                       source=request.github_url, status="queued", stage="queued"))
    db.commit()
    return {"run_id": run_id}


@router.get("/runs")
def list_runs(current_user: User = Depends(get_current_user),
              db: Session = Depends(get_db_session)):
    rows = db.query(PracticeRun).filter(PracticeRun.user_id == current_user.id).order_by(
        PracticeRun.created_at.desc()).limit(30).all()
    # Paper titles come from the run's own 'evidence' event, so listing needs no checkpoint reads.
    titles = {}
    if rows:
        for event in db.query(PracticeEvent).filter(PracticeEvent.run_id.in_([r.run_id for r in rows]),
                                                    PracticeEvent.kind == "evidence"):
            title = json.loads(event.payload_json or "{}").get("title")
            if title:
                titles[event.run_id] = str(title)[:200]
    return {"runs": [{"run_id": row.run_id, "kind": row.kind, "source": row.source,
                      "title": titles.get(row.run_id) or (row.source.replace("https://github.com/", "")
                                                          if row.kind == "repo" else row.source),
                      "status": row.status, "stage": row.stage,
                      "created_at": row.created_at.isoformat() if row.created_at else None,
                      "updated_at": row.updated_at.isoformat() if row.updated_at else None} for row in rows]}


@router.get("/runs/{run_id}")
def get_run(run_id: str, current_user: User = Depends(get_current_user),
            db: Session = Depends(get_db_session)):
    response = _snapshot(_owned_run(db, run_id, current_user.id))
    generated_code = response["result"].get("code") if response["kind"] == "paper" else None
    rows = db.query(PracticeExecution).filter(PracticeExecution.run_id == run_id,
        PracticeExecution.user_id == current_user.id).order_by(PracticeExecution.id.desc()).limit(10).all()
    response["user_executions"] = [{"status": row.status, "code": row.code,
        "evidence_status": "agent_version" if row.code == generated_code else "user_modified_unverified",
        "result": json.loads(row.result_json) if row.result_json else None} for row in rows]
    messages = db.query(PracticeMessage).filter(PracticeMessage.run_id == run_id,
        PracticeMessage.user_id == current_user.id).order_by(PracticeMessage.id).limit(30).all()
    response["messages"] = [_message_view(row) for row in messages]
    events = db.query(PracticeEvent).filter(PracticeEvent.run_id == run_id).order_by(PracticeEvent.id.desc()).limit(120).all()
    response["events"] = [_event_view(row) for row in reversed(events)]
    comparison = db.get(PracticeComparison, run_id)
    response["comparison"] = (json.loads(comparison.result_json) if comparison and comparison.result_json else None)
    response["comparison_status"] = comparison.status if comparison else None
    response["comparison_error"] = comparison.error if comparison else None
    return response


@router.post("/runs/{run_id}/comparison", status_code=202)
def compare_with_repository(run_id: str, current_user: User = Depends(demo_quota),
                            db: Session = Depends(get_db_session)):
    row = _owned_run(db, run_id, current_user.id)
    _require_available()
    if row.kind != "paper" or row.status != "completed":
        raise HTTPException(status_code=409, detail="Finish a paper implementation first")
    existing = db.get(PracticeComparison, run_id)
    if existing:
        if existing.status in ("queued", "running"):
            return {"status": existing.status}
        if existing.status == "completed":
            return {"status": "completed", "result": json.loads(existing.result_json)}
        db.delete(existing)
        db.commit()
    state = _snapshot(row)["result"]
    if not state.get("code") or not (state.get("linked_repo") or {}).get("url"):
        raise HTTPException(status_code=409, detail="No paper-linked repository or teaching code is available")
    comparison = PracticeComparison(run_id=run_id, user_id=current_user.id, status="queued")
    db.add(comparison)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Comparison is already in progress")
    return {"status": "queued"}


def _start_message(db: Session, run_id: str, request: MessageRequest, user: User):
    """Validate a discussion turn and reserve its row; returns (run, state, history, message) or an earlier reply."""
    row = _owned_run(db, run_id, user.id)
    _require_available()  # Reading an existing run only needs the model, not the sandbox worker.
    if row.status != "completed":
        raise HTTPException(status_code=409, detail="Wait for the Coding Agent run to finish")
    existing = db.query(PracticeMessage).filter(PracticeMessage.run_id == run_id,
        PracticeMessage.idempotency_key == request.idempotency_key).first()
    if existing:
        if existing.response is None:
            raise HTTPException(status_code=409, detail="This reply is still being generated")
        return None, None, None, existing
    if db.query(PracticeMessage).filter(PracticeMessage.run_id == run_id).count() >= 20:
        raise HTTPException(status_code=429, detail="This discussion reached its 20-message limit")
    if db.query(PracticeMessage).filter(PracticeMessage.run_id == run_id,
                                       PracticeMessage.response.is_(None)).count():
        raise HTTPException(status_code=409, detail="Wait for the current reply")
    state = graph_state(run_id, row.kind)
    if row.kind == "paper" and not state.get("code"):
        raise HTTPException(status_code=409, detail="No program was generated for this paper")
    allowed = state.get("questions", []) if row.kind == "paper" else [s["question"] for s in state.get("tour", [])]
    if request.guided_question and request.guided_question not in allowed:
        raise HTTPException(status_code=400, detail="Unknown guided question")
    earlier = db.query(PracticeMessage).filter(PracticeMessage.run_id == run_id,
        PracticeMessage.user_id == user.id).order_by(PracticeMessage.id.desc()).limit(8).all()
    history = [{"user": item.prompt, "assistant": item.response or "",
                "citations": json.loads(item.citations_json) if item.citations_json else []}
               for item in reversed(earlier)]
    message = PracticeMessage(run_id=run_id, user_id=user.id,
        idempotency_key=request.idempotency_key, prompt=request.message,
        guided_question=request.guided_question or None)
    db.add(message)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="This message is already in progress")
    return row, state, history, message


def _answer_message(db: Session, row: PracticeRun, message: PracticeMessage, state: Dict[str, Any],
                    history: List[Dict[str, Any]], user: User,
                    on_step: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    from src.agents.coding_services import CodingServices
    services = CodingServices(db, worker=None, user_id=user.id)
    try:
        if row.kind == "repo":
            # Repository questions are investigated with read-only tools over the pinned snapshot.
            result = services.answer_repo_question(state, message.prompt, history,
                                                   message.guided_question or "", on_step=on_step)
            response = result["answer"]
            message.trace_json = json.dumps(result["trace"])
            message.citations_json = json.dumps(result["citations"])
        else:
            response = services.answer_about_run(row.kind, state, message.prompt, history,
                                                 message.guided_question or "")
        message.response = response or "I could not answer from the available code evidence."
        row.input_tokens = (row.input_tokens or 0) + services.input_tokens
        row.output_tokens = (row.output_tokens or 0) + services.output_tokens
        db.commit()
    except Exception:
        db.rollback()
        db.delete(message)
        db.commit()
        logger.exception("Coding Agent reply failed for run %s", row.run_id)
        raise HTTPException(status_code=503, detail="Could not generate a reply; please try again")
    return _message_view(message)


@router.post("/runs/{run_id}/messages")
def send_message(run_id: str, request: MessageRequest, current_user: User = Depends(demo_quota),
                 db: Session = Depends(get_db_session)):
    row, state, history, message = _start_message(db, run_id, request, current_user)
    if row is None:
        return _message_view(message)
    return _answer_message(db, row, message, state, history, current_user)


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/runs/{run_id}/messages/stream")
async def stream_message(run_id: str, request: MessageRequest, http_request: Request,
                         current_user: User = Depends(demo_quota), db: Session = Depends(get_db_session)):
    """The same turn as POST /messages, with each agent tool step pushed as it happens."""
    row, state, history, message = await run_in_threadpool(_start_message, db, run_id, request, current_user)
    if row is None:
        view = _message_view(message)
        return StreamingResponse(iter([_sse({"type": "answer", "message": view})]),
                                 media_type="text/event-stream", headers=SSE_HEADERS)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    message_id, kind = message.id, row.kind

    def work():
        session = SessionLocal()
        try:
            run = session.get(PracticeRun, run_id)
            own = session.get(PracticeMessage, message_id)
            view = _answer_message(session, run, own, state, history, current_user,
                                   on_step=lambda step: loop.call_soon_threadsafe(queue.put_nowait, {"type": "step", **step}))
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "answer", "message": view})
        except HTTPException as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "detail": exc.detail})
        except Exception:
            logger.exception("Streaming reply failed for run %s", run_id)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "detail": "Could not generate a reply"})
        finally:
            session.close()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=work, daemon=True).start()

    async def events():
        yield _sse({"type": "started", "kind": kind})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if event is None:
                break
            yield _sse(event)

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


def _poll_events(run_id: str, after: int):
    with SessionLocal() as session:
        rows = session.query(PracticeEvent).filter(PracticeEvent.run_id == run_id, PracticeEvent.id > after) \
            .order_by(PracticeEvent.id).limit(100).all()
        run = session.get(PracticeRun, run_id)
        return [_event_view(row) for row in rows], (run.status, run.stage) if run else ("failed", "missing")


@router.get("/runs/{run_id}/events/stream")
async def stream_run_events(run_id: str, http_request: Request, after: int = 0,
                            current_user: User = Depends(get_current_user), db: Session = Depends(get_db_session)):
    """Server-sent progress events of a run until it finishes or waits for the learner."""
    _owned_run(db, run_id, current_user.id)

    async def events():
        cursor, last, deadline, quiet = after, None, clock.monotonic() + EVENT_STREAM_SECONDS, 0
        while clock.monotonic() < deadline and not await http_request.is_disconnected():
            rows, (run_status, stage) = await run_in_threadpool(_poll_events, run_id, cursor)
            for row in rows:
                cursor = row["id"]
                yield _sse({"type": "event", **row})
            if (run_status, stage) != last:
                last = (run_status, stage)
                yield _sse({"type": "status", "status": run_status, "stage": stage})
            if run_status in TERMINAL and not rows:
                yield _sse({"type": "end", "status": run_status})
                break
            quiet = 0 if rows else quiet + 1
            if quiet and quiet % 20 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.7)

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/runs/{run_id}/plan")
def confirm_plan(run_id: str, request: PlanRequest, current_user: User = Depends(demo_quota),
                 db: Session = Depends(get_db_session)):
    """The learner's decisions for a run paused at its plan check; the worker resumes it with them."""
    row = _owned_run(db, run_id, current_user.id)
    _require_available()
    if row.kind != "paper" or row.status != "waiting":
        raise HTTPException(status_code=409, detail="This run is not waiting for a plan decision")
    decision = {"choices": {str(k)[:8]: str(v).strip()[:200] for k, v in request.choices.items()},
                "simplifications": [str(x).strip()[:200] for x in request.simplifications if str(x).strip()],
                "instructions": request.instructions.strip()}
    updated = db.execute(update(PracticeRun).where(PracticeRun.run_id == run_id, PracticeRun.status == "waiting")
                         .values(resume_json=json.dumps(decision), status="queued", stage="plan_confirmed",
                                 updated_at=datetime.now(timezone.utc))).rowcount
    db.commit()
    if not updated:
        raise HTTPException(status_code=409, detail="This run is not waiting for a plan decision")
    record_event(db, run_id, "plan_submitted", {"choices": len(decision["choices"]),
                                               "instructions": bool(decision["instructions"])})
    return {"status": "queued"}


def _completed_state(db: Session, run_id: str, user: User, kind: str) -> Dict[str, Any]:
    row = _owned_run(db, run_id, user.id)
    if row.kind != kind or row.status != "completed":
        raise HTTPException(status_code=409, detail=f"This needs a finished {kind} run")
    return graph_state(run_id, row.kind)


@router.get("/runs/{run_id}/file")
def read_repo_file(run_id: str, path: str, start: int = 1, end: int = 0,
                   current_user: User = Depends(get_current_user), db: Session = Depends(get_db_session)):
    """Lines of one file of the run's pinned snapshot, for the in-page code viewer."""
    state = _completed_state(db, run_id, current_user, "repo")
    from src.agents.coding_services import CodingServices
    from src.agents.repo_snapshot import load_snapshot
    try:
        owner, repo = CodingServices._repo_parts(state["repo_url"])
        snapshot = load_snapshot(owner, repo, state["commit_sha"], state.get("focus") or "")
    except Exception as exc:  # SnapshotError, network or cache failures alike
        raise HTTPException(status_code=503, detail="The repository snapshot is unavailable") from exc
    path = path.strip().strip("/")
    if not snapshot.ensure_text(path):  # files outside the snapshot budget are fetched on demand
        raise HTTPException(status_code=404, detail="That file is not readable in this snapshot")
    lines = snapshot.lines(path)
    start = min(max(1, start), max(len(lines), 1))
    end = min(max(end, start), len(lines)) if lines else start
    window_start = max(1, start - 30)
    window_end = min(len(lines), max(end + 30, window_start + 60), window_start + 399)
    return {"path": path, "start": window_start, "end": window_end, "total": len(lines),
            "highlight": [start, end], "lines": [line[:400] for line in lines[window_start - 1:window_end]],
            "url": snapshot.url(path, start, end)}


@router.get("/runs/{run_id}/exercise")
def get_exercise(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db_session)):
    """Fill-in-the-blank version of the generated program; answers stay on the server."""
    from src.agents.coding_practice import make_exercise
    state = _completed_state(db, run_id, current_user, "paper")
    exercise = make_exercise(state.get("code") or "", state.get("manifest") or {})
    if exercise is None:
        raise HTTPException(status_code=404, detail="No evidence-marked lines in this program can be turned into blanks")
    return {"template": exercise["template"], "blanks": exercise["blanks"]}


@router.post("/runs/{run_id}/exercise/grade")
def grade_exercise(run_id: str, request: GradeRequest, current_user: User = Depends(demo_quota),
                   db: Session = Depends(get_db_session)):
    from src.agents.coding_practice import PracticeError, _check_code, fill_exercise, make_exercise
    from src.agents.coding_services import CodingServices, WorkerClient
    state = _completed_state(db, run_id, current_user, "paper")
    _require_available()
    if db.query(PracticeEvent).filter(PracticeEvent.run_id == run_id,
                                     PracticeEvent.kind == "exercise_graded").count() >= MAX_EXERCISE_GRADES:
        raise HTTPException(status_code=429, detail="This exercise reached its grading limit")
    exercise = make_exercise(state.get("code") or "", state.get("manifest") or {})
    if exercise is None:
        raise HTTPException(status_code=404, detail="This program has no exercise")
    answers = {str(k): str(v)[:300] for k, v in request.answers.items()}
    services = CodingServices(db, worker=None, user_id=current_user.id)
    results = services.grade_exercise(state, exercise, answers)
    execution = None
    if _worker_available() and all(r["verdict"] != "missing" for r in results):
        filled = fill_exercise(state["code"], exercise, answers)
        try:
            _check_code(filled)
            execution = WorkerClient().execute(filled)
        except (PracticeError, SyntaxError) as exc:
            execution = {"passed": False, "stderr": f"Your program does not compile: {exc}"}
        except Exception:
            execution = {"passed": False, "error": "Sandbox worker unavailable"}
    row = db.get(PracticeRun, run_id)
    row.input_tokens = (row.input_tokens or 0) + services.input_tokens
    row.output_tokens = (row.output_tokens or 0) + services.output_tokens
    db.commit()
    record_event(db, run_id, "exercise_graded", {"correct": sum(r["verdict"] == "correct" for r in results),
                                                "blanks": len(results)})
    return {"results": results, "execution": execution}


@router.post("/runs/{run_id}/cancel")
def cancel(run_id: str, current_user: User = Depends(forbid_demo),
           db: Session = Depends(get_db_session)):
    row = _owned_run(db, run_id, current_user.id)
    if row.status not in ("queued", "running", "waiting", "cancelled"):
        raise HTTPException(status_code=409, detail="Run is already finished")
    db.execute(update(PracticeRun).where(PracticeRun.run_id == run_id,
        PracticeRun.user_id == current_user.id,
        PracticeRun.status.in_(["queued", "running", "waiting"]))
        .values(status="cancelled", stage="cancelled", updated_at=datetime.now(timezone.utc)))
    db.commit()
    return {"status": "cancelled"}


@router.post("/runs/{run_id}/executions")
def execute_edited_paper(run_id: str, request: ActionRequest,
                         current_user: User = Depends(demo_quota),
                         db: Session = Depends(get_db_session)):
    _require_execution_available()
    row = _owned_run(db, run_id, current_user.id)
    if row.kind != "paper" or row.status != "completed":
        raise HTTPException(status_code=409, detail="Paper practice must finish before running your edit")
    generated_code = _snapshot(row)["result"].get("code")
    from src.agents.coding_practice import PracticeError, _check_code
    from src.agents.coding_services import WorkerClient
    try:
        _check_code(request.code or "")
    except (PracticeError, SyntaxError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    existing = db.query(PracticeExecution).filter(PracticeExecution.run_id == run_id,
        PracticeExecution.idempotency_key == request.idempotency_key).first()
    if existing:
        return {"status": existing.status,
                "result": json.loads(existing.result_json) if existing.result_json else None,
                "evidence_status": "agent_version" if existing.code == generated_code
                                   else "user_modified_unverified"}
    if db.query(PracticeExecution).filter(PracticeExecution.run_id == run_id).count() >= 10:
        raise HTTPException(status_code=429, detail="This run reached its 10-edit execution limit")
    execution = PracticeExecution(run_id=run_id, user_id=current_user.id,
        idempotency_key=request.idempotency_key, code=request.code, status="running")
    db.add(execution)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="That execution is already in progress")
    try:
        result = WorkerClient().execute(request.code)
        execution.status = "completed"
        execution.result_json = json.dumps(result)
    except Exception:
        logger.exception("User execution failed for run %s", run_id)
        execution.status = "failed"
        execution.result_json = json.dumps({"error": "Sandbox worker unavailable"})
    db.commit()
    return {"status": execution.status, "result": json.loads(execution.result_json),
            "evidence_status": "agent_version" if request.code == generated_code
                               else "user_modified_unverified"}
