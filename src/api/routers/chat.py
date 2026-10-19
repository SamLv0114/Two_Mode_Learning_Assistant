"""
Chat endpoint: unified conversational interface with agent routing.

POST   /chat/stream     — stream a routed agent response
GET    /chat/history/{session_id}    — retrieve conversation history
DELETE /chat/history/{session_id}    — clear a session
GET    /chat/trace/{session_id}      — retrieve agent tool-call trace
POST   /chat/eval/run                — batch LLM-as-Judge evaluation

Each turn:
  - injects long-term user facts (UserFactMemory), extracted in a background
    thread after every completed answer
  - selects semantically relevant history (ContextBuilder) instead of last-N
  - records tool calls to Redis through the ToolAwareAgent wrapper
"""
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from contextlib import closing
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.demo import demo_quota, forbid_demo
from src.api.deps import get_db_session, get_current_user, get_embedding_manager
from src.database.models import User
from src.models.embeddings import EmbeddingManager
from src.utils.config import settings
from src.agents.router import AgentRouter
from src.agents.memory import ConversationMemory, UserFactMemory
from src.agents.context_builder import ContextBuilder
from src.agents.tool_aware_agent import ToolAwareAgent
from src.agents.metrics import record_request
from src.rag.retriever import Retriever
from src.utils.llm_usage import request_budget
from src.utils.redis_client import get_redis

router = APIRouter(prefix="/chat", tags=["Chat Agent"])
logger = logging.getLogger(__name__)

_agent_router: Optional[AgentRouter] = None
_fact_memory = UserFactMemory()


def _get_agent_router(embedding_manager: EmbeddingManager) -> AgentRouter:
    global _agent_router
    if _agent_router is None:
        _agent_router = AgentRouter(embedding_model=getattr(embedding_manager, "model", None))
    return _agent_router


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    resume_run_id: Optional[str] = None


class EvalTestCase(BaseModel):
    question: str
    expected_intent: Optional[str] = None


class EvalRunRequest(BaseModel):
    test_cases: List[EvalTestCase]


class EvalRunResponse(BaseModel):
    results: List[dict]
    intent_accuracy: Optional[float]
    avg_scores: dict
    total_cases: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_facts_bg(
    user_id: int,
    user_message: str,
    assistant_response: str,
    redis_client,
) -> None:
    """Background worker: extract user facts and store in Redis."""
    try:
        _fact_memory.extract_and_store(user_id, user_message, assistant_response, redis_client)
    except Exception as e:
        logger.debug(f"UserFactMemory background extraction failed: {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/history/{session_id}")
async def get_history(session_id: str, current_user: User = Depends(get_current_user)):
    memory = ConversationMemory(redis_client=get_redis(), user_id=current_user.id)
    messages = memory.get_history(session_id, max_messages=20)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@router.delete("/history/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(session_id: str, current_user: User = Depends(get_current_user)):
    memory = ConversationMemory(redis_client=get_redis(), user_id=current_user.id)
    memory.clear_session(session_id)
    return None


# ── Trace endpoint ─────────────────────────────────────────────────

@router.get("/trace/{session_id}")
async def get_trace(session_id: str, current_user: User = Depends(get_current_user)):
    """
    Return the agent tool-call trace for a session.

    Each entry records:
      tool         — tool name called
      agent        — which agent called it
      args         — truncated argument values
      result_preview — first 250 chars of the JSON result
      duration_ms  — wall-clock time for the tool call
      timestamp    — Unix epoch

    Useful for debugging agent behaviour, demos, and observability dashboards.
    Traces expire after 1 hour. Returns empty list if Redis is unavailable or
    the session has expired.
    """
    redis_client = get_redis()
    trace = ToolAwareAgent.load_trace(current_user.id, session_id, redis_client)
    return {
        "session_id": session_id,
        "trace": trace,
        "count": len(trace),
        "redis_available": redis_client is not None,
    }


@router.get("/research-runs")
async def list_research_runs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Return this user's recent durable research runs for recovery after reload."""
    from src.agents.research_run_store import ResearchRunStore
    return {"runs": ResearchRunStore(db, current_user.id).list_recent()}


# ── Streaming endpoint ────────────────────────────────────────────────────────
#
# One turn runs on a worker thread in four steps — route, cache lookup,
# stream, finish — and hands events to the async response through a queue.

class _TurnRejected(Exception):
    """A turn that cannot start; its message is shown to the user."""


@dataclass
class _Route:
    question: str
    agent: Any
    intent: str
    method: str
    confidence: float


@dataclass
class _CacheLookup:
    cache: Any = None
    scope: Optional[dict] = None
    answer: Optional[dict] = None


@dataclass
class _Outcome:
    reply: str = ""
    done: bool = False
    tools_called: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    eval_scores: Optional[dict] = None


def _route_turn(request: "ChatRequest", db, user_id: int, embedding_manager) -> _Route:
    """Pick the agent: a resumed research run goes back to DeepResearch, anything else is routed."""
    if not request.resume_run_id:
        agent, intent, method, confidence = _get_agent_router(embedding_manager).route_stream(request.message)
        return _Route(request.message, agent, intent, method, confidence)
    from src.agents.deep_research_agent import DeepResearchAgent
    from src.agents.research_run_store import ResearchRunStore
    saved_run = ResearchRunStore(db, user_id).load(request.resume_run_id)
    if saved_run is None:
        raise _TurnRejected("Research run not found for this user")
    return _Route(saved_run["question"], DeepResearchAgent(), "research_qa", "resume", 1.0)


def _lookup_cached_answer(route: _Route, user_id: int, raw_history: list, facts: str,
                          redis_client, embedding_manager, resuming: bool) -> _CacheLookup:
    """A stored answer to an equivalent question, for turns that do not depend on this conversation."""
    if not (settings.ANSWER_CACHE_ENABLED and redis_client and not resuming):
        return _CacheLookup()
    from src.rag.answer_cache import SemanticAnswerCache, eligible_answer
    if not eligible_answer(route.intent, raw_history, facts, route.agent.name):
        return _CacheLookup()
    try:
        cache = SemanticAnswerCache(redis_client, embedding_manager.generate_embedding)
        scope = cache.scope(user_id, settings.LLM_MODEL, route.agent.system_prompt,
                            embedding_manager.collection.count())
        return _CacheLookup(cache, scope, cache.get(route.question, scope))
    except Exception as exc:
        logger.debug("Answer cache lookup failed: %s", exc)
        return _CacheLookup()


def _replay(cached: dict):
    yield {"type": "token", "value": cached["reply"]}
    yield {"type": "done", "tools_called": [], "citations": cached["citations"], "cache_hit": True}


def _relay(events, emit: Callable[[dict], None], cancel_event: threading.Event) -> _Outcome:
    """Forward agent events to the client while assembling the final reply."""
    outcome, parts = _Outcome(), []
    with closing(events) as stream:
        for event in stream:
            if cancel_event.is_set() or event.get("type") == "cancelled":
                break
            etype = event.get("type")
            if etype in ("token", "draft_token"):
                parts.append(event["value"])
            elif etype == "replace":
                parts = [event["value"]]
            elif etype == "done":
                outcome.done = True
                outcome.tools_called = event.get("tools_called", [])
                outcome.citations = event.get("citations", [])
            elif etype == "critique_result":
                # Only ReflectionAgent emits this — real CriticAgent scores
                # from the reflection loop, not the separate opt-in LLMJudge.
                outcome.eval_scores = {k: v for k, v in event.items() if k != "type"}
            emit(event)
    outcome.reply = "".join(parts)
    return outcome


def _finish_turn(route: _Route, outcome: _Outcome, lookup: _CacheLookup, memory: ConversationMemory,
                 session_id: str, raw_history: list, user_id: int, redis_client, resuming: bool) -> None:
    """Keep a completed answer: answer cache, conversation memory, then background fact extraction."""
    if lookup.cache and lookup.scope and not lookup.answer:
        from src.rag.answer_cache import cacheable_result
        if cacheable_result(outcome.reply, outcome.tools_called, outcome.citations):
            lookup.cache.put(route.question, lookup.scope,
                             {"reply": outcome.reply, "citations": outcome.citations})
    # A resumed run's question is already in the history from its first attempt.
    if not resuming or not any(item.get("role") == "user" and item.get("content") == route.question
                               for item in raw_history):
        memory.add_message(session_id, "user", route.question)
    memory.add_message(session_id, "assistant", outcome.reply)
    threading.Thread(
        target=_extract_facts_bg,
        args=(user_id, route.question, outcome.reply, redis_client),
        daemon=True,
    ).start()


def _run_turn(request: "ChatRequest", user: User, session_id: str, embedding_manager,
              emit: Callable[[dict], None], cancel_event: threading.Event) -> None:
    from src.database.models import SessionLocal

    db = SessionLocal()
    try:
        # One token budget for everything this turn asks of the model; background fact
        # extraction runs on its own thread afterwards and is only counted.
        with request_budget("chat_turn", settings.CHAT_TURN_TOKEN_BUDGET):
            _run_budgeted_turn(request, user, session_id, embedding_manager, emit, cancel_event,
                               db, get_redis(), bool(request.resume_run_id))
    finally:
        db.close()


def _run_budgeted_turn(request: "ChatRequest", user: User, session_id: str, embedding_manager,
                       emit: Callable[[dict], None], cancel_event: threading.Event,
                       db, redis_client, resuming: bool) -> None:
    from src.rag.retriever import Retriever

    try:
        facts = _fact_memory.get_context_string(user.id, redis_client)
        start = time.time()
        route = _route_turn(request, db, user.id, embedding_manager)
        agent = ToolAwareAgent(route.agent, user_id=user.id, session_id=session_id, redis_client=redis_client)
        emit({"type": "intent", "value": route.intent, "method": route.method,
              "confidence": round(route.confidence, 3)})
        emit({"type": "agent", "value": agent.name})

        memory = ConversationMemory(redis_client=redis_client, user_id=user.id)
        raw_history = memory.get_history(session_id, max_messages=100)
        history = ContextBuilder(embedding_model=getattr(embedding_manager, "model", None)).build(
            route.question, raw_history, top_k=20)
        context = {
            "db": db,
            "user": user,
            "session_id": session_id,
            "redis_client": redis_client,
            "_cancel_event": cancel_event,
            "retriever": Retriever(embedding_manager),
            "embedding_manager": embedding_manager,
            "user_facts_context": facts,
            "resume_run_id": request.resume_run_id,
            "_full_history": raw_history,
        }

        lookup = _lookup_cached_answer(route, user.id, raw_history, facts, redis_client,
                                       embedding_manager, resuming)
        events = _replay(lookup.answer) if lookup.answer else agent.stream(route.question, history, context)
        outcome = _relay(events, emit, cancel_event)

        if outcome.reply and outcome.done and not cancel_event.is_set():
            _finish_turn(route, outcome, lookup, memory, session_id, raw_history, user.id,
                         redis_client, resuming)
        record_request(
            intent=route.intent,
            agent=agent.name,
            method=route.method,
            latency_ms=int((time.time() - start) * 1000),
            tools_called=outcome.tools_called,
            eval_scores=outcome.eval_scores,
        )
    except _TurnRejected as exc:
        emit({"type": "error", "value": str(exc)})
    except Exception as exc:
        logger.error("Stream error: %s", exc)
        emit({"type": "error", "value": str(exc)})


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    current_user: User = Depends(demo_quota),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Yields a sequence of JSON events:
      {"type": "session",      "session_id": str}
      {"type": "intent",       "value": str, "method": str, "confidence": float}
      {"type": "agent",        "value": str}
      --- standard agent events ---
      {"type": "tool_call",    "tool": str}
      {"type": "tool_result",  "tool": str, "count": int}
      {"type": "generating"}
      {"type": "token",        "value": str}    ← repeats per token
      {"type": "done",         "tools_called": list, "citations": list}
      --- deep research / plan-and-solve events ---
      {"type": "plan",         "tasks": [{id, title, intent}, ...]}
      {"type": "task_started", "id": int, "title": str}
      {"type": "task_done",    "id": int, "citations": int}
    """
    if not request.message.strip() and not request.resume_run_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    session_id = request.session_id or ConversationMemory.new_session_id()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()

    def emit(event: Optional[dict]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work():
        try:
            _run_turn(request, current_user, session_id, embedding_manager, emit, cancel_event)
        finally:
            emit(None)  # end of stream

    threading.Thread(target=work, daemon=True).start()

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            while True:
                if await http_request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            cancel_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Eval endpoint ─────────────────────────────────────────────────────────────

@router.post("/eval/run", response_model=EvalRunResponse)
async def eval_run(
    request: EvalRunRequest,
    current_user: User = Depends(forbid_demo),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
):
    """
    Batch evaluation over a list of test cases.

    Runs the full agent pipeline and scores each reply with LLMJudge.
    Returns per-case results and aggregate averages.
    """
    if not request.test_cases:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide at least one test case")

    from src.evaluation.llm_judge import LLMJudge
    judge = LLMJudge()
    retriever = Retriever(embedding_manager)
    context = {
        "db": db,
        "user": current_user,
        "retriever": retriever,
        "embedding_manager": embedding_manager,
        "_trace": [],
    }
    agent_router = _get_agent_router(embedding_manager)

    results = []
    intent_correct = 0
    has_expected = 0
    totals = {"relevance": 0.0, "accuracy": 0.0, "completeness": 0.0, "usefulness": 0.0}

    for case in request.test_cases:
        result, intent, method, confidence, agent_name = agent_router.route(
            message=case.question,
            conversation_history=[],
            context=context,
        )
        score = judge.evaluate(question=case.question, response=result.reply,
                               citations=result.citations)

        for k in totals:
            totals[k] += getattr(score, k)

        intent_match = None
        if case.expected_intent:
            has_expected += 1
            intent_match = intent == case.expected_intent
            if intent_match:
                intent_correct += 1

        results.append({
            "question": case.question,
            "expected_intent": case.expected_intent,
            "detected_intent": intent,
            "recognition_method": method,
            "intent_correct": intent_match,
            "agent_used": agent_name,
            "tools_called": result.tools_called,
            "reply_preview": result.reply[:300],
            "scores": score.to_dict(),
        })

    n = len(results)
    avg_scores = {k: round(v / n, 2) for k, v in totals.items()} if n else totals
    intent_accuracy = round(intent_correct / has_expected, 3) if has_expected else None

    return EvalRunResponse(
        results=results,
        intent_accuracy=intent_accuracy,
        avg_scores=avg_scores,
        total_cases=n,
    )
