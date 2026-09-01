"""
Chat endpoint: unified conversational interface with agent routing.

POST   /chat            — send a message, get a routed agent response
GET    /chat/history/{session_id}    — retrieve conversation history
DELETE /chat/history/{session_id}    — clear a session
GET    /chat/trace/{session_id}      — retrieve agent tool-call trace (Feature 4)
POST   /chat/eval/run                — batch LLM-as-Judge evaluation

New in this version:
  - UserFactMemory auto-injection: user facts extracted after each turn (BG thread)
    and prepended to agent system prompts on the next request (Feature 1)
  - ContextBuilder: semantically relevant history selection instead of last-N (Feature 2)
  - Tool-call tracing: all tool calls recorded in context["_trace"] and stored in Redis (Feature 4)
  - ToolAwareAgent wraps the streaming agent for richer per-call observability
"""
import asyncio
import json
import logging
import threading
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

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

router = APIRouter(prefix="/chat", tags=["Chat Agent"])
logger = logging.getLogger(__name__)

_agent_router: Optional[AgentRouter] = None
_fact_memory = UserFactMemory()


def _get_agent_router(embedding_manager: EmbeddingManager) -> AgentRouter:
    global _agent_router
    if _agent_router is None:
        _agent_router = AgentRouter(embedding_model=getattr(embedding_manager, "model", None))
    return _agent_router


def _get_redis():
    if not settings.REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    enable_eval: bool = False


class ChatResponse(BaseModel):
    reply: str
    intent: str
    agent_used: str
    citations: List[dict]
    tools_called: List[str]
    session_id: str
    recognition_method: str
    confidence: float
    processing_time_ms: int
    eval_scores: Optional[dict] = None


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

def _store_trace(session_id: str, trace: List[dict], redis_client) -> None:
    """Persist the tool-call trace for a session to Redis."""
    if not trace or not redis_client:
        return
    try:
        redis_client.setex(f"agent_trace:{session_id}", 3600, json.dumps(trace))
    except Exception as e:
        logger.debug(f"Trace persist failed: {e}")


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

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
):
    """
    Main conversational endpoint with automatic agent routing.

    Pipeline per request:
      1. Load relevant conversation history (ContextBuilder: semantic selection)
      2. Inject long-term user facts into system prompt (UserFactMemory)
      3. Classify intent → dispatch to agent (router)
      4. Record tool-call trace in Redis
      5. Background: extract and store new user facts (UserFactMemory)

    Pass `session_id` from a previous response to continue the same conversation.
    """
    if not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    session_id = request.session_id or ConversationMemory.new_session_id()
    redis_client = _get_redis()
    memory = ConversationMemory(redis_client=redis_client, user_id=current_user.id)
    retriever = Retriever(embedding_manager)
    agent_router = _get_agent_router(embedding_manager)

    # ── Feature 2: Semantic history selection ─────────────────────────────────
    raw_history = memory.get_history(session_id, max_messages=20)
    builder = ContextBuilder(embedding_model=getattr(embedding_manager, "model", None))
    history = builder.build(request.message, raw_history, top_k=8)

    # ── Feature 1: Inject user facts into system prompt ───────────────────────
    user_facts_context = _fact_memory.get_context_string(current_user.id, redis_client)

    # ── Feature 4: Initialise trace list ──────────────────────────────────────
    trace: List[dict] = []

    context = {
        "db": db,
        "user": current_user,
        "retriever": retriever,
        "embedding_manager": embedding_manager,
        "user_facts_context": user_facts_context,   # Feature 1
        "_trace": trace,                             # Feature 4
    }

    result, intent, method, confidence, agent_used = agent_router.route(
        message=request.message,
        conversation_history=history,
        context=context,
    )

    # ── Feature 4: Persist trace ───────────────────────────────────────────────
    _store_trace(session_id, trace, redis_client)

    memory.add_message(session_id, "user", request.message)
    memory.add_message(session_id, "assistant", result.reply)

    # ── Feature 1: Background fact extraction ─────────────────────────────────
    threading.Thread(
        target=_extract_facts_bg,
        args=(current_user.id, request.message, result.reply, redis_client),
        daemon=True,
    ).start()

    eval_scores = None
    if request.enable_eval:
        try:
            from src.evaluation.llm_judge import LLMJudge
            score = LLMJudge().evaluate(question=request.message, response=result.reply)
            eval_scores = score.to_dict()
        except Exception as e:
            logger.warning(f"LLM judge skipped: {e}")

    record_request(
        intent=intent,
        agent=agent_used,
        method=method,
        latency_ms=result.processing_time_ms,
        tools_called=result.tools_called,
        eval_scores=eval_scores,
    )

    return ChatResponse(
        reply=result.reply,
        intent=intent,
        agent_used=agent_used,
        citations=result.citations,
        tools_called=result.tools_called,
        session_id=session_id,
        recognition_method=method,
        confidence=round(confidence, 3),
        processing_time_ms=result.processing_time_ms,
        eval_scores=eval_scores,
    )


@router.get("/history/{session_id}")
async def get_history(session_id: str, current_user: User = Depends(get_current_user)):
    memory = ConversationMemory(redis_client=_get_redis(), user_id=current_user.id)
    messages = memory.get_history(session_id, max_messages=20)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@router.delete("/history/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(session_id: str, current_user: User = Depends(get_current_user)):
    memory = ConversationMemory(redis_client=_get_redis(), user_id=current_user.id)
    memory.clear_session(session_id)
    return None


# ── Feature 4: Trace endpoint ─────────────────────────────────────────────────

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
    redis_client = _get_redis()
    trace = ToolAwareAgent.load_trace(session_id, redis_client)
    return {
        "session_id": session_id,
        "trace": trace,
        "count": len(trace),
        "redis_available": redis_client is not None,
    }


# ── Streaming endpoint ────────────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
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
    if not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    session_id = request.session_id or ConversationMemory.new_session_id()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_in_thread():
        from src.database.models import SessionLocal
        from src.rag.retriever import Retriever

        db = SessionLocal()
        redis_client = _get_redis()
        try:
            retriever = Retriever(embedding_manager)

            # ── Feature 1: inject user facts ──────────────────────────────────
            user_facts_context = _fact_memory.get_context_string(current_user.id, redis_client)

            context = {
                "db": db,
                "user": current_user,
                "retriever": retriever,
                "embedding_manager": embedding_manager,
                "user_facts_context": user_facts_context,
            }

            agent_router = _get_agent_router(embedding_manager)
            raw_agent, intent, method, confidence = agent_router.route_stream(request.message)

            # ── Feature 4: wrap agent for observability ────────────────────────
            aware_agent = ToolAwareAgent(raw_agent, session_id=session_id, redis_client=redis_client)

            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "intent",
                "value": intent,
                "method": method,
                "confidence": round(confidence, 3),
            })
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": "agent",
                "value": aware_agent.name,
            })

            # ── Feature 2: semantic history ────────────────────────────────────
            memory = ConversationMemory(redis_client=redis_client, user_id=current_user.id)
            raw_history = memory.get_history(session_id, max_messages=20)
            builder = ContextBuilder(embedding_model=getattr(embedding_manager, "model", None))
            history = builder.build(request.message, raw_history, top_k=8)

            reply_parts = []
            for event in aware_agent.stream(request.message, history, context):
                if event.get("type") == "token":
                    reply_parts.append(event["value"])
                loop.call_soon_threadsafe(queue.put_nowait, event)

            full_reply = "".join(reply_parts)
            if full_reply:
                memory.add_message(session_id, "user", request.message)
                memory.add_message(session_id, "assistant", full_reply)

                # ── Feature 1: background fact extraction ─────────────────────
                threading.Thread(
                    target=_extract_facts_bg,
                    args=(current_user.id, request.message, full_reply, redis_client),
                    daemon=True,
                ).start()

        except Exception as e:
            logger.error(f"Stream error: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "value": str(e)})
        finally:
            db.close()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    async def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Eval endpoint ─────────────────────────────────────────────────────────────

@router.post("/eval/run", response_model=EvalRunResponse)
async def eval_run(
    request: EvalRunRequest,
    current_user: User = Depends(get_current_user),
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
        score = judge.evaluate(question=case.question, response=result.reply)

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
