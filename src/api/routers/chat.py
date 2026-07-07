"""
Chat endpoint: unified conversational interface with agent routing.

POST /chat        — send a message, get a routed agent response
GET  /chat/history/{session_id}    — retrieve conversation history
DELETE /chat/history/{session_id}  — clear a session
POST /chat/eval/run                — batch LLM-as-Judge evaluation
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_db_session, get_current_user, get_embedding_manager
from src.database.models import User
from src.models.embeddings import EmbeddingManager
from src.utils.config import settings
from src.agents.router import AgentRouter
from src.agents.memory import ConversationMemory
from src.agents.metrics import record_request
from src.rag.retriever import Retriever

router = APIRouter(prefix="/chat", tags=["Chat Agent"])
logger = logging.getLogger(__name__)

# Module-level singleton — initialized once on first request
_agent_router: Optional[AgentRouter] = None


def _get_agent_router(embedding_manager: EmbeddingManager) -> AgentRouter:
    global _agent_router
    if _agent_router is None:
        # Share the sentence-transformer already loaded by EmbeddingManager
        _agent_router = AgentRouter(embedding_model=getattr(embedding_manager, "model", None))
    return _agent_router


def _get_redis():
    """Return a Redis client if REDIS_URL is set and reachable, else None."""
    if not settings.REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable, using in-memory memory: {e}")
        return None


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    enable_eval: bool = False   # Run LLM-as-Judge on this response


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


_INTENT_TO_AGENT_NAME = {
    "research_qa": "ResearchAgent",
    "recommendation": "RecommendationAgent",
    "document_management": "DocumentAgent",
    "general_chat": "GeneralAgent",
}


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

    The system:
    1. Classifies intent (research_qa / recommendation / document_management / general_chat)
    2. Routes to the matching specialized agent
    3. Agent calls tools (RAG search, feed fetch) as needed via OpenAI function calling
    4. Conversation history is stored in Redis (falls back to in-memory)

    Pass `session_id` from a previous response to continue the same conversation.
    Set `enable_eval: true` to also run LLM-as-Judge scoring on the reply.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    session_id = request.session_id or ConversationMemory.new_session_id()
    memory = ConversationMemory(redis_client=_get_redis(), user_id=current_user.id)
    retriever = Retriever(embedding_manager)

    context = {
        "db": db,
        "user": current_user,
        "retriever": retriever,
        "embedding_manager": embedding_manager,
    }

    history = memory.get_history(session_id, max_messages=8)
    agent_router = _get_agent_router(embedding_manager)

    result, intent, method, confidence = agent_router.route(
        message=request.message,
        conversation_history=history,
        context=context,
    )

    memory.add_message(session_id, "user", request.message)
    memory.add_message(session_id, "assistant", result.reply)

    eval_scores = None
    if request.enable_eval:
        try:
            from src.evaluation.llm_judge import LLMJudge
            score = LLMJudge().evaluate(
                question=request.message,
                response=result.reply,
            )
            eval_scores = score.to_dict()
        except Exception as e:
            logger.warning(f"LLM judge skipped: {e}")

    agent_used = _INTENT_TO_AGENT_NAME.get(intent, "GeneralAgent")

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
async def get_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Return the conversation history for a session."""
    memory = ConversationMemory(redis_client=_get_redis(), user_id=current_user.id)
    messages = memory.get_history(session_id, max_messages=20)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@router.delete("/history/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Clear conversation history for a session."""
    memory = ConversationMemory(redis_client=_get_redis(), user_id=current_user.id)
    memory.clear_session(session_id)
    return None


@router.post("/eval/run", response_model=EvalRunResponse)
async def eval_run(
    request: EvalRunRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
):
    """
    Batch LLM-as-Judge evaluation over a list of test cases.

    For each case:
    - Runs the full agent pipeline (intent → agent → tools → reply)
    - Scores the reply on relevance / accuracy / completeness / usefulness (0-10 each)
    - If `expected_intent` is provided, also tracks intent classification accuracy

    Returns per-case results and aggregate averages.
    """
    if not request.test_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one test case",
        )

    from src.evaluation.llm_judge import LLMJudge

    judge = LLMJudge()
    retriever = Retriever(embedding_manager)
    context = {
        "db": db,
        "user": current_user,
        "retriever": retriever,
        "embedding_manager": embedding_manager,
    }
    agent_router = _get_agent_router(embedding_manager)

    results = []
    intent_correct = 0
    has_expected = 0
    totals = {"relevance": 0.0, "accuracy": 0.0, "completeness": 0.0, "usefulness": 0.0}

    for case in request.test_cases:
        result, intent, method, confidence = agent_router.route(
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
            "agent_used": _INTENT_TO_AGENT_NAME.get(intent, "GeneralAgent"),
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
