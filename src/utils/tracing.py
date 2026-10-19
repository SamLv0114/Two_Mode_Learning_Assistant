"""Optional LangSmith tracing for every agent.

Off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set. When on:
- every OpenAI call made through `openai_client()` is recorded (prompt, output, tokens, latency);
- functions decorated with `traceable` become nested runs (agent turns, tools, retrieval);
- LangGraph graphs (DeepResearch, Coding Agent) are traced natively, node by node.

LangSmith and LangGraph read their settings from os.environ, while pydantic settings only
load .env into the settings object, so the values are exported here once at import.
Only numeric user IDs are attached as metadata, never emails. Set LANGSMITH_HIDE_INPUTS /
LANGSMITH_HIDE_OUTPUTS to keep prompts and answers out of LangSmith entirely.
"""
import logging
import os
from typing import Any, Callable, Dict, Optional

import openai

from src.utils.config import settings

logger = logging.getLogger(__name__)


def _export_settings() -> None:
    if not (settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY):
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)
    if settings.LANGSMITH_HIDE_INPUTS:
        os.environ.setdefault("LANGSMITH_HIDE_INPUTS", "true")
    if settings.LANGSMITH_HIDE_OUTPUTS:
        os.environ.setdefault("LANGSMITH_HIDE_OUTPUTS", "true")


_export_settings()


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def openai_client(**kwargs: Any) -> Any:
    """
    The OpenAI client every agent uses: traced in LangSmith when tracing is on, and with
    every chat completion budgeted, costed and given a fallback model (see llm_usage).
    """
    from src.utils.llm_usage import AccountedClient

    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY, **kwargs)
    if tracing_enabled():
        try:
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
        except Exception as exc:  # tracing must never take the product down
            logger.warning("LangSmith wrapping unavailable: %s", exc)
    return AccountedClient(client)


def traceable(*args: Any, **kwargs: Any) -> Callable:
    """langsmith.traceable when the SDK is installed (a cheap pass-through while tracing is off)."""
    try:
        from langsmith import traceable as langsmith_traceable
    except ImportError:  # pragma: no cover - langsmith ships with langgraph's dependencies
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function
    return langsmith_traceable(*args, **kwargs)


def annotate(name: Optional[str] = None, tags: Optional[list] = None, **metadata: Any) -> None:
    """Rename the current run or attach metadata (agent, user_id, run_id) to it, if tracing."""
    if not tracing_enabled():
        return
    try:
        from langsmith import get_current_run_tree
        run = get_current_run_tree()
        if run is None:
            return
        if name:
            run.name = name
        if tags:
            run.tags = list(dict.fromkeys((run.tags or []) + tags))
        run.metadata.update({k: v for k, v in metadata.items() if v is not None})
    except Exception as exc:
        logger.debug("Could not annotate LangSmith run: %s", exc)


def graph_config(run_name: str, tags: list, **metadata: Any) -> Dict[str, Any]:
    """LangGraph config fields LangSmith shows for a graph run (harmless when tracing is off)."""
    return {"run_name": run_name, "tags": tags, "metadata": {k: v for k, v in metadata.items() if v is not None}}


def stream_summary(events: list) -> Dict[str, Any]:
    """What a traced streaming agent run shows as its output: the answer text, not every token event."""
    tokens = [e.get("value", "") for e in events if isinstance(e, dict) and e.get("type") == "token"]
    done = next((e for e in reversed(events) if isinstance(e, dict) and e.get("type") == "done"), {})
    return {"answer": "".join(tokens)[:8000], "events": len(events),
            "tools_called": done.get("tools_called"), "citations": len(done.get("citations") or [])}
