"""
Base agent with an OpenAI function-calling (tool-use) loop.

Subclasses set: name, system_prompt, tool_schemas

Added capabilities:
  - tool_call_listener: optional callback fired after each tool execution
    (used by ToolAwareAgent for observability)
  - context["user_facts_context"]: if present, prepended to system prompt
    (injected by chat.py from UserFactMemory for personalisation)
  - context["_trace"]: if present as a list, tool call records are appended
    (used by the chat endpoint to store session traces in Redis)
"""
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import openai

from src.utils.config import settings
from src.agents.tools import execute_tool

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5


@dataclass
class AgentResult:
    reply: str
    tools_called: List[str] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    processing_time_ms: int = 0


class BaseAgent:
    """
    Runs a tool-calling loop:
      1. Build messages (system + history + user turn)
      2. Call OpenAI; if finish_reason == "tool_calls", execute tools and loop
      3. Return when finish_reason == "stop" or max iterations reached

    Observability hook:
      Set agent.tool_call_listener = fn(name, args, result, duration_ms)
      to receive a callback after every tool execution. ToolAwareAgent uses this.
    """

    name: str = "BaseAgent"
    system_prompt: str = "You are a helpful AI assistant."
    tool_schemas: List[Dict] = []

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.tool_call_listener: Optional[Callable] = None

    def _build_system(self, context: Dict[str, Any]) -> str:
        """Prepend UserFactMemory personalisation context if available."""
        base = self.system_prompt
        facts = context.get("user_facts_context", "")
        return base + facts if facts else base

    def _fire_listener(
        self,
        name: str,
        args: Dict,
        result: Dict,
        duration_ms: int,
        context: Dict[str, Any],
    ) -> None:
        """Notify tool_call_listener and append to context trace list."""
        record = {
            "tool": name,
            "args": {k: str(v)[:120] for k, v in args.items()},
            "result_preview": json.dumps(result)[:250],
            "duration_ms": duration_ms,
            "agent": self.name,
            "timestamp": time.time(),
        }
        # Context-level trace list (used by chat.py for Redis storage)
        if isinstance(context.get("_trace"), list):
            context["_trace"].append(record)
        # Agent-level listener (used by ToolAwareAgent)
        if self.tool_call_listener:
            try:
                self.tool_call_listener(name, args, result, duration_ms)
            except Exception as e:
                logger.debug(f"tool_call_listener raised: {e}")

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        start = time.time()
        messages: List[Dict] = [{"role": "system", "content": self._build_system(context)}]
        messages.extend(conversation_history[-8:])
        messages.append({"role": "user", "content": message})

        tools_called: List[str] = []
        citations: List[Dict] = []

        call_kwargs: Dict[str, Any] = {
            "model": settings.LLM_MODEL,
            "messages": messages,
        }
        if self.tool_schemas:
            call_kwargs["tools"] = self.tool_schemas
            call_kwargs["tool_choice"] = "auto"

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(**call_kwargs)
            choice = response.choices[0]

            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                return AgentResult(
                    reply=choice.message.content or "",
                    tools_called=tools_called,
                    citations=citations,
                    processing_time_ms=int((time.time() - start) * 1000),
                )

            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                logger.info(f"[{self.name}] tool call: {fn_name}({fn_args})")
                tools_called.append(fn_name)

                t0 = time.time()
                result = execute_tool(fn_name, fn_args, context)
                self._fire_listener(fn_name, fn_args, result, int((time.time() - t0) * 1000), context)

                if fn_name in ("search_knowledge_base", "search_user_documents"):
                    for item in result.get("results", []):
                        if item.get("title") or item.get("url"):
                            citations.append({
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "type": item.get("type", "unknown"),
                            })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            call_kwargs["messages"] = messages

        return AgentResult(
            reply="I had trouble completing this request. Please try again.",
            tools_called=tools_called,
            citations=citations,
            processing_time_ms=int((time.time() - start) * 1000),
        )

    def stream(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ):
        """
        Synchronous generator that yields agent events for SSE streaming.

        Event types:
          {"type": "tool_call",   "tool": str}
          {"type": "tool_result", "tool": str, "count": int}
          {"type": "generating"}
          {"type": "token",       "value": str}
          {"type": "done",        "tools_called": list, "citations": list}
          {"type": "error",       "value": str}
        """
        messages: List[Dict] = [{"role": "system", "content": self._build_system(context)}]
        messages.extend(conversation_history[-8:])
        messages.append({"role": "user", "content": message})

        tools_called: List[str] = []
        citations: List[Dict] = []

        call_kwargs: Dict[str, Any] = {"model": settings.LLM_MODEL, "messages": messages}
        if self.tool_schemas:
            call_kwargs["tools"] = self.tool_schemas
            call_kwargs["tool_choice"] = "auto"

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(**call_kwargs)
            choice = response.choices[0]

            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                yield {"type": "generating"}
                # The non-streaming call already has the full reply — yield it directly
                # rather than making a redundant second streaming API call.
                content = choice.message.content or ""
                if content:
                    yield {"type": "token", "value": content}
                yield {"type": "done", "tools_called": tools_called, "citations": citations}
                return

            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                yield {"type": "tool_call", "tool": fn_name}
                tools_called.append(fn_name)

                t0 = time.time()
                result = execute_tool(fn_name, fn_args, context)
                self._fire_listener(fn_name, fn_args, result, int((time.time() - t0) * 1000), context)

                if fn_name in ("search_knowledge_base", "search_user_documents"):
                    for item in result.get("results", []):
                        if item.get("title") or item.get("url"):
                            citations.append({
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "type": item.get("type", "unknown"),
                            })

                count = result.get("count", 0)
                yield {"type": "tool_result", "tool": fn_name, "count": count}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            call_kwargs["messages"] = messages

        yield {"type": "error", "value": "Max iterations reached"}
