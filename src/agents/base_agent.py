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
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import openai

from src.utils.config import settings
from src.agents.tools import build_default_registry
from src.agents.context_budget import ContextBudget, OUTPUT_RESERVE
from src.agents.source_safety import SOURCE_POLICY, untrusted_block

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
        # V4: shared typed dispatch registry (stateless, built once per process)
        self.registry = build_default_registry()
        self.context_budget = ContextBudget(settings.LLM_MODEL)

    def _prepare_messages(self, message: str, history: List[Dict], context: Dict[str, Any]) -> List[Dict]:
        return self.context_budget.build(
            system=self.system_prompt + "\n\n" + SOURCE_POLICY,
            facts=context.get("user_facts_context", ""),
            selected_history=history,
            full_history=context.get("_full_history", history),
            user_message=message,
            tool_schemas=self.tool_schemas,
            context=context,
            client=self.client,
            model=settings.LLM_MODEL,
        )

    @staticmethod
    def _cancelled(context: Dict[str, Any]) -> bool:
        event = context.get("_cancel_event")
        return event is not None and event.is_set()

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
        # A request-local callback takes precedence over the legacy agent
        # callback. Agents are cached by the router and can serve concurrent
        # requests, so observability must not mutate an agent-wide attribute.
        listener = context.get("_tool_call_listener") or self.tool_call_listener
        if listener:
            try:
                listener(name, args, result, duration_ms)
            except Exception as e:
                logger.debug(f"tool_call_listener raised: {e}")

    # Tools whose results carry sources worth surfacing to the user.
    _CITING_TOOLS = ("search_knowledge_base", "search_user_documents", "search_web", "fetch_full_paper")

    @staticmethod
    def _partial_reply(citations: List[Dict], tools_called: List[str]) -> str:
        """Expose only observed evidence when the tool-step budget is exhausted."""
        lines = ["I reached the research step limit before completing the answer."]
        if citations:
            lines.append("Evidence found so far:")
            for item in citations[:5]:
                title = item.get("title") or item.get("url") or "Source"
                excerpt = str(item.get("evidence") or "").strip()[:240]
                lines.append(f"- {title}: {excerpt}" if excerpt else f"- {title}")
        elif tools_called:
            lines.append("The tools returned no citable evidence yet.")
        return "\n".join(lines)

    def _dispatch_tool_call(
        self,
        tc,
        context: Dict[str, Any],
        tools_called: List[str],
        citations: List[Dict],
        messages: List[Dict],
    ) -> Dict:
        """
        Execute one tool call and record its effects.

        Shared by run() and stream() so the two paths cannot drift apart:
        appends the tool name, fires the observability listener, harvests
        citations, and appends the tool-result message. Returns the raw result
        so the caller can emit whatever progress events it needs.
        """
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[{self.name}] malformed tool arguments for {fn_name}: {e}")
            fn_args = {}

        tools_called.append(fn_name)

        t0 = time.time()
        allowed = {schema.get("function", {}).get("name") for schema in self.tool_schemas}
        result = (
            self.registry.execute(fn_name, fn_args, context)
            if fn_name in allowed else {"error": f"Tool {fn_name} is not allowed for this agent"}
        )
        self._fire_listener(fn_name, fn_args, result, int((time.time() - t0) * 1000), context)

        if fn_name in self._CITING_TOOLS:
            for item in result.get("results", []):
                title = item.get("title") or result.get("title", "")
                url = item.get("url") or result.get("url", "")
                if fn_name == "fetch_full_paper" and not url and result.get("arxiv_id"):
                    url = f"https://arxiv.org/abs/{result['arxiv_id']}"
                if title or url:
                    citations.append({
                        "title": title,
                        "url": url,
                        "type": item.get("type", fn_name),
                        "evidence": item.get("content", "")[:800],
                    })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": untrusted_block(
                self.context_budget.fit_tool_result(result, messages, self.tool_schemas), fn_name
            ),
        })
        return result

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        start = time.time()
        messages: List[Dict] = self._prepare_messages(message, conversation_history, context)

        tools_called: List[str] = []
        citations: List[Dict] = []

        call_kwargs: Dict[str, Any] = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "max_tokens": OUTPUT_RESERVE,
        }
        if self.tool_schemas:
            call_kwargs["tools"] = self.tool_schemas
            call_kwargs["tool_choice"] = "auto"

        for _ in range(MAX_TOOL_ITERATIONS):
            if self._cancelled(context):
                return AgentResult(reply="", tools_called=tools_called, citations=citations)
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
                if self._cancelled(context):
                    return AgentResult(reply="", tools_called=tools_called, citations=citations)
                logger.info(f"[{self.name}] tool call: {tc.function.name}")
                self._dispatch_tool_call(tc, context, tools_called, citations, messages)

            if self.context_budget.remaining(messages, self.tool_schemas) < 128:
                return AgentResult(
                    reply=self._partial_reply(citations, tools_called),
                    tools_called=tools_called, citations=citations,
                    processing_time_ms=int((time.time() - start) * 1000),
                )

            call_kwargs["messages"] = messages

        return AgentResult(
            reply=self._partial_reply(citations, tools_called),
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
        messages: List[Dict] = self._prepare_messages(message, conversation_history, context)

        tools_called: List[str] = []
        citations: List[Dict] = []

        call_kwargs: Dict[str, Any] = {"model": settings.LLM_MODEL, "messages": messages, "max_tokens": OUTPUT_RESERVE}
        if self.tool_schemas:
            call_kwargs["tools"] = self.tool_schemas
            call_kwargs["tool_choice"] = "auto"

        for _ in range(MAX_TOOL_ITERATIONS):
            if self._cancelled(context):
                yield {"type": "cancelled"}
                return
            yield {"type": "generating"}
            stream = self.client.chat.completions.create(**call_kwargs, stream=True)
            content_parts: List[str] = []
            tool_buffers: Dict[int, Dict] = {}
            try:
                for chunk in stream:
                    if self._cancelled(context):
                        yield {"type": "cancelled"}
                        return
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                        yield {"type": "token", "value": delta.content}
                    for tc in delta.tool_calls or []:
                        buf = tool_buffers.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            buf["id"] = tc.id
                        if tc.function:
                            buf["name"] += tc.function.name or ""
                            buf["arguments"] += tc.function.arguments or ""
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()

            if not tool_buffers:
                yield {"type": "done", "tools_called": tools_called, "citations": citations}
                return

            pending = [tool_buffers[i] for i in sorted(tool_buffers)]
            messages.append({
                "role": "assistant", "content": "".join(content_parts) or None,
                "tool_calls": [
                    {"id": item["id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}}
                    for item in pending
                ],
            })
            for item in pending:
                if self._cancelled(context):
                    yield {"type": "cancelled"}
                    return
                tc = SimpleNamespace(id=item["id"], function=SimpleNamespace(name=item["name"], arguments=item["arguments"]))
                yield {"type": "tool_call", "tool": item["name"]}
                result = self._dispatch_tool_call(tc, context, tools_called, citations, messages)
                yield {"type": "tool_result", "tool": item["name"], "count": result.get("count", 0)}
            if self.context_budget.remaining(messages, self.tool_schemas) < 128:
                yield {"type": "token", "value": self._partial_reply(citations, tools_called)}
                yield {"type": "done", "tools_called": tools_called, "citations": citations,
                       "stop_reason": "context_budget"}
                return
            call_kwargs["messages"] = messages

        yield {"type": "token", "value": self._partial_reply(citations, tools_called)}
        yield {"type": "done", "tools_called": tools_called, "citations": citations,
               "stop_reason": "tool_step_limit"}
