"""
Base agent with an OpenAI function-calling (tool-use) loop.

Subclasses set: name, system_prompt, tool_schemas
"""
import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

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
    """

    name: str = "BaseAgent"
    system_prompt: str = "You are a helpful AI assistant."
    tool_schemas: List[Dict] = []

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def run(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> AgentResult:
        """
        Run the agent on a single user message.

        Args:
            message: The current user message.
            conversation_history: Prior turns in OpenAI message format.
            context: Runtime objects — db, user, retriever, embedding_manager.
        """
        start = time.time()

        messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
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

            # No more tool calls — return final answer
            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                return AgentResult(
                    reply=choice.message.content or "",
                    tools_called=tools_called,
                    citations=citations,
                    processing_time_ms=int((time.time() - start) * 1000),
                )

            # Execute each requested tool call
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                logger.info(f"[{self.name}] tool call: {fn_name}({fn_args})")
                tools_called.append(fn_name)

                result = execute_tool(fn_name, fn_args, context)

                # Collect citations from search results
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

        Tool-calling iterations use non-streaming OpenAI calls (tool JSON must
        be complete). The final answer uses stream=True for true token-by-token
        output.
        """
        messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
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
                # Stream the final answer token by token
                yield {"type": "generating"}
                stream_resp = self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    stream=True,
                )
                for chunk in stream_resp:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield {"type": "token", "value": token}

                yield {"type": "done", "tools_called": tools_called, "citations": citations}
                return

            # Execute each tool call and yield progress events
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                yield {"type": "tool_call", "tool": fn_name}
                tools_called.append(fn_name)

                result = execute_tool(fn_name, fn_args, context)

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
