"""
ToolRegistry — central registry for OpenAI tool schemas and their executors.

Inspired by Hello-Agents ch7 "everything is a tool" philosophy.
Agents call registry.schemas() to get their tool list and registry.execute()
to dispatch calls, rather than scattering schema dicts and if/elif chains.
"""
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

# Every executor already wraps its own body in try/except and returns an
# {"error": ...} dict rather than raising — that handles the failure modes a
# tool anticipates (missing API key, malformed args). It does nothing for a
# call that simply hangs (a slow ChromaDB query, a stalled outbound HTTP
# request with no timeout of its own): a bare try/except never gets a chance
# to run because the call never returns or raises. TOOL_TIMEOUT_SECONDS
# bounds that from outside the call, and TOOL_MAX_RETRIES absorbs genuinely
# transient failures (a dropped connection, a momentary rate limit) instead
# of failing the whole agent turn on the first hiccup.
TOOL_TIMEOUT_SECONDS = 15
TOOL_MAX_RETRIES = 1
TOOL_RETRY_BACKOFF_BASE_SECONDS = 0.5


class ToolRegistry:
    """
    Central store for (schema, executor) pairs.

    Usage:
        registry = ToolRegistry()
        registry.register(SEARCH_KB_SCHEMA, _exec_search_kb)
        registry.register(SEARCH_WEB_SCHEMA, _exec_search_web)

        # Pass to OpenAI
        tools=registry.schemas()

        # Dispatch a tool_call
        result = registry.execute(tool_name, args, context)

        # Get a subset for a specific agent
        sub = registry.subset("search_knowledge_base", "search_web")
    """

    # Shared across every ToolRegistry instance (including subset() copies) —
    # a thread pool per instance would multiply worker threads for no benefit,
    # since execute() is the one dispatch path every agent already funnels
    # through. Lives for the process lifetime, same as EmbeddingManager's
    # model/client singletons; nothing here needs explicit shutdown.
    _executor_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-exec")

    def __init__(self):
        self._schemas: Dict[str, Dict] = {}
        self._executors: Dict[str, Callable] = {}

    def register(self, schema: Dict, executor: Callable) -> "ToolRegistry":
        """Register a tool. Returns self for chaining."""
        name = schema["function"]["name"]
        self._schemas[name] = schema
        self._executors[name] = executor
        logger.debug(f"ToolRegistry: registered tool '{name}'")
        return self

    def schemas(self) -> List[Dict]:
        """Return all registered OpenAI tool schemas."""
        return list(self._schemas.values())

    def execute(self, name: str, args: Dict[str, Any], context: Dict) -> Dict:
        """
        Execute a tool by name, with a hard timeout and bounded retries.

        Runs the executor in a worker thread so a hang can't block the agent
        turn forever — future.result(timeout=...) gives up waiting even
        though Python has no way to forcibly kill the underlying thread, so
        an abandoned call may keep running in the background until it
        eventually finishes or errors on its own; the point is the agent
        stops waiting on it, not that the call is destroyed.
        """
        fn = self._executors.get(name)
        if fn is None:
            logger.warning(f"ToolRegistry: unknown tool '{name}'")
            return {"error": f"Unknown tool: {name}"}

        last_error = "unknown error"
        for attempt in range(TOOL_MAX_RETRIES + 1):
            future: Future = self._executor_pool.submit(fn, args, context)
            try:
                return future.result(timeout=TOOL_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                last_error = f"timed out after {TOOL_TIMEOUT_SECONDS}s"
                logger.warning(f"ToolRegistry: '{name}' {last_error} (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"ToolRegistry: '{name}' raised on attempt {attempt + 1}: {e}")

            if attempt < TOOL_MAX_RETRIES:
                time.sleep(TOOL_RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt))

        return {"error": f"Tool '{name}' failed after {TOOL_MAX_RETRIES + 1} attempt(s): {last_error}"}

    def subset(self, *names: str) -> "ToolRegistry":
        """Return a new registry containing only the named tools."""
        sub = ToolRegistry()
        for name in names:
            if name in self._schemas:
                sub._schemas[name] = self._schemas[name]
                sub._executors[name] = self._executors[name]
            else:
                logger.warning(f"ToolRegistry.subset: tool '{name}' not found")
        return sub

    def names(self) -> List[str]:
        return list(self._schemas.keys())

    def __len__(self) -> int:
        return len(self._schemas)

    def __contains__(self, name: str) -> bool:
        return name in self._schemas
