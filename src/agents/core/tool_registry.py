"""
ToolRegistry — central registry for OpenAI tool schemas and their executors.

Inspired by Hello-Agents ch7 "everything is a tool" philosophy.
Agents call registry.schemas() to get their tool list and registry.execute()
to dispatch calls, rather than scattering schema dicts and if/elif chains.
"""
import logging
import random
import threading
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
# Randomizes the backoff by up to +30% so several concurrent callers retrying
# the same failing tool don't all wake up and retry at exactly the same
# instant (a thundering herd against a dependency that's already struggling).
TOOL_RETRY_JITTER_FRACTION = 0.3

# Circuit breaker: once a tool has failed this many *calls* in a row (each
# call already exhausted its own timeout+retry above), stop even trying it
# for a cooldown period — return an error immediately instead of making
# every caller pay the full timeout+retry cost again while the dependency is
# still down. After the cooldown, the next call is let through as a trial;
# success closes the circuit, failure reopens it for another cooldown.
CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_COOLDOWN_SECONDS = 30


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

    # Circuit state is keyed by tool name and shared the same way as the
    # executor pool above — the thing being protected (a flaky downstream
    # dependency behind a given tool) is process-wide, not specific to
    # whichever ToolRegistry/subset instance happens to call it.
    _circuit_lock = threading.Lock()
    _circuit_state: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self._schemas: Dict[str, Dict] = {}
        self._executors: Dict[str, Callable] = {}
        self._timeouts: Dict[str, int] = {}

    def register(self, schema: Dict, executor: Callable, timeout: int = TOOL_TIMEOUT_SECONDS) -> "ToolRegistry":
        """
        Register a tool. Returns self for chaining.

        timeout overrides TOOL_TIMEOUT_SECONDS for tools whose normal-case
        latency is legitimately higher than a hang-detection threshold should
        be for everything else — e.g. fetch_full_paper's first call per paper
        downloads and parses a PDF, not just a local vector/DB lookup.
        """
        name = schema["function"]["name"]
        self._schemas[name] = schema
        self._executors[name] = executor
        self._timeouts[name] = timeout
        logger.debug(f"ToolRegistry: registered tool '{name}' (timeout={timeout}s)")
        return self

    def schemas(self) -> List[Dict]:
        """Return all registered OpenAI tool schemas."""
        return list(self._schemas.values())

    def _circuit_is_open(self, name: str) -> bool:
        """True if `name` is currently short-circuited (too many recent failures)."""
        with self._circuit_lock:
            state = self._circuit_state.get(name)
            if not state or state["opened_at"] is None:
                return False
            if time.time() - state["opened_at"] >= CIRCUIT_COOLDOWN_SECONDS:
                # Cooldown elapsed — half-open: let the next call through as a
                # trial rather than staying open forever with no way to heal.
                state["opened_at"] = None
                return False
            return True

    def _circuit_record(self, name: str, success: bool) -> None:
        """Update circuit state after one execute() call (post retries)."""
        with self._circuit_lock:
            state = self._circuit_state.setdefault(name, {"failures": 0, "opened_at": None})
            if success:
                state["failures"] = 0
                state["opened_at"] = None
            else:
                state["failures"] += 1
                if state["failures"] >= CIRCUIT_FAILURE_THRESHOLD and state["opened_at"] is None:
                    state["opened_at"] = time.time()
                    logger.warning(
                        f"ToolRegistry: circuit OPEN for '{name}' after "
                        f"{state['failures']} consecutive failures"
                    )

    def execute(self, name: str, args: Dict[str, Any], context: Dict) -> Dict:
        """
        Execute a tool by name, with a hard timeout, bounded jittered retries,
        and a circuit breaker across calls.

        Runs the executor in a worker thread so a hang can't block the agent
        turn forever — future.result(timeout=...) gives up waiting even
        though Python has no way to forcibly kill the underlying thread, so
        an abandoned call may keep running in the background until it
        eventually finishes or errors on its own; the point is the agent
        stops waiting on it, not that the call is destroyed.

        The circuit breaker guards across separate execute() calls, not the
        retries within one: if a tool's last CIRCUIT_FAILURE_THRESHOLD calls
        all failed (each already having exhausted its own timeout+retry),
        further calls fail immediately instead of each paying the full
        timeout+retry cost again while the dependency is still down.
        """
        fn = self._executors.get(name)
        if fn is None:
            logger.warning(f"ToolRegistry: unknown tool '{name}'")
            return {"error": f"Unknown tool: {name}"}

        if self._circuit_is_open(name):
            logger.warning(f"ToolRegistry: circuit open for '{name}', short-circuiting")
            return {"error": f"Tool '{name}' is temporarily unavailable (too many recent failures) — try again shortly"}

        timeout = self._timeouts.get(name, TOOL_TIMEOUT_SECONDS)
        last_error = "unknown error"
        for attempt in range(TOOL_MAX_RETRIES + 1):
            future: Future = self._executor_pool.submit(fn, args, context)
            try:
                result = future.result(timeout=timeout)
                self._circuit_record(name, success=True)
                return result
            except FutureTimeoutError:
                last_error = f"timed out after {timeout}s"
                logger.warning(f"ToolRegistry: '{name}' {last_error} (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"ToolRegistry: '{name}' raised on attempt {attempt + 1}: {e}")

            if attempt < TOOL_MAX_RETRIES:
                delay = TOOL_RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt)
                delay *= 1 + random.random() * TOOL_RETRY_JITTER_FRACTION
                time.sleep(delay)

        self._circuit_record(name, success=False)
        return {"error": f"Tool '{name}' failed after {TOOL_MAX_RETRIES + 1} attempt(s): {last_error}"}

    def subset(self, *names: str) -> "ToolRegistry":
        """Return a new registry containing only the named tools."""
        sub = ToolRegistry()
        for name in names:
            if name in self._schemas:
                sub._schemas[name] = self._schemas[name]
                sub._executors[name] = self._executors[name]
                sub._timeouts[name] = self._timeouts[name]
            else:
                logger.warning(f"ToolRegistry.subset: tool '{name}' not found")
        return sub

    def names(self) -> List[str]:
        return list(self._schemas.keys())

    def __len__(self) -> int:
        return len(self._schemas)

    def __contains__(self, name: str) -> bool:
        return name in self._schemas
