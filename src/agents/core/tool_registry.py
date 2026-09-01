"""
ToolRegistry — central registry for OpenAI tool schemas and their executors.

Inspired by Hello-Agents ch7 "everything is a tool" philosophy.
Agents call registry.schemas() to get their tool list and registry.execute()
to dispatch calls, rather than scattering schema dicts and if/elif chains.
"""
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


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
        """Execute a tool by name. Returns an error dict on unknown tool."""
        fn = self._executors.get(name)
        if fn is None:
            logger.warning(f"ToolRegistry: unknown tool '{name}'")
            return {"error": f"Unknown tool: {name}"}
        return fn(args, context)

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
