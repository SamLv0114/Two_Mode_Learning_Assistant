"""Small authenticated MCP surface over the existing ToolRegistry.

Run locally over stdio with LEARNING_ASSISTANT_ACCESS_TOKEN set. The bearer is
validated on each call, so expiry and account deactivation take effect without
restarting the process. The MCP transport adds discovery and interoperability;
tool execution remains in the application's existing registry.
"""
import os
import re
from typing import Callable, Dict

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.agents.tools import build_default_registry
from src.api.security import decode_token
from src.database.models import SessionLocal, User


def authenticated_context(token: str) -> Dict:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise PermissionError("A valid access token is required")
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PermissionError("Invalid access token subject") from exc
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
        if user is None:
            raise PermissionError("User account is unavailable")
        from src.api.deps import get_embedding_manager
        from src.rag.retriever import Retriever
        manager = get_embedding_manager()
        return {"user": user, "db": db,
                "retriever": Retriever(manager), "embedding_manager": manager}
    except Exception:
        db.close()
        raise


def create_mcp_server(token: str, context_factory: Callable[[str], Dict] = authenticated_context,
                      registry=None) -> MCPServer:
    if not token:
        raise PermissionError("LEARNING_ASSISTANT_ACCESS_TOKEN is required")
    registry = registry or build_default_registry()
    server = MCPServer("Learning Assistant Research Tools")

    @server.tool()
    def search_knowledge_base(query: str, n_results: int = 5) -> Dict:
        """Search papers and articles visible to the authenticated user."""
        if not query.strip() or len(query) > 500:
            raise ToolError("query must contain 1–500 characters")
        context = context_factory(token)
        try:
            result = registry.execute("search_knowledge_base", {
                "query": query, "n_results": max(1, min(n_results, 10)),
            }, context)
            if result.get("error"):
                raise ToolError(result["error"])
            return result
        finally:
            if context.get("db"):
                context["db"].close()

    @server.tool()
    def fetch_full_paper(arxiv_id: str, query: str) -> Dict:
        """Search bounded PDF excerpts for one arXiv paper."""
        if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv_id):
            raise ToolError("Expected a modern arXiv identifier")
        if not query.strip() or len(query) > 500:
            raise ToolError("query must contain 1–500 characters")
        context = context_factory(token)
        try:
            result = registry.execute("fetch_full_paper", {"arxiv_id": arxiv_id, "query": query}, context)
            if result.get("error"):
                raise ToolError(result["error"])
            return result
        finally:
            if context.get("db"):
                context["db"].close()

    return server


if __name__ == "__main__":
    create_mcp_server(os.environ.get("LEARNING_ASSISTANT_ACCESS_TOKEN", "")).run(transport="stdio")
