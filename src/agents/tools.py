"""
Tool definitions (OpenAI function-calling schemas) and their executors.

Each tool has:
  - A schema dict (passed to OpenAI as `tools=`)
  - An executor(args, context) → dict

context keys expected: db, user, retriever, embedding_manager
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────

SEARCH_KNOWLEDGE_BASE = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the user's knowledge base (research papers, tech articles, "
            "and uploaded documents) using semantic vector search. "
            "Use this whenever the user asks about a concept, paper, or topic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                },
                "filter_type": {
                    "type": "string",
                    "enum": ["paper", "article", "user_doc"],
                    "description": "Optional: restrict to a specific content type",
                },
            },
            "required": ["query"],
        },
    },
}

GET_PERSONALIZED_FEED = {
    "type": "function",
    "function": {
        "name": "get_personalized_feed",
        "description": (
            "Fetch recent personalized paper and article recommendations "
            "for this user from the database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic_filter": {
                    "type": "string",
                    "description": "Optional topic keyword to narrow results (e.g. 'transformers', 'RL')",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of items to return",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
}

LIST_USER_DOCUMENTS = {
    "type": "function",
    "function": {
        "name": "list_user_documents",
        "description": "List all documents the user has uploaded to their knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

SEARCH_USER_DOCUMENTS = {
    "type": "function",
    "function": {
        "name": "search_user_documents",
        "description": "Search specifically within the user's uploaded documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "n_results": {
                    "type": "integer",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ── Executors ─────────────────────────────────────────────────────────────────

def _exec_search_knowledge_base(args: Dict[str, Any], context: Dict) -> Dict:
    try:
        retriever = context["retriever"]
        results = retriever.retrieve(
            query=args["query"],
            n_results=args.get("n_results", 5),
            filter_type=args.get("filter_type"),
        )
        return {
            "results": [
                {
                    "content": r.get("document", "")[:600],
                    "title": r.get("metadata", {}).get("title", "Unknown"),
                    "type": r.get("metadata", {}).get("type", "unknown"),
                    "url": r.get("metadata", {}).get("url", ""),
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"search_knowledge_base error: {e}")
        return {"error": str(e), "results": [], "count": 0}


def _exec_get_personalized_feed(args: Dict[str, Any], context: Dict) -> Dict:
    try:
        from src.database.models import Paper, Article
        db = context["db"]
        count = args.get("count", 5)
        topic = args.get("topic_filter", "").lower()

        papers = (
            db.query(Paper)
            .order_by(Paper.collected_date.desc())
            .limit(count * 3)
            .all()
        )
        articles = (
            db.query(Article)
            .order_by(Article.collected_date.desc())
            .limit(count * 2)
            .all()
        )

        results = []
        for p in papers:
            if topic and topic not in (p.title or "").lower() and topic not in (p.abstract or "").lower():
                continue
            results.append({
                "type": "paper",
                "title": p.title,
                "summary": p.personalized_summary or (p.abstract[:250] if p.abstract else ""),
                "url": p.arxiv_url or (f"https://arxiv.org/abs/{p.arxiv_id}" if p.arxiv_id else ""),
                "relevance_score": round(p.relevance_score or 0, 3),
            })
            if len(results) >= count:
                break

        for a in articles:
            if len(results) >= count:
                break
            if topic and topic not in (a.title or "").lower():
                continue
            results.append({
                "type": "article",
                "title": a.title,
                "summary": a.personalized_summary or "",
                "url": a.url or "",
                "source": a.source,
                "relevance_score": round(a.relevance_score or 0, 3),
            })

        return {"recommendations": results, "count": len(results)}
    except Exception as e:
        logger.error(f"get_personalized_feed error: {e}")
        return {"error": str(e), "recommendations": [], "count": 0}


def _exec_list_user_documents(args: Dict[str, Any], context: Dict) -> Dict:
    try:
        from src.database.models import UserDocument
        db = context["db"]
        user = context["user"]
        docs = (
            db.query(UserDocument)
            .filter(UserDocument.user_id == user.id)
            .order_by(UserDocument.created_at.desc())
            .all()
        )
        return {
            "documents": [
                {"id": d.id, "title": d.title, "source": d.source, "chunks": d.chunk_count}
                for d in docs
            ],
            "count": len(docs),
        }
    except Exception as e:
        logger.error(f"list_user_documents error: {e}")
        return {"error": str(e), "documents": [], "count": 0}


def _exec_search_user_documents(args: Dict[str, Any], context: Dict) -> Dict:
    try:
        retriever = context["retriever"]
        results = retriever.retrieve(
            query=args["query"],
            n_results=args.get("n_results", 5),
            filter_type="user_doc",
        )
        return {
            "results": [
                {
                    "content": r.get("document", "")[:600],
                    "title": r.get("metadata", {}).get("title", "Unknown"),
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"search_user_documents error: {e}")
        return {"error": str(e), "results": [], "count": 0}


# ── Dispatcher ────────────────────────────────────────────────────────────────

_EXECUTORS = {
    "search_knowledge_base": _exec_search_knowledge_base,
    "get_personalized_feed": _exec_get_personalized_feed,
    "list_user_documents": _exec_list_user_documents,
    "search_user_documents": _exec_search_user_documents,
}


def execute_tool(name: str, args: Dict[str, Any], context: Dict) -> Dict:
    executor = _EXECUTORS.get(name)
    if not executor:
        logger.warning(f"Unknown tool requested: {name}")
        return {"error": f"Unknown tool: {name}"}
    return executor(args, context)
