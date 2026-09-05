"""
Tool definitions (OpenAI function-calling schemas) and their executors.

Each tool has:
  - A schema dict (passed to OpenAI as `tools=`)
  - An executor(args, context) → dict

context keys expected: db, user, retriever, embedding_manager

Replacement 3 — Multi-query retrieval:
  _exec_search_knowledge_base now generates 2 query variants with gpt-4o-mini,
  runs 3 parallel searches, and deduplicates by document ID for higher recall.

Feed access:
  _exec_get_personalized_feed reads back the ranked list DailyFeedPipeline
  produced (UserPaperRecommendation, ordered by rank). It deliberately does NOT
  re-rank — the pipeline is the single authority on what a user should read, so
  the chat answer and the Daily Feed tab always show the same thing.
"""
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ── Multi-query expansion helper ──────────────────────────────────────────────

def _expand_query(query: str) -> List[str]:
    """
    Generate 2 semantically diverse variants of a search query using gpt-4o-mini.
    Returns [original, variant1, variant2] — falls back to [original] on failure.
    """
    try:
        from openai import OpenAI
        from src.utils.config import settings

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = (
            f"Generate 2 short, semantically diverse search query variants for the following query.\n"
            f"Each variant should approach the topic from a different angle (e.g. different keywords,\n"
            f"synonyms, or narrower scope). Return ONLY a JSON array of 2 strings, no explanation.\n\n"
            f"Query: {query}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.4,
        )
        variants = json.loads(resp.choices[0].message.content.strip())
        if isinstance(variants, list) and len(variants) >= 2:
            return [query] + [str(v) for v in variants[:2]]
    except Exception as e:
        logger.debug(f"_expand_query failed: {e}")
    return [query]


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
            "Read this user's current personalized paper feed — the ranked list "
            "already produced by the recommendation pipeline. Returns papers in the "
            "pipeline's own ranking order. If the user has never generated a feed, "
            "the result is empty and carries a 'note' field explaining that; relay "
            "that to the user instead of inventing recommendations."
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

SEARCH_WEB = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for recent information, news, tutorials, or external sources "
            "not in the user's knowledge base. Use when the user asks about recent events, "
            "cutting-edge tools, new model releases, or anything that may not be in the local database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-5)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

SAVE_NOTE = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": (
            "Save a note or intermediate finding to a session scratchpad. "
            "Use this to store key insights, partial answers, or facts you discover "
            "during multi-step research so you can retrieve them later with get_note."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Short label for this note (e.g. 'definition_of_rlhf', 'step2_findings')",
                },
                "content": {
                    "type": "string",
                    "description": "The note content to save",
                },
            },
            "required": ["key", "content"],
        },
    },
}

GET_NOTE = {
    "type": "function",
    "function": {
        "name": "get_note",
        "description": (
            "Retrieve a previously saved note from the session scratchpad by its key. "
            "Returns the content saved with save_note, or an empty string if not found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key used when save_note was called",
                },
            },
            "required": ["key"],
        },
    },
}


# ── Executors ─────────────────────────────────────────────────────────────────

def _exec_search_knowledge_base(args: Dict[str, Any], context: Dict) -> Dict:
    """
    Multi-query retrieval: generates 2 query variants with gpt-4o-mini,
    runs 3 searches, deduplicates by document ID for higher recall.
    Replacement 3: replaces single-vector search with multi-query fusion.
    """
    try:
        retriever = context["retriever"]
        query = args["query"]
        n = args.get("n_results", 5)
        filter_type = args.get("filter_type")

        queries = _expand_query(query)
        seen_ids: set = set()
        merged: List[Dict] = []

        for q in queries:
            batch = retriever.retrieve(query=q, n_results=n, filter_type=filter_type)
            for r in batch:
                # Deduplicate on ChromaDB document id (stored in metadata or use title)
                doc_id = r.get("id") or r.get("metadata", {}).get("paper_id") or r.get("metadata", {}).get("title", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(r)

        # Sort by distance (lower = more similar); take top-n
        merged.sort(key=lambda r: r.get("distance", 1.0))
        top = merged[:n]

        logger.info(f"search_knowledge_base: {len(queries)} queries → {len(merged)} unique → top {len(top)}")
        return {
            "results": [
                {
                    "content": r.get("document", "")[:600],
                    "title": r.get("metadata", {}).get("title", "Unknown"),
                    "type": r.get("metadata", {}).get("type", "unknown"),
                    "url": r.get("metadata", {}).get("url", ""),
                }
                for r in top
            ],
            "count": len(top),
            "queries_used": len(queries),
        }
    except Exception as e:
        logger.error(f"search_knowledge_base error: {e}")
        return {"error": str(e), "results": [], "count": 0}


def _exec_get_personalized_feed(args: Dict[str, Any], context: Dict) -> Dict:
    """
    Read this user's current feed — the ranked list DailyFeedPipeline produced.

    The pipeline is the single authority on "what should this user read": it runs
    ChromaDB retrieval, mode-aware multi-signal ranking, familiar-paper penalty and
    (past 50 interactions) LightGBM. This tool does not re-rank; it reads that
    result back in the pipeline's own order so the chat answer and the Daily Feed
    tab always agree.

    Returns an explicit empty result when the user has never generated a feed,
    so the agent can say so rather than inventing recommendations.
    """
    try:
        from src.database.models import Paper, UserPaperRecommendation

        db = context["db"]
        user = context["user"]
        count = args.get("count", 5)
        topic = (args.get("topic_filter") or "").lower()

        q = (
            db.query(Paper, UserPaperRecommendation)
            .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
            .filter(UserPaperRecommendation.user_id == user.id)
            .order_by(UserPaperRecommendation.rank.asc())
        )

        # A topic filter narrows the existing feed — it never reorders it.
        rows = q.all() if topic else q.limit(count).all()

        results = []
        for paper, rec in rows:
            if topic and topic not in f"{paper.title or ''} {paper.abstract or ''}".lower():
                continue
            results.append({
                "type": "paper",
                "rank": rec.rank,
                "title": paper.title,
                "summary": rec.personalized_summary or (paper.abstract[:250] if paper.abstract else ""),
                "url": paper.arxiv_url or (f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else ""),
                "relevance_score": round(rec.relevance_score or 0.0, 3),
                "citation_count": paper.citation_count,
            })
            if len(results) >= count:
                break

        if not results:
            return {
                "recommendations": [],
                "count": 0,
                "note": (
                    "No feed has been generated for this user yet. "
                    "Ask them to click Generate Feed on the Daily Feed tab first."
                    if not topic else
                    f"The current feed has no papers matching '{topic}'."
                ),
            }

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


def _exec_search_web(args: Dict[str, Any], context: Dict) -> Dict:
    try:
        from tavily import TavilyClient
        from src.utils.config import settings

        api_key = settings.TAVILY_API_KEY
        if not api_key:
            return {"error": "TAVILY_API_KEY not configured", "results": [], "count": 0}

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=args["query"],
            max_results=min(args.get("max_results", 3), 5),
            search_depth="basic",
        )

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
                "score": round(r.get("score", 0), 3),
            }
            for r in response.get("results", [])
        ]
        return {"results": results, "count": len(results)}
    except ImportError:
        return {"error": "tavily-python not installed", "results": [], "count": 0}
    except Exception as e:
        logger.error(f"search_web error: {e}")
        return {"error": str(e), "results": [], "count": 0}


def _exec_save_note(args: Dict[str, Any], context: Dict) -> Dict:
    """Store a note in the session scratchpad (context["_agent_notes"])."""
    notes = context.setdefault("_agent_notes", {})
    key = args.get("key", "note")
    content = args.get("content", "")
    notes[key] = content
    logger.debug(f"save_note: key='{key}' ({len(content)} chars)")
    return {"saved": True, "key": key, "length": len(content)}


def _exec_get_note(args: Dict[str, Any], context: Dict) -> Dict:
    """Retrieve a note from the session scratchpad."""
    notes = context.get("_agent_notes", {})
    key = args.get("key", "")
    content = notes.get(key)
    return {"found": content is not None, "key": key, "content": content or ""}


# ── Dispatcher ────────────────────────────────────────────────────────────────


def execute_tool(name: str, args: Dict[str, Any], context: Dict) -> Dict:
    """
    Dispatch a tool call by name.

    Thin wrapper over the shared ToolRegistry — kept so callers that dispatch a
    single known tool (DeepResearchAgent, PlanAndSolveAgent) don't each need a
    registry handle. There is exactly one dispatch path underneath.
    """
    return build_default_registry().execute(name, args, context)


_DEFAULT_REGISTRY = None


def build_default_registry():
    """
    V4: Return the shared ToolRegistry pre-loaded with all standard agent tools.

    The registry holds only (schema, executor) pairs — it is stateless and
    read-only at call time, so a single module-level instance is shared by every
    agent rather than rebuilt per instantiation.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        from src.agents.core.tool_registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(SEARCH_KNOWLEDGE_BASE, _exec_search_knowledge_base)
        reg.register(GET_PERSONALIZED_FEED, _exec_get_personalized_feed)
        reg.register(LIST_USER_DOCUMENTS, _exec_list_user_documents)
        reg.register(SEARCH_USER_DOCUMENTS, _exec_search_user_documents)
        reg.register(SEARCH_WEB, _exec_search_web)
        reg.register(SAVE_NOTE, _exec_save_note)
        reg.register(GET_NOTE, _exec_get_note)
        _DEFAULT_REGISTRY = reg
    return _DEFAULT_REGISTRY
