"""
Research profile endpoints: the editable card recommendation is built from.

  GET  /profile/research            the saved profile (null before onboarding)
  POST /profile/research/draft      a description in the user's words -> fields to confirm
  PUT  /profile/research            save the confirmed or edited profile
  POST /profile/research/feedback   plain-language feedback -> a profile edit
  GET  /profile/research/papers     find papers to pin as "papers I found useful"
  POST /profile/reading-path        an ordered path of papers into a topic (reading_path_agent)
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.agents import research_profile
from src.agents.reading_path_agent import build_reading_path
from src.agents.core.structured import StructuredOutputError
from src.api.demo import demo_quota, forbid_demo
from src.api.deps import get_current_user, get_db_session, get_embedding_manager
from src.collectors import semantic_scholar
from src.database.models import Paper, User
from src.models.embeddings import EmbeddingManager
from src.pipelines.feed.channels import search_corpus
from src.utils.config import settings
from src.utils.llm_usage import request_budget
from src.utils.tracing import openai_client

router = APIRouter(prefix="/profile", tags=["Research Profile"])
MODEL_UNAVAILABLE = "The model could not process this right now. Please try again, or fill in the fields yourself."


def _payload(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The profile as the client sees it; set_up says whether recommendation can use it yet."""
    return {**data, "set_up": research_profile.is_set_up(data)} if data else None


class SeedPaper(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    title: str = Field(default="", max_length=300)
    year: Optional[int] = None


class ProfileBody(BaseModel):
    goal: str = Field(default="", max_length=500)
    level: Optional[str] = None
    known: List[str] = Field(default_factory=list, max_length=30)
    interests: List[str] = Field(default_factory=list, max_length=30)
    preferences: List[str] = Field(default_factory=list, max_length=20)
    avoid_types: List[str] = Field(default_factory=list, max_length=10)
    seed_papers: List[SeedPaper] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=1500)


class DraftBody(BaseModel):
    description: str = Field(min_length=10, max_length=2000)


class ReadingPathBody(BaseModel):
    topic: str = Field(min_length=3, max_length=500)


class FeedbackBody(BaseModel):
    text: str = Field(min_length=3, max_length=1000)
    paper_id: Optional[int] = None


def _paper_option(paper: Paper) -> Dict[str, Any]:
    return {"id": paper.arxiv_id, "title": paper.title,
            "year": paper.published_date.year if paper.published_date else None,
            "url": paper.arxiv_url or f"https://arxiv.org/abs/{paper.arxiv_id}", "source": "library"}


def _library_matches(db: Session, em: EmbeddingManager, query: str, limit: int) -> List[Dict[str, Any]]:
    """Papers already in the corpus: title matches first, then the closest by meaning."""
    by_title = db.query(Paper).filter(Paper.title.ilike(f"%{query[:200]}%")).limit(limit).all()
    papers = {p.arxiv_id: p for p in by_title + search_corpus(db, em, query, limit)}
    return [_paper_option(p) for p in papers.values()][:limit]


def find_papers(db: Session, em: EmbeddingManager, query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """The corpus, plus Semantic Scholar for papers the corpus does not have (best effort)."""
    results = _library_matches(db, em, query, limit)
    seen = {r["id"] for r in results}
    for paper in semantic_scholar.search(query, limit=limit) or []:
        identifier = paper.arxiv_id or paper.paper_id
        if identifier not in seen:
            seen.add(identifier)
            results.append({"id": identifier, "title": paper.title, "year": paper.year, "url": paper.url,
                            "source": "semantic_scholar"})
    return results[:limit]


@router.get("/research")
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db_session)):
    return {"profile": _payload(research_profile.to_dict(research_profile.load(db, current_user.id)))}


@router.post("/research/draft")
def draft_profile(body: DraftBody, current_user: User = Depends(demo_quota), db: Session = Depends(get_db_session),
                  em: EmbeddingManager = Depends(get_embedding_manager)):
    try:
        draft = research_profile.draft_from_description(openai_client(), body.description)
    except StructuredOutputError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=MODEL_UNAVAILABLE) from None
    suggestions: Dict[str, Dict[str, Any]] = {}
    for term in draft.pop("seed_search_terms"):
        for paper in _library_matches(db, em, term, 2):
            suggestions.setdefault(paper["id"], paper)
    return {"draft": draft, "suggested_papers": list(suggestions.values())[:6]}


@router.put("/research")
def save_profile(body: ProfileBody, current_user: User = Depends(forbid_demo), db: Session = Depends(get_db_session)):
    profile = research_profile.save(db, current_user, body.model_dump())
    return {"profile": _payload(research_profile.to_dict(profile))}


@router.post("/research/feedback")
def profile_feedback(body: FeedbackBody, current_user: User = Depends(forbid_demo),
                     db: Session = Depends(get_db_session)):
    title = ""
    if body.paper_id is not None:
        paper = db.query(Paper).filter(Paper.id == body.paper_id).first()
        title = paper.title if paper else ""
    try:
        result = research_profile.apply_feedback(db, openai_client(), current_user, body.text, title)
    except StructuredOutputError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=MODEL_UNAVAILABLE) from None
    return {**result, "profile": _payload(result["profile"])}


@router.get("/research/papers")
def search_papers(q: str = Query(min_length=3, max_length=300), current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db_session), em: EmbeddingManager = Depends(get_embedding_manager)):
    return {"papers": find_papers(db, em, q)}


@router.post("/reading-path")
def reading_path(body: ReadingPathBody, current_user: User = Depends(demo_quota),
                 db: Session = Depends(get_db_session), em: EmbeddingManager = Depends(get_embedding_manager)):
    profile = research_profile.to_dict(research_profile.load(db, current_user.id))
    with request_budget("reading_path", settings.READING_PATH_TOKEN_BUDGET):
        return build_reading_path(db, em, openai_client(), body.topic, profile)
