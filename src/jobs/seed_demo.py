"""
Create or top up the shared demo account behind the landing page's "Try the demo".

    docker compose exec api python -m src.jobs.seed_demo           # create, or fill in what is missing
    docker compose exec api python -m src.jobs.seed_demo --reset   # also clear what visitors changed

Each step is skipped when it is already done:
  1. the demo user, with a research profile and a random password nobody knows
     (visitors sign in through POST /auth/demo, which needs DEMO_ENABLED=true)
  2. a few saved papers matching that profile, so the feed is personalised from the first visit
  3. today's feed (summaries by gpt-4o-mini: a few cents)
  4. showcase Coding Agent runs, one paper and one repository, queued for the practice worker,
     so visitors open finished results instead of waiting for one (needs PRACTICE_ENABLED)

--reset removes the account's interactions, feed and trained ranker first; visitors' saves and
dismissals accumulate on the shared account otherwise. Coding Agent runs are kept.
"""
import argparse
import json
import logging
import secrets
import uuid

from src.api.security import get_password_hash
from src.agents import research_profile
from src.database.models import (Paper, PracticeRun, SessionLocal, User, UserInteraction, UserModelState,
                                 UserPaperImpression, UserPaperRecommendation, init_db,
                                 upsert_user_interaction)
from src.utils.config import settings

logger = logging.getLogger(__name__)

PROFILE = {
    "full_name": "Demo learner",
    "focus_areas": ["LLM", "Agent", "DL"],
    "interests": [
        "parameter-efficient fine-tuning of large language models",
        "retrieval-augmented generation",
        "LLM agents and tool use",
    ],
}
# The research profile recommendation is built from (src/agents/research_profile.py).
# Visitors cannot edit it (PUT /profile/research is forbid_demo); seeding restores it.
RESEARCH_PROFILE = {
    "goal": "Build LLM agents that use tools and retrieval reliably, and fine-tune them cheaply",
    "level": "intermediate",
    "known": ["transformers", "prompt engineering"],
    "interests": PROFILE["interests"],
    "preferences": ["papers with code", "empirical results over theory"],
    "avoid_types": ["position"],
    "seed_papers": [
        {"id": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Language Models", "year": 2021},
        {"id": "2210.03629", "title": "ReAct: Synergizing Reasoning and Acting in Language Models", "year": 2022},
        {"id": "2005.11401", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "year": 2020},
    ],
    "summary": "",
}
SAVED_PER_INTEREST = 2
SHOWCASE_RUNS = [
    {"kind": "paper", "source": "2106.09685",  # LoRA
     "objective": "Implement the LoRA update of a linear layer and show that only the adapter trains"},
    {"kind": "repo", "source": "https://github.com/karpathy/nanoGPT"},
]


def ensure_user(db) -> User:
    user = db.query(User).filter(User.email == settings.DEMO_USER_EMAIL).first()
    if user is None:
        user = User(email=settings.DEMO_USER_EMAIL, hashed_password=get_password_hash(secrets.token_urlsafe(32)))
        db.add(user)
        logger.info("Created demo user %s", settings.DEMO_USER_EMAIL)
    user.full_name = PROFILE["full_name"]
    user.is_active = True
    user.set_interests_list(PROFILE["interests"])
    user.set_focus_areas_list(PROFILE["focus_areas"])
    db.commit()
    research_profile.save(db, user, RESEARCH_PROFILE)
    return user


def reset(db, user: User) -> None:
    for model in (UserInteraction, UserPaperRecommendation, UserPaperImpression, UserModelState):
        db.query(model).filter(model.user_id == user.id).delete(synchronize_session=False)
    db.commit()
    logger.info("Cleared the demo account's interactions, feed and ranker")


def seed_saved_papers(db, user: User, embedding_manager) -> None:
    if db.query(UserInteraction).filter(UserInteraction.user_id == user.id).count():
        return
    saved = 0
    for interest in PROFILE["interests"]:
        for hit in embedding_manager.search(interest, n_results=SAVED_PER_INTEREST, filter_type="paper"):
            # Nightly-indexed vectors carry only paper_id; add_paper also writes arxiv_id.
            meta = hit.get("metadata") or {}
            arxiv_id = meta.get("paper_id") or meta.get("arxiv_id")
            paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first() if arxiv_id else None
            if paper:
                upsert_user_interaction(db, user.id, "paper", paper.id, "saved")
                saved += 1
    db.commit()
    logger.info("Saved %d papers for the demo profile", saved)


def seed_feed(db, user: User, embedding_manager) -> None:
    from src.pipelines.daily_feed import DailyFeedPipeline
    DailyFeedPipeline(user.id, db, embedding_manager).run(
        focus_areas=PROFILE["focus_areas"], user_interests=PROFILE["interests"], mode="recommended")
    logger.info("Demo feed ready")


def seed_showcase_runs(db, user: User) -> None:
    if not settings.PRACTICE_ENABLED:
        logger.warning("PRACTICE_ENABLED is off: skipping showcase Coding Agent runs")
        return
    from src.api.routers.practice import _worker_available
    for run in SHOWCASE_RUNS:
        exists = db.query(PracticeRun).filter(
            PracticeRun.user_id == user.id, PracticeRun.source == run["source"],
            PracticeRun.status.notin_(["failed", "cancelled"])).first()
        if exists:
            continue
        request = ({"objective": run["objective"], "execution_enabled": _worker_available(), "confirm_plan": False}
                   if run["kind"] == "paper" else None)
        db.add(PracticeRun(run_id=str(uuid.uuid4()), user_id=user.id, kind=run["kind"], source=run["source"],
                           request_json=json.dumps(request) if request else None,
                           status="queued", stage="queued"))
        db.commit()
        logger.info("Queued showcase %s run for %s", run["kind"], run["source"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reset", action="store_true", help="clear what visitors changed before seeding")
    args = parser.parse_args()

    from src.models.embeddings import EmbeddingManager
    init_db()
    db = SessionLocal()
    try:
        user = ensure_user(db)
        if args.reset:
            reset(db, user)
        embedding_manager = EmbeddingManager()
        seed_saved_papers(db, user, embedding_manager)
        seed_feed(db, user, embedding_manager)
        seed_showcase_runs(db, user)
    finally:
        db.close()
    if not settings.DEMO_ENABLED:
        logger.warning("DEMO_ENABLED is off: the account exists but 'Try the demo' stays hidden")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
