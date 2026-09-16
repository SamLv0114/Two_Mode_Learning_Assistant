"""
One-time backfill: regenerate UserPaperRecommendation.personalized_summary
for existing rows using the current Generator.generate_summary prompt.

Existing rows keep whatever wording was current when they were generated.
A prompt change (e.g. persona/topic framing) never retroactively updates
already-stored summaries, only new ones. Run this after such a change to
refresh what's already in the database instead of waiting for each row's
next natural feed regeneration.

Run (inside API container):
    docker exec learning_assistant_api python -m src.jobs.backfill_summaries
    docker exec learning_assistant_api python -m src.jobs.backfill_summaries --user-id 3
    docker exec learning_assistant_api python -m src.jobs.backfill_summaries --dry-run

Costs one gpt-4o-mini call per row. Use --dry-run first to see the row count
before spending anything.
"""
import argparse
import logging
from datetime import datetime, timezone

from src.database.models import SessionLocal, Paper, User, UserPaperRecommendation
from src.rag.generator import Generator
from src.utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMMIT_EVERY = 20


def run_backfill(user_id: int = None, dry_run: bool = False) -> dict:
    start = datetime.now(timezone.utc)
    result = {"total": 0, "updated": 0, "failed": 0, "elapsed_s": 0.0, "error": None}

    db = SessionLocal()
    generator = Generator()

    try:
        query = (
            db.query(UserPaperRecommendation, Paper, User)
            .join(Paper, Paper.id == UserPaperRecommendation.paper_id)
            .join(User, User.id == UserPaperRecommendation.user_id)
        )
        if user_id is not None:
            query = query.filter(UserPaperRecommendation.user_id == user_id)

        rows = query.all()
        result["total"] = len(rows)
        logger.info(f"{len(rows)} recommendation rows to regenerate" + (" (dry run)" if dry_run else ""))

        for i, (rec, paper, user) in enumerate(rows, 1):
            interests = user.get_focus_areas_list() or settings.USER_INTERESTS
            logger.info(f"[{i}/{len(rows)}] user={user.id} paper={paper.arxiv_id!r} {paper.title[:60]!r}")

            if dry_run:
                continue

            try:
                rec.personalized_summary = generator.generate_summary(
                    paper.title, paper.abstract or "", interests
                )
                result["updated"] += 1
            except Exception as e:
                logger.warning(f"  failed: {e}")
                result["failed"] += 1

            if i % COMMIT_EVERY == 0:
                db.commit()

        if not dry_run:
            db.commit()

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Backfill failed: {e}", exc_info=True)
    finally:
        db.close()

    result["elapsed_s"] = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        f"Done: {result['updated']}/{result['total']} updated, "
        f"{result['failed']} failed, {result['elapsed_s']:.1f}s"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, default=None, help="Only regenerate for this user")
    parser.add_argument("--dry-run", action="store_true", help="List affected rows without calling the LLM or writing")
    args = parser.parse_args()
    run_backfill(user_id=args.user_id, dry_run=args.dry_run)
