"""Durable database-backed worker for Coding Agent graph and comparison jobs."""
import logging
import signal
import time

from sqlalchemy import select, update

from src.agents.coding_runtime import run_comparison_job, run_graph_job
from src.database.models import PracticeComparison, PracticeRun, SessionLocal
from src.utils.config import settings

logger = logging.getLogger(__name__)
_stop = False


def _request_stop(*_args):
    global _stop
    _stop = True


def recover_interrupted_jobs() -> None:
    """Return jobs owned by a terminated worker to the durable queue."""
    with SessionLocal() as db:
        runs = db.execute(update(PracticeRun).where(PracticeRun.status == "running").values(
            status="queued", stage="resuming", error=None)).rowcount
        comparisons = db.execute(update(PracticeComparison).where(
            PracticeComparison.status == "running").values(status="queued", error=None)).rowcount
        db.commit()
    logger.info("Recovered %s Coding Agent runs and %s comparisons", runs, comparisons)


def _claim_run() -> str | None:
    with SessionLocal() as db:
        row = db.execute(select(PracticeRun).where(PracticeRun.status == "queued")
                         .order_by(PracticeRun.created_at).with_for_update(skip_locked=True).limit(1)).scalar_one_or_none()
        if row is None:
            return None
        row.status = "running"
        row.stage = "starting" if row.stage == "queued" else "resuming"
        db.commit()
        return row.run_id


def _claim_comparison() -> str | None:
    with SessionLocal() as db:
        row = db.execute(select(PracticeComparison).where(PracticeComparison.status == "queued")
                         .order_by(PracticeComparison.created_at).with_for_update(skip_locked=True).limit(1)).scalar_one_or_none()
        if row is None:
            return None
        row.status = "running"
        db.commit()
        return row.run_id


def run_forever() -> None:
    # The API owns schema initialization. Compose starts this process only
    # after the API is healthy, avoiding concurrent ALTER TABLE attempts.
    recover_interrupted_jobs()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    logger.info("Coding Agent task worker started")
    while not _stop:
        run_id = _claim_run()
        if run_id:
            run_graph_job(run_id)
            continue
        comparison_id = _claim_comparison()
        if comparison_id:
            run_comparison_job(comparison_id)
            continue
        time.sleep(settings.PRACTICE_TASK_POLL_SECONDS)
    logger.info("Coding Agent task worker stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_forever()
