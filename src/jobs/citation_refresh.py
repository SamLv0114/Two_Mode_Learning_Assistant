"""
Citation count refresh job.

The nightly indexer stores brand-new ArXiv papers with citation_count=0
because Semantic Scholar takes 2–4 weeks to index new papers. This job
backfills the real counts once papers are old enough for S2 to have them.

Run weekly (e.g. every Sunday):
    python -m src.jobs.citation_refresh
    python -m src.jobs.citation_refresh --min-age-days 14 --batch-size 500
"""
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from src.database.models import SessionLocal, Paper
from src.models import EmbeddingManager
from src.utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_MAX_BATCH = 500  # S2 hard limit per batch request


def _fetch_s2_batch(arxiv_ids: list[str], api_key: Optional[str]) -> dict[str, int]:
    """
    POST to S2 batch endpoint for up to 500 ArXiv IDs.
    Returns {arxiv_id: citation_count} for papers S2 found.
    Papers not found on S2 are omitted from the result.
    """
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(3):
        try:
            resp = requests.post(
                _S2_BATCH_URL,
                json={"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]},
                params={"fields": "externalIds,citationCount"},
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning(f"  Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                logger.error(f"  Request failed after 3 attempts: {e}")
                return {}
            time.sleep(5)
    else:
        return {}

    found = {}
    for item in resp.json():
        if not item:
            continue
        arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
        if arxiv_id and item.get("citationCount") is not None:
            found[arxiv_id] = item["citationCount"]
    return found


def _update_postgres(updates: dict[str, int]) -> int:
    """Update citation_count rows in PostgreSQL. Returns number of rows updated."""
    if not updates:
        return 0
    db = SessionLocal()
    try:
        for arxiv_id, count in updates.items():
            db.query(Paper).filter(Paper.arxiv_id == arxiv_id).update(
                {"citation_count": count},
                synchronize_session=False,
            )
        db.commit()
        return len(updates)
    finally:
        db.close()


def _update_chromadb(em: EmbeddingManager, updates: dict[str, int]) -> int:
    """
    Batch-update citation_count in ChromaDB metadata.
    Fetches existing metadata first to preserve all other fields.
    Returns number of documents updated.
    """
    if not updates:
        return 0

    chroma_ids = [f"paper_{aid}" for aid in updates]
    try:
        existing = em.collection.get(ids=chroma_ids, include=["metadatas"])
    except Exception as e:
        logger.warning(f"  ChromaDB batch get failed: {e}")
        return 0

    updated_ids, updated_metadatas = [], []
    for i, cid in enumerate(existing["ids"]):
        arxiv_id = cid[len("paper_"):]
        if arxiv_id in updates:
            meta = existing["metadatas"][i].copy()
            meta["citation_count"] = updates[arxiv_id]
            updated_ids.append(cid)
            updated_metadatas.append(meta)

    if not updated_ids:
        return 0
    try:
        em.collection.update(ids=updated_ids, metadatas=updated_metadatas)
    except Exception as e:
        logger.warning(f"  ChromaDB batch update failed: {e}")
        return 0
    return len(updated_ids)


def run_citation_refresh(min_age_days: int = 14, batch_size: int = _MAX_BATCH) -> dict:
    """
    Find papers with citation_count=0 that are old enough for S2 to have indexed
    them, fetch their real citation counts, and update PostgreSQL + ChromaDB.

    citation_count=0 is used as the "not yet fetched" marker since the nightly
    indexer doesn't look up S2 (new papers aren't on S2 yet when first ingested).
    After min_age_days, S2 will typically have indexed them.
    """
    batch_size = min(batch_size, _MAX_BATCH)
    start = datetime.now(timezone.utc)
    cutoff = start - timedelta(days=min_age_days)
    api_key = settings.SEMANTIC_SCHOLAR_API_KEY

    result = {
        "papers_checked": 0,
        "found_on_s2": 0,
        "postgres_updated": 0,
        "chromadb_updated": 0,
        "elapsed_s": 0.0,
        "error": None,
    }

    logger.info("=" * 65)
    logger.info("CITATION REFRESH")
    logger.info(f"  Target:         citation_count = 0, age >= {min_age_days}d")
    logger.info(f"  Cutoff date:    {cutoff.date()}")
    logger.info(f"  Batch size:     {batch_size}")
    logger.info(f"  API key:        {'yes (higher limits)' if api_key else 'no (public rate limit)'}")
    logger.info("=" * 65)

    try:
        db = SessionLocal()
        try:
            rows = (
                db.query(Paper.arxiv_id)
                .filter(Paper.citation_count == 0)
                .filter(Paper.published_date < cutoff)
                .all()
            )
        finally:
            db.close()

        arxiv_ids = [r[0] for r in rows]
        result["papers_checked"] = len(arxiv_ids)
        logger.info(f"\nPapers with citation_count=0 (age >= {min_age_days}d): {len(arxiv_ids):,}")

        if not arxiv_ids:
            logger.info("Nothing to update — all caught up.")
            return result

        em = EmbeddingManager()
        total_batches = (len(arxiv_ids) + batch_size - 1) // batch_size

        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i: i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"\n[{batch_num}/{total_batches}] Querying S2 for {len(batch)} papers...")

            updates = _fetch_s2_batch(batch, api_key)
            result["found_on_s2"] += len(updates)

            pg_updated = _update_postgres(updates)
            result["postgres_updated"] += pg_updated

            ch_updated = _update_chromadb(em, updates)
            result["chromadb_updated"] += ch_updated

            logger.info(
                f"  Found on S2: {len(updates):>4}/{len(batch)}  "
                f"| PG updated: {pg_updated:>4}  "
                f"| ChromaDB updated: {ch_updated:>4}"
            )
            time.sleep(1.1 if not api_key else 0.2)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"\nCitation refresh FAILED: {e}", exc_info=True)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    result["elapsed_s"] = round(elapsed, 1)

    logger.info("\n" + "=" * 65)
    logger.info("CITATION REFRESH SUMMARY")
    logger.info(f"  Papers checked:      {result['papers_checked']:>8,}")
    logger.info(f"  Found on S2:         {result['found_on_s2']:>8,}")
    logger.info(f"  PostgreSQL updated:  {result['postgres_updated']:>8,}")
    logger.info(f"  ChromaDB updated:    {result['chromadb_updated']:>8,}")
    logger.info(f"  Total time:          {elapsed:>8.1f}s")
    if result["error"]:
        logger.info(f"  Error:               {result['error']}")
    logger.info("=" * 65)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill citation counts for papers with citation_count=0")
    parser.add_argument("--min-age-days", type=int, default=14,
                        help="Only update papers at least this many days old (default: 14)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Papers per S2 batch API call (max 500, default: 500)")
    args = parser.parse_args()
    run_citation_refresh(min_age_days=args.min_age_days, batch_size=args.batch_size)
