"""
One-time backfill: index all PostgreSQL papers into ChromaDB.

Run on the server:
    docker exec learning_assistant_api python -m src.jobs.backfill_chromadb

Safe to re-run — already-indexed papers are skipped via ChromaDB ID check.
"""
import logging
from datetime import datetime, timezone

from src.database.models import SessionLocal, Paper
from src.models import EmbeddingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 200


def run_backfill() -> dict:
    start = datetime.now(timezone.utc)
    result = {"total": 0, "indexed": 0, "skipped": 0, "elapsed_s": 0.0, "error": None}

    db = SessionLocal()
    em = EmbeddingManager()

    try:
        total = db.query(Paper).count()
        result["total"] = total
        logger.info(f"PostgreSQL has {total} papers total")

        indexed = 0
        skipped = 0

        for offset in range(0, total, BATCH_SIZE):
            papers = db.query(Paper).offset(offset).limit(BATCH_SIZE).all()
            if not papers:
                break

            # Check which are already in ChromaDB (batch lookup — fast)
            chroma_ids = [f"paper_{p.arxiv_id}" for p in papers]
            existing = em.collection.get(ids=chroma_ids, include=[])
            existing_set = set(existing["ids"])

            to_index = [p for p in papers if f"paper_{p.arxiv_id}" not in existing_set]
            skipped += len(papers) - len(to_index)

            if not to_index:
                logger.info(
                    f"Batch {offset // BATCH_SIZE + 1}: all {len(papers)} already indexed"
                )
                continue

            # Batch embed (title + abstract)
            texts = [f"{p.title}\n\n{p.abstract or ''}" for p in to_index]
            embeddings = em.model.encode(
                texts, show_progress_bar=False, batch_size=32
            ).tolist()

            em.collection.upsert(
                ids=[f"paper_{p.arxiv_id}" for p in to_index],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "type": "paper",
                        "paper_id": p.arxiv_id,
                        "title": p.title,
                        "url": p.arxiv_url or f"https://arxiv.org/abs/{p.arxiv_id}",
                        "published_date": str(p.published_date) if p.published_date else "",
                    }
                    for p in to_index
                ],
            )

            indexed += len(to_index)
            done = min(offset + BATCH_SIZE, total)
            logger.info(
                f"Progress: {done}/{total} processed — "
                f"{indexed} indexed, {skipped} skipped"
            )

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        result.update({"indexed": indexed, "skipped": skipped, "elapsed_s": round(elapsed, 1)})
        logger.info(
            f"Backfill complete: {indexed} new papers indexed, "
            f"{skipped} already existed — {elapsed:.1f}s"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Backfill failed: {e}", exc_info=True)
    finally:
        db.close()

    return result


if __name__ == "__main__":
    run_backfill()
