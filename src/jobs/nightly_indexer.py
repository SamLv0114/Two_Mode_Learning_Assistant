"""
Nightly paper indexer.

Runs at 06:00 UTC daily (after ArXiv RSS updates at ~05:00 UTC).
Fetches today's new papers, embeds them in one batch, and upserts into ChromaDB.

This separates ingestion from feed generation so users never wait for embedding.
"""
import logging
from datetime import datetime, timezone

from src.collectors.arxiv_rss_collector import ArxivRSSCollector
from src.database.models import SessionLocal, Paper
from src.models import EmbeddingManager
from src.utils.config import settings

logger = logging.getLogger(__name__)


def run_nightly_index() -> dict:
    """
    Fetch today's ArXiv papers via RSS, embed them, and upsert into ChromaDB.
    Returns a summary dict with counts and elapsed time.
    """
    logger.info("=== Nightly indexer started ===")
    start = datetime.now(timezone.utc)

    result = {"fetched": 0, "new": 0, "indexed": 0, "elapsed_s": 0.0, "error": None}

    try:
        # Step 1: fetch from RSS (fast — one HTTP request per category)
        collector = ArxivRSSCollector()
        papers = collector.fetch(settings.ARXIV_CATEGORIES)
        result["fetched"] = len(papers)
        logger.info(f"Fetched {len(papers)} papers from ArXiv RSS ({len(settings.ARXIV_CATEGORIES)} categories)")

        if not papers:
            logger.warning("No papers fetched — ArXiv RSS may not have updated yet")
            return result

        # Step 2: save new papers to PostgreSQL, skip already-known ones
        # One bulk query instead of N individual lookups
        db = SessionLocal()
        new_papers = []
        try:
            all_ids = [p.arxiv_id for p in papers]
            existing_ids = {
                row[0]
                for row in db.query(Paper.arxiv_id).filter(Paper.arxiv_id.in_(all_ids)).all()
            }
            for p in papers:
                if p.arxiv_id not in existing_ids:
                    db.add(Paper(
                        arxiv_id=p.arxiv_id,
                        title=p.title,
                        authors=", ".join(p.authors),
                        abstract=p.abstract,
                        categories=", ".join(p.categories),
                        published_date=p.published_date,
                        arxiv_url=p.arxiv_url,
                        pdf_url=p.pdf_url,
                    ))
                    new_papers.append(p)
            db.commit()
        finally:
            db.close()

        result["new"] = len(new_papers)
        logger.info(f"{len(new_papers)} new papers (skipped {len(papers) - len(new_papers)} already in DB)")

        if not new_papers:
            logger.info("All papers already indexed — nothing to embed")
            return result

        # Step 3: batch embed all new papers in one pass (most efficient)
        em = EmbeddingManager()
        texts = [f"{p.title}\n\n{p.abstract}" for p in new_papers]
        logger.info(f"Embedding {len(new_papers)} papers in one batch...")
        embeddings = em.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        ).tolist()

        # Step 4: upsert into ChromaDB
        ids = [f"paper_{p.arxiv_id}" for p in new_papers]
        metadatas = [
            {
                "type": "paper",
                "paper_id": p.arxiv_id,
                "title": p.title,
                "url": p.arxiv_url,
                "published_date": str(p.published_date) if p.published_date else "",
            }
            for p in new_papers
        ]
        em.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        result["indexed"] = len(new_papers)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        result["elapsed_s"] = round(elapsed, 1)
        logger.info(f"=== Nightly indexer done: {len(new_papers)} papers indexed in {elapsed:.1f}s ===")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Nightly indexer failed: {e}", exc_info=True)

    return result
