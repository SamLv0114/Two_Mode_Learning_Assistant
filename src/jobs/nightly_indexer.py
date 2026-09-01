"""
Nightly paper indexer.

Runs at 06:00 UTC daily (after ArXiv RSS updates at ~05:00 UTC).
Fetches today's new papers, embeds them in one batch, and upserts into ChromaDB.

This separates ingestion from feed generation so users never wait for embedding.

Replacement 2 — UserFactMemory integration:
  generate_personalized_summaries(user_id, redis_client) can be called after
  run_nightly_index() to produce per-user personalized_summary fields for new
  papers using dynamic facts extracted from their conversation history.
  Replaces the static settings.USER_INTERESTS filter with live per-user context.
"""
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

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
                "citation_count": 0,  # real count backfilled by citation_refresh after S2 indexes the paper
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


def generate_personalized_summaries(
    user_id: int,
    redis_client=None,
    max_papers: int = 20,
) -> dict:
    """
    Replacement 2 — UserFactMemory replaces static USER_INTERESTS.

    For a given user, reads their dynamic long-term facts from UserFactMemory
    and generates a personalized_summary for recent papers that match their
    interests using gpt-4o-mini. Stores the summary back to the Paper row.

    Call this after run_nightly_index() as a per-user post-processing step,
    e.g. triggered lazily on first login after midnight.

    Returns a dict with counts and any error.
    """
    from src.agents.memory import UserFactMemory
    from openai import OpenAI

    result = {"user_id": user_id, "processed": 0, "skipped": 0, "error": None}

    try:
        fact_mem = UserFactMemory()
        facts_str = fact_mem.get_context_string(user_id, redis_client)
        if not facts_str.strip():
            logger.info(f"generate_personalized_summaries: no facts for user {user_id}, skipping")
            result["skipped"] = max_papers
            return result

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        db = SessionLocal()
        try:
            # Fetch recent papers without a personalized summary for this user
            recent_papers: List[Paper] = (
                db.query(Paper)
                .filter(Paper.personalized_summary.is_(None))
                .order_by(Paper.collected_date.desc())
                .limit(max_papers)
                .all()
            )

            for paper in recent_papers:
                try:
                    prompt = (
                        f"You are summarizing an academic paper for a specific researcher.\n\n"
                        f"Paper: {paper.title}\n"
                        f"Abstract: {(paper.abstract or '')[:600]}\n\n"
                        f"Researcher's known interests and goals:\n{facts_str}\n\n"
                        f"Write a 2-sentence personalized summary that:\n"
                        f"1. Highlights the aspects most relevant to this researcher's interests\n"
                        f"2. Explains why this paper matters for their specific goals\n"
                        f"Be concrete and direct. No filler phrases."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.3,
                    )
                    paper.personalized_summary = resp.choices[0].message.content.strip()
                    result["processed"] += 1
                except Exception as e:
                    logger.warning(f"Summary generation failed for paper {paper.arxiv_id}: {e}")
                    result["skipped"] += 1

            db.commit()
        finally:
            db.close()

        logger.info(
            f"generate_personalized_summaries: user {user_id} — "
            f"{result['processed']} summaries generated, {result['skipped']} skipped"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"generate_personalized_summaries failed: {e}", exc_info=True)

    return result
