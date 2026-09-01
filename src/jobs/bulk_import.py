"""
Bulk historical paper import — citation-ranked via Semantic Scholar.

Instead of random ArXiv time-slices, this fetches the MOST CITED papers
across key ML/AI topics using Semantic Scholar's search API, which already
includes citation counts. Papers are deduplicated, sorted by citation count
descending, then saved to PostgreSQL and indexed into ChromaDB.

Run (inside API container):
    python -m src.jobs.bulk_import
    python -m src.jobs.bulk_import --min-citations 50 --max-per-query 1000
    python -m src.jobs.bulk_import --min-citations 10 --max-per-query 2000
"""
import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from src.collectors.arxiv_collector import PaperData
from src.database.models import SessionLocal, Paper
from src.models import EmbeddingManager
from src.utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_S2_BASE = "https://api.semanticscholar.org"
_EMBED_BATCH_SIZE = 200

# Broad queries covering all major ML/AI subfields.
# Overlap between queries is intentional — duplicates are removed after fetching.
SEARCH_QUERIES = [
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "transformer attention mechanism",
    "convolutional neural network",
    "generative adversarial network",
    "large language model",
    "graph neural network",
    "object detection image",
    "transfer learning fine-tuning",
    "self-supervised contrastive learning",
    "knowledge distillation compression",
    "diffusion model generative",
    "BERT GPT language model",
    "federated learning privacy",
    "neural architecture search",
    "recommendation system collaborative filtering",
]


# ── Step 1: Fetch from Semantic Scholar ──────────────────────────────────────

def fetch_from_semantic_scholar(
    queries: list[str],
    min_citations: int = 50,
    max_per_query: int = 1000,
    api_key: Optional[str] = None,
) -> list[PaperData]:
    """
    Search Semantic Scholar for highly-cited ML papers across multiple queries.

    - Filters to papers with >= min_citations citations
    - Only keeps papers that have an ArXiv ID (so we can link back to ArXiv)
    - Deduplicates across queries by arxiv_id
    - Returns papers sorted by citation count descending (most cited first)
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    seen: dict[str, dict] = {}  # arxiv_id → raw S2 item (deduplication)

    for q_idx, query in enumerate(queries, 1):
        logger.info(f"  [{q_idx:>2}/{len(queries)}] Query: \"{query}\"")
        fetched_this_query = 0
        offset = 0

        while fetched_this_query < max_per_query:
            limit = min(100, max_per_query - fetched_this_query)

            for attempt in range(3):
                try:
                    resp = requests.get(
                        f"{_S2_BASE}/graph/v1/paper/search",
                        params={
                            "query": query,
                            "fields": "paperId,externalIds,title,abstract,authors,year,citationCount,publicationDate",
                            "limit": limit,
                            "offset": offset,
                            "minCitationCount": min_citations,
                        },
                        headers=headers,
                        timeout=15,
                    )
                    if resp.status_code == 429:
                        wait = 60 * (attempt + 1)
                        logger.warning(f"    Rate limited — waiting {wait}s")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        logger.error(f"    Request failed after 3 attempts: {e}")
                        break
                    time.sleep(5)
            else:
                break

            items = resp.json().get("data", [])
            if not items:
                break  # No more results for this query

            new_this_page = 0
            for item in items:
                arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
                if arxiv_id and arxiv_id not in seen:
                    seen[arxiv_id] = item
                    new_this_page += 1

            fetched_this_query += len(items)
            offset += len(items)

            logger.info(
                f"    Page offset {offset - len(items):>5}: "
                f"{len(items)} results, {new_this_page} new unique  "
                f"(total unique so far: {len(seen):,})"
            )

            if len(items) < limit:
                break  # Last page

            time.sleep(1.1 if not api_key else 0.2)

        time.sleep(1.1 if not api_key else 0.2)

    # Sort by citation count descending — most cited first
    sorted_items = sorted(
        seen.values(),
        key=lambda x: x.get("citationCount") or 0,
        reverse=True,
    )

    logger.info(f"\n  Deduplication: {len(sorted_items):,} unique ArXiv papers across {len(queries)} queries")
    if sorted_items:
        top = sorted_items[0]
        bottom = sorted_items[-1]
        logger.info(f"  Citation range: {top.get('citationCount', 0):,} (highest) → {bottom.get('citationCount', 0):,} (lowest)")

    # Convert to PaperData
    papers = []
    for item in sorted_items:
        arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
        if not arxiv_id:
            continue

        pub_date = None
        if item.get("publicationDate"):
            try:
                pub_date = datetime.strptime(item["publicationDate"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass
        if pub_date is None and item.get("year"):
            pub_date = datetime(item["year"], 1, 1, tzinfo=timezone.utc)

        authors = [a.get("name", "") for a in (item.get("authors") or [])]

        papers.append(PaperData(
            arxiv_id=arxiv_id,
            title=(item.get("title") or "").strip(),
            abstract=(item.get("abstract") or "").strip(),
            authors=authors,
            categories=[],  # S2 doesn't return ArXiv categories
            published_date=pub_date,
            arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            citation_count=item.get("citationCount"),
        ))

    return papers


# ── Step 2: Save to PostgreSQL ───────────────────────────────────────────────

def save_to_postgres(papers: list[PaperData]) -> list[PaperData]:
    """
    Insert new papers into PostgreSQL. Skip already-known ones.
    Citation count is taken directly from S2 (already fetched in step 1).
    Returns only the truly new papers.
    """
    db = SessionLocal()
    try:
        all_ids = [p.arxiv_id for p in papers]
        existing_ids = {
            row[0]
            for row in db.query(Paper.arxiv_id).filter(Paper.arxiv_id.in_(all_ids)).all()
        }

        new_papers = []
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
                    citation_count=p.citation_count,  # None = not found on S2
                ))
                new_papers.append(p)

        db.commit()
        skipped = len(papers) - len(new_papers)
        logger.info(f"  Inserted {len(new_papers):,} new rows  ({skipped:,} already existed)")
        return new_papers
    finally:
        db.close()


# ── Step 3: Embed into ChromaDB ──────────────────────────────────────────────

def index_to_chromadb(papers: list[PaperData], em: EmbeddingManager) -> int:
    """Embed papers and upsert into ChromaDB in batches. Returns count indexed."""
    total = len(papers)
    indexed = 0
    total_batches = (total + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE

    for i in range(0, total, _EMBED_BATCH_SIZE):
        batch = papers[i : i + _EMBED_BATCH_SIZE]
        batch_num = i // _EMBED_BATCH_SIZE + 1

        texts = [f"{p.title}\n\n{p.abstract or ''}" for p in batch]
        embeddings = em.model.encode(texts, show_progress_bar=False, batch_size=32).tolist()

        em.collection.upsert(
            ids=[f"paper_{p.arxiv_id}" for p in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "type": "paper",
                    "paper_id": p.arxiv_id,
                    "title": p.title,
                    "url": p.arxiv_url or f"https://arxiv.org/abs/{p.arxiv_id}",
                    "published_date": str(p.published_date) if p.published_date else "",
                    "citation_count": p.citation_count or 0,
                }
                for p in batch
            ],
        )

        indexed += len(batch)
        pct = indexed / total * 100
        logger.info(
            f"  Embed batch {batch_num:>3}/{total_batches}:  "
            f"{indexed:>6,}/{total:,} papers  ({pct:.0f}%)"
        )

    return indexed


# ── Main ─────────────────────────────────────────────────────────────────────

def run_bulk_import(
    min_citations: int = 50,
    max_per_query: int = 1000,
) -> dict:
    start = datetime.now(timezone.utc)
    result = {
        "queries": len(SEARCH_QUERIES),
        "fetched_unique": 0,
        "new_in_postgres": 0,
        "indexed_in_chromadb": 0,
        "elapsed_s": 0.0,
        "error": None,
    }

    api_key = settings.SEMANTIC_SCHOLAR_API_KEY

    logger.info("=" * 65)
    logger.info("BULK IMPORT — Citation-ranked via Semantic Scholar")
    logger.info(f"  Queries:          {len(SEARCH_QUERIES)}")
    logger.info(f"  Min citations:    {min_citations:,}")
    logger.info(f"  Max per query:    {max_per_query:,}")
    logger.info(f"  Est. total:       ~{len(SEARCH_QUERIES) * max_per_query // 3:,} unique papers (after dedup)")
    logger.info(f"  API key:          {'yes (higher limits)' if api_key else 'no (public rate limit)'}")
    logger.info("=" * 65)

    try:
        # ── 1. Fetch from Semantic Scholar (citation counts included) ────────
        logger.info("\n[1/3] Fetching highly-cited papers from Semantic Scholar...")
        papers = fetch_from_semantic_scholar(
            queries=SEARCH_QUERIES,
            min_citations=min_citations,
            max_per_query=max_per_query,
            api_key=api_key,
        )
        result["fetched_unique"] = len(papers)
        logger.info(f"\n✓ [1/3] Fetched {len(papers):,} unique highly-cited papers")

        if not papers:
            logger.warning("No papers fetched — check Semantic Scholar API or network")
            return result

        # ── 2. Save to PostgreSQL ────────────────────────────────────────────
        logger.info("\n[2/3] Saving to PostgreSQL...")
        new_papers = save_to_postgres(papers)
        result["new_in_postgres"] = len(new_papers)
        logger.info(f"\n✓ [2/3] PostgreSQL: {len(new_papers):,} new papers saved")

        if not new_papers:
            logger.info("All papers already in DB — nothing new to embed")
            return result

        # ── 3. Embed and index into ChromaDB ─────────────────────────────────
        logger.info(f"\n[3/3] Embedding {len(new_papers):,} papers into ChromaDB...")
        em = EmbeddingManager()
        indexed = index_to_chromadb(new_papers, em)
        result["indexed_in_chromadb"] = indexed
        logger.info(f"\n✓ [3/3] ChromaDB: {indexed:,} papers indexed")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"\nBulk import FAILED: {e}", exc_info=True)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    result["elapsed_s"] = round(elapsed, 1)

    logger.info("\n" + "=" * 65)
    logger.info("BULK IMPORT SUMMARY")
    logger.info(f"  Search queries run:      {result['queries']:>8,}")
    logger.info(f"  Unique papers fetched:   {result['fetched_unique']:>8,}  (all have >= {min_citations} citations)")
    logger.info(f"  New in PostgreSQL:       {result['new_in_postgres']:>8,}")
    logger.info(f"  Indexed in ChromaDB:     {result['indexed_in_chromadb']:>8,}")
    logger.info(f"  Total time:              {elapsed:>8.0f}s  ({elapsed/60:.1f} min)")
    if result["error"]:
        logger.info(f"  Error:                   {result['error']}")
    logger.info("=" * 65)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk historical paper import via Semantic Scholar")
    parser.add_argument("--min-citations", type=int, default=50,
                        help="Minimum citation count to include a paper (default: 50)")
    parser.add_argument("--max-per-query", type=int, default=1000,
                        help="Max papers to fetch per search query (default: 1000)")
    args = parser.parse_args()

    run_bulk_import(min_citations=args.min_citations, max_per_query=args.max_per_query)
