"""
HotNewsCollector — fetches What's Hot from two community sources:

  1. HuggingFace Papers (trending ML research)
     GET https://huggingface.co/api/papers?sort=trending&limit=N
     → community-voted paper hotness, no time parameter needed

  2. GitHub Trending repos (trending ML tools/libraries)
     GitHub Search API: repos pushed recently, sorted by stars
     Window auto-inferred from user's last visit:
       ≤ 1 day  → "daily"  (last day's stars)
       > 1 day  → "weekly" (last 7 days)

Results are cached in Redis for 6 hours (shared across all users —
What's Hot is global, not personalised).
"""
import json
import logging
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/papers?sort=trending&limit={limit}"
GH_API = (
    "https://api.github.com/search/repositories"
    "?q=topic:machine-learning+pushed:>{date}"
    "&sort=stars&order=desc&per_page={limit}"
)
CACHE_TTL = 6 * 3600  # 6 hours
CACHE_KEY = "whats_hot:{date}"


class HotNewsCollector:

    def collect(
        self,
        user_last_visit_days: int = 7,
        redis_client=None,
        hf_limit: int = 6,
        gh_limit: int = 5,
    ) -> Dict:
        """
        Return combined What's Hot payload.

        Checks Redis cache first (6h TTL, date-keyed).
        If miss: fetches both sources, synthesizes HF digests, caches result.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_key = CACHE_KEY.format(date=today)

        # ── Cache read ────────────────────────────────────────────────────────
        if redis_client:
            try:
                raw = redis_client.get(cache_key)
                if raw:
                    logger.info("What's Hot: cache hit")
                    return json.loads(raw)
            except Exception:
                pass

        # ── Fetch ─────────────────────────────────────────────────────────────
        gh_window = "daily" if user_last_visit_days <= 1 else "weekly"
        logger.info(f"What's Hot: fetching (gh_window={gh_window})")

        hf_papers = self._fetch_hf_trending(limit=hf_limit)
        gh_repos = self._fetch_github_trending(window=gh_window, limit=gh_limit)

        # ── Digest synthesis — only for papers HF gave no ai_summary ──────────
        needs_digest = [p for p in hf_papers if not p.get("digest")]
        if needs_digest:
            try:
                from src.agents.digest_synthesizer import DigestSynthesizer
                DigestSynthesizer().synthesize_batch(needs_digest)
            except Exception as e:
                logger.warning(f"Digest synthesis skipped: {e}")
            for p in needs_digest:
                if not p.get("digest"):
                    p["digest"] = (p.get("abstract") or "")[:200]

        result = {
            "hf_papers": hf_papers,
            "github_repos": gh_repos,
            "github_window": gh_window,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        # ── Cache write ───────────────────────────────────────────────────────
        if redis_client:
            try:
                redis_client.setex(cache_key, CACHE_TTL, json.dumps(result))
            except Exception:
                pass

        return result

    # ── HuggingFace ───────────────────────────────────────────────────────────

    def _fetch_hf_trending(self, limit: int = 6) -> List[Dict]:
        try:
            url = HF_API.format(limit=limit)
            req = urllib.request.Request(
                url, headers={"User-Agent": "LearningAssistant/1.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            papers = []
            for item in data:
                arxiv_id = item.get("id", "")
                if not arxiv_id:
                    continue
                # HF already ships a one-line ai_summary — reuse it and skip the
                # LLM call entirely. DigestSynthesizer only fills the gaps.
                papers.append({
                    "id": arxiv_id,
                    "title": item.get("title", ""),
                    "abstract": (item.get("summary") or item.get("abstract") or "")[:800],
                    "authors": [a.get("name", "") for a in item.get("authors", [])][:4],
                    "upvotes": item.get("upvotes", 0),
                    "url": f"https://huggingface.co/papers/{arxiv_id}",
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "source": "HuggingFace",
                    "digest": (item.get("ai_summary") or "").strip(),
                })
            logger.info(f"HuggingFace trending: {len(papers)} papers")
            return papers
        except Exception as e:
            logger.warning(f"HF trending fetch failed: {e}")
            return []

    # ── GitHub ────────────────────────────────────────────────────────────────

    def _fetch_github_trending(self, window: str = "weekly", limit: int = 5) -> List[Dict]:
        days = 1 if window == "daily" else 7
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        # Topic filter: machine-learning OR deep-learning OR llm
        queries = [
            f"topic:machine-learning+pushed:>{cutoff}",
            f"topic:deep-learning+pushed:>{cutoff}",
            f"topic:llm+pushed:>{cutoff}",
        ]

        seen: Dict[str, Dict] = {}
        for q in queries:
            if len(seen) >= limit * 2:
                break
            try:
                url = (
                    f"https://api.github.com/search/repositories"
                    f"?q={q}&sort=stars&order=desc&per_page={limit}"
                )
                req = urllib.request.Request(url, headers={
                    "User-Agent": "LearningAssistant/1.0",
                    "Accept": "application/vnd.github.v3+json",
                })
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                for item in data.get("items", []):
                    full_name = item.get("full_name", "")
                    if full_name and full_name not in seen:
                        seen[full_name] = {
                            "name": item.get("name", ""),
                            "full_name": full_name,
                            "description": item.get("description") or "",
                            "stars": item.get("stargazers_count", 0),
                            "language": item.get("language") or "",
                            "url": item.get("html_url", ""),
                            "source": "GitHub",
                        }
            except Exception as e:
                logger.debug(f"GitHub fetch failed for query '{q}': {e}")

        repos = sorted(seen.values(), key=lambda r: r["stars"], reverse=True)[:limit]
        logger.info(f"GitHub trending ({window}): {len(repos)} repos")
        return repos
