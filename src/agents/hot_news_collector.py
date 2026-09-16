"""
HotNewsCollector — fetches What's Hot from two community sources:

  1. HuggingFace Papers (trending ML research)
     GET https://huggingface.co/api/papers?sort=trending&limit=N
     → community-voted paper hotness, no time parameter needed

  2. GitHub Trending repos (trending AI/ML/data tools, not just research code)
     GitHub Search API: repos pushed recently, sorted by stars
     Window auto-inferred from user's last visit:
       ≤ 1 day  → "daily"  (last day's stars)
       > 1 day  → "weekly" (last 7 days)

     Topic coverage is deliberately wider than ArXiv's: papers rarely cover
     the practical/engineering side of AI-adjacent roles (MLOps, data
     pipelines, agent frameworks), so GitHub is where that content actually
     lives. See _fetch_github_trending's topic list.

Results are cached in Redis for 6 hours, keyed by date AND github_window.
The paper/repo *content* is shared across all users (What's Hot is global,
not personalised), but the cache key still has to vary with the window,
since two viewers inferred into different windows must not silently reuse
each other's github_repos list. A date-only key let whoever happened to
populate the cache first decide the window everyone else saw for the next
6 hours, regardless of their own visit history.
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
CACHE_KEY = "whats_hot:{date}:{window}"


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

        Checks Redis cache first (6h TTL, keyed by date + this caller's
        inferred github_window (see module docstring for why the window is
        part of the key). If miss: fetches both sources, synthesizes HF
        digests, caches result.
        """
        gh_window = "daily" if user_last_visit_days <= 1 else "weekly"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_key = CACHE_KEY.format(date=today, window=gh_window)

        # ── Cache read ────────────────────────────────────────────────────────
        if redis_client:
            try:
                raw = redis_client.get(cache_key)
                if raw:
                    logger.info(f"What's Hot: cache hit (window={gh_window})")
                    return json.loads(raw)
            except Exception:
                pass

        # ── Fetch ─────────────────────────────────────────────────────────────
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

        # Topic filter: covers DL/LLM/agent research tooling plus the
        # practical AI-engineer / data-engineer / data-scientist side
        # (MLOps, data pipelines) that ArXiv's paper categories miss.
        # GitHub's unauthenticated Search API caps at 10 req/min. This list
        # is kept short enough that one collect() call (one request per
        # topic) stays well under that even under light concurrent misses.
        topics = [
            "machine-learning", "deep-learning", "llm",
            "ai-agents", "mlops", "data-engineering", "data-science",
        ]
        queries = [f"topic:{t}+pushed:>{cutoff}" for t in topics]

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
