"""
HotNewsCollector — fetches What's Hot from two community sources:

  1. HuggingFace Papers (trending ML research)
     GET https://huggingface.co/api/papers?sort=trending&limit=N
     → community-voted paper hotness, no time parameter needed

  2. GitHub Trending repos (trending AI/ML/data tools, not just research code)
     Primary: the github.com/trending pages (all languages + Python),
     ranked by stars GAINED in the window ("1,234 stars this week"), then
     filtered to AI-related repos by name/description keywords.
     Window auto-inferred from user's last visit:
       ≤ 1 day  → "daily"  (stars gained today)
       > 1 day  → "weekly" (stars gained this week)

     Ranking by gained stars is the whole point: the old Search-API query
     ("AI topic + pushed recently, sort by total stars") returned
     tensorflow/pytorch every time, since any giant gets a commit every day.

     Fallback (Trending unreachable, markup changed, or too few AI repos
     that day): Search API for AI-topic repos CREATED in the last 30 days,
     sorted by total stars — still "new and popular", never the old giants.
     Each repo carries trend_basis ("trending" | "new") so the UI can say
     which ranking it came from.

Results are cached in Redis for 6 hours, keyed by date AND github_window.
The paper/repo *content* is shared across all users (What's Hot is global,
not personalised), but the cache key still has to vary with the window,
since two viewers inferred into different windows must not silently reuse
each other's github_repos list. A date-only key let whoever happened to
populate the cache first decide the window everyone else saw for the next
6 hours, regardless of their own visit history.
"""
import html
import json
import logging
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/papers?sort=trending&limit={limit}"
CACHE_TTL = 6 * 3600  # 6 hours
# v2: repo entries gained stars_gained/trend_basis and a different ranking;
# a new prefix keeps pre-deploy payloads from being served for 6 more hours.
CACHE_KEY = "whats_hot:v2:{date}:{window}"

GH_TRENDING_PAGES = (
    "https://github.com/trending?since={since}",
    "https://github.com/trending/python?since={since}",
)
GH_NEW_REPO_DAYS = 30
# Trending is every language and every subject; keep the AI/ML/data ones.
# Word-boundary matched on "owner/name description" (case-insensitive).
AI_KEYWORDS = re.compile(
    r"\b(ai|llms?|gpts?|agents?|agentic|rag|mcp|ml|mlops|nlp|"
    r"machine[- ]learning|deep[- ]learning|neural|transformers?|diffusion|"
    r"models?|inference|embeddings?|vector|fine[- ]?tun\w*|lora|"
    r"multimodal|vision|speech|tts|asr|chatbots?|copilot|prompts?|"
    r"openai|claude|gemini|llama|qwen|deepseek|codex|langchain|langgraph|"
    r"pytorch|tensorflow|jax|cuda|reinforcement|robotics?|dataset|"
    r"data[- ](?:science|engineering|pipelines?))\b",
    re.I,
)


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
            except Exception as exc:
                logger.warning("What's Hot cache read failed: %s", exc)

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
            except Exception as exc:
                logger.warning("What's Hot cache write failed: %s", exc)

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
        since = "daily" if window == "daily" else "weekly"
        repos = self._scrape_trending(since)
        ai = [r for r in repos if AI_KEYWORDS.search(f"{r['full_name']} {r['description']}")]
        ai.sort(key=lambda r: r["stars_gained"] or 0, reverse=True)
        picked = ai[:limit]
        if len(picked) < limit:
            known = {r["full_name"] for r in picked}
            for repo in self._search_new_repos(limit * 2):
                if len(picked) >= limit:
                    break
                if repo["full_name"] not in known:
                    picked.append(repo)
                    known.add(repo["full_name"])
        logger.info(
            f"GitHub trending ({since}): {len(picked)} repos "
            f"({sum(r['trend_basis'] == 'trending' for r in picked)} from Trending, "
            f"{len(repos)} scraped, {len(ai)} AI-related)"
        )
        return picked

    def _scrape_trending(self, since: str) -> List[Dict]:
        """Every repo on the Trending pages, deduped. [] if GitHub is unreachable."""
        seen: Dict[str, Dict] = {}
        for page in GH_TRENDING_PAGES:
            url = page.format(since=since)
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ResearchMate/1.0)",
                    "Accept": "text/html",
                })
                with urllib.request.urlopen(req, timeout=8) as resp:
                    body = resp.read().decode("utf-8", "replace")
            except Exception as e:
                logger.warning(f"GitHub Trending fetch failed for {url}: {e}")
                continue
            for repo in parse_trending_html(body):
                seen.setdefault(repo["full_name"], repo)
        return list(seen.values())

    def _search_new_repos(self, limit: int) -> List[Dict]:
        """Fallback: AI-topic repos created recently, most-starred first."""
        from src.utils.config import settings

        cutoff = (datetime.now(timezone.utc) - timedelta(days=GH_NEW_REPO_DAYS)).strftime("%Y-%m-%d")
        # Unauthenticated Search allows 10 req/min — four topics stays well
        # under that even with a few concurrent cache misses.
        topics = ["llm", "ai-agents", "machine-learning", "deep-learning"]
        headers = {
            "User-Agent": "ResearchMate/1.0",
            "Accept": "application/vnd.github.v3+json",
        }
        if settings.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

        seen: Dict[str, Dict] = {}
        for topic in topics:
            url = (
                "https://api.github.com/search/repositories"
                f"?q=topic:{topic}+created:>{cutoff}&sort=stars&order=desc&per_page={limit}"
            )
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as e:
                logger.debug(f"GitHub search failed for topic '{topic}': {e}")
                continue
            for item in data.get("items", []):
                full_name = item.get("full_name", "")
                if full_name and full_name not in seen:
                    seen[full_name] = {
                        "name": item.get("name", ""),
                        "full_name": full_name,
                        "description": item.get("description") or "",
                        "stars": item.get("stargazers_count", 0),
                        "stars_gained": None,
                        "language": item.get("language") or "",
                        "url": item.get("html_url", ""),
                        "source": "GitHub",
                        "trend_basis": "new",
                    }
        return sorted(seen.values(), key=lambda r: r["stars"], reverse=True)[:limit]


_ARTICLE = re.compile(r'<article class="Box-row">(.*?)</article>', re.S)
_REPO_LINK = re.compile(r'<h2[^>]*>\s*<a[^>]*href="/([\w.-]+/[\w.-]+)"', re.S)
_DESCRIPTION = re.compile(r'<p class="col-9[^"]*">(.*?)</p>', re.S)
_LANGUAGE = re.compile(r'itemprop="programmingLanguage">([^<]+)<')
_TOTAL_STARS = re.compile(r'href="/[\w.-]+/[\w.-]+/stargazers"[^>]*>.*?([\d,]+)\s*</a>', re.S)
_GAINED = re.compile(r'([\d,]+)\s+stars\s+(?:today|this week|this month)')
_TAGS = re.compile(r"<[^>]+>")


def _count(text: str) -> int:
    return int(text.replace(",", ""))


def parse_trending_html(body: str) -> List[Dict]:
    """Parse github.com/trending markup. Skips any row it can't read rather
    than failing the page; an unrecognised layout simply yields [] and the
    caller falls back to the Search API."""
    repos = []
    for article in _ARTICLE.findall(body):
        link = _REPO_LINK.search(article)
        if not link:
            continue
        full_name = link.group(1)
        description = _DESCRIPTION.search(article)
        language = _LANGUAGE.search(article)
        total = _TOTAL_STARS.search(article)
        gained = _GAINED.search(article)
        repos.append({
            "name": full_name.split("/", 1)[1],
            "full_name": full_name,
            "description": html.unescape(_TAGS.sub("", description.group(1))).strip() if description else "",
            "stars": _count(total.group(1)) if total else 0,
            "stars_gained": _count(gained.group(1)) if gained else None,
            "language": language.group(1).strip() if language else "",
            "url": f"https://github.com/{full_name}",
            "source": "GitHub",
            "trend_basis": "trending",
        })
    return repos
