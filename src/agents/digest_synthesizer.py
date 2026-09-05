"""
DigestSynthesizer — generic 2-3 sentence digest generator for ML content.

Replaces ArticleDigestSynthesizer with a broader interface that handles
HuggingFace papers, GitHub repos, or any title+content pair.
Parallel batch synthesis via ThreadPoolExecutor.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from src.utils.config import settings

logger = logging.getLogger(__name__)

_PROMPT = """\
You are writing a concise digest blurb for a CS/ML community newsletter.

Title: {title}
Source: {source}
Content:
{content}

Write exactly 2-3 sentences for a technical CS/ML audience:
1. What this is about (specific, not vague)
2. Why it matters or what is novel
3. (optional) One concrete takeaway

Rules: No filler like "This article discusses...". Start with the substance. Be informative and punchy.
"""


class DigestSynthesizer:
    """
    Generates concise digest summaries for any ML content item.
    Sources are preserved separately so the frontend can render citations.
    """

    MODEL = "gpt-4o-mini"
    MAX_CONTENT_CHARS = 1200

    def __init__(self):
        self._client = None
        if settings.OPENAI_API_KEY:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def synthesize_batch(self, items: List[Dict]) -> List[Dict]:
        """
        Add 'digest' key to each item dict in-place (parallel).

        Each item must have: 'title'. Optional: 'content', 'abstract', 'source', 'url'.
        Returns the same list with 'digest' populated (empty string on failure).
        """
        if not items:
            return items

        if not self._client:
            for item in items:
                item["digest"] = item.get("description") or ""
            return items

        def _process(item):
            item["digest"] = self._synthesize_one(item)
            return item

        with ThreadPoolExecutor(max_workers=min(len(items), 4)) as ex:
            futures = {ex.submit(_process, item): item for item in items}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Digest failed: {e}")

        return items

    def _synthesize_one(self, item: Dict) -> str:
        title = (item.get("title") or "Untitled")[:300]
        content = (
            item.get("abstract") or item.get("content") or item.get("description") or ""
        )[:self.MAX_CONTENT_CHARS]
        source = item.get("source") or item.get("url") or ""

        if not content:
            return ""

        prompt = _PROMPT.format(title=title, source=source, content=content)
        try:
            resp = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"DigestSynthesizer failed for '{title[:60]}': {e}")
            return ""
