"""
Personalised paper summaries for the feed ("Key Insight / Why Care / Relation to Interests").
"""
import logging
from typing import List

from src.utils.config import settings
from src.utils.llm_usage import model_for
from src.utils.tracing import openai_client

logger = logging.getLogger(__name__)

SYSTEM = "You are a helpful AI assistant specialized in deep learning, LLMs, AI agents, and research."


class Generator:
    """Writes the short summary shown on each feed paper."""

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.client = openai_client()

    def generate_summary(self, title: str, content: str, user_interests: List[str] = None) -> str:
        """The summary, or "" when the model call fails (the card then shows no summary)."""
        interests = ", ".join(user_interests or settings.USER_INTERESTS)
        prompt = f"""Summarize this paper for a reader interested in {interests}.

Title: {title}
Content: {content[:2000]}

Use EXACTLY this format (keep each section to 1-2 sentences, total under 100 words):

**Key Insight:** <one sentence>
**Why Care:** <1-2 sentences>
**Relation to Interests:** <1-2 sentences>

Use the listed interests only to explain a specific connection supported by the paper. Do not assume the reader is a student, researcher, or any other profession. Avoid generic relevance claims. Be concise and actionable. Do not add any other sections or headings."""
        try:
            response = self.client.chat.completions.create(
                model=model_for("feed_summary"), max_completion_tokens=200,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            # Used to return "Error generating response: ...", which was then stored and shown as the summary.
            logger.error("Summary generation failed for %r: %s", title[:80], exc)
            return ""
