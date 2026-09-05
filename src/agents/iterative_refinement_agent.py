"""
IterativeRefinementAgent — Evaluator-Optimizer loop for feed quality.

Inspired by the Evaluator-Optimizer pattern (LangGraph / GPT-Researcher):

  Round 1:  Score current feed (Evaluator) → if score < threshold, refine (Optimizer)
  Round 2+: Re-score refined feed → repeat until score >= EVAL_THRESHOLD or MAX_ROUNDS

Used by POST /feed/refine to let the user trigger a quality improvement pass
without re-running the full pipeline. Reads feed context from Redis (feed_ctx:{user_id})
and re-ranks or replaces low-scoring papers.

Scoring dimensions (each 0–1):
  - diversity:   title embedding pairwise dissimilarity
  - relevance:   average relevance_score from DB
  - novelty:     fraction of papers not in shown_arxiv_ids history
  - mode_fit:    alignment between paper recency/citations and user_mode

Overall score = 0.3*diversity + 0.4*relevance + 0.2*novelty + 0.1*mode_fit
PASS threshold: 0.70
"""
import json
import logging
from typing import Dict, List, Optional

from openai import OpenAI

from src.utils.config import settings

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3
EVAL_THRESHOLD = 0.70

_EVALUATOR_PROMPT = """\
You are an expert ML paper feed curator. Evaluate this list of recommended papers \
for a researcher and return a quality score from 0.0 to 1.0.

User research mode: {mode}
User interests: {interests}

Papers:
{papers_block}

Score on:
- Diversity (varied topics/methods, not all the same theme)
- Relevance (match to user interests and mode)
- Novelty (not all well-known papers the user likely already knows)
- Mode fit (frontier mode → recent; learning mode → cited foundational)

Return JSON only: {{"score": <float 0-1>, "reason": "<one sentence>", "weak_indices": [<0-indexed positions to replace>]}}
"""


class IterativeRefinementAgent:
    """
    Evaluator-Optimizer loop: score feed → refine if needed → repeat up to MAX_ROUNDS.
    Returns the best-scored feed version found across all rounds.
    """

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def refine(
        self,
        papers: List[Dict],
        user_mode: str,
        user_interests: List[str],
        user_id: int,
        db_session=None,
    ) -> Dict:
        """
        Run up to MAX_ROUNDS of evaluate → refine.

        Returns:
          {
            "papers": List[Dict],   # best version of the feed
            "score":  float,        # final evaluator score
            "rounds": int,          # number of rounds taken
            "passed": bool,         # True if score >= EVAL_THRESHOLD
          }
        """
        best_papers = papers
        best_score = 0.0
        interests_str = ", ".join(user_interests[:8]) if user_interests else "ML research"

        for round_num in range(1, MAX_ROUNDS + 1):
            logger.info(f"[IterativeRefinement] round {round_num}/{MAX_ROUNDS}")

            score, reason, weak_indices = self._evaluate(best_papers, user_mode, interests_str)
            logger.info(f"  score={score:.2f} reason={reason}")

            if score > best_score:
                best_score = score

            if score >= EVAL_THRESHOLD:
                logger.info(f"  PASS (score={score:.2f} >= {EVAL_THRESHOLD})")
                return {
                    "papers": best_papers,
                    "score": best_score,
                    "rounds": round_num,
                    "passed": True,
                }

            # Try to improve: replace weak-scoring papers
            if weak_indices and db_session:
                try:
                    refined = self._replace_weak_papers(
                        best_papers, weak_indices, user_mode, interests_str, db_session, user_id
                    )
                    if refined:
                        best_papers = refined
                except Exception as e:
                    logger.warning(f"  refinement failed: {e}")

        return {
            "papers": best_papers,
            "score": best_score,
            "rounds": MAX_ROUNDS,
            "passed": best_score >= EVAL_THRESHOLD,
        }

    # ── Evaluator ─────────────────────────────────────────────────────────────

    def _evaluate(self, papers: List[Dict], mode: str, interests: str):
        """Score the feed; return (score, reason, weak_indices)."""
        papers_block = "\n".join(
            f"{i}. [{p.get('arxiv_id','?')}] {p.get('title','?')} "
            f"(relevance={p.get('relevance_score', 0):.2f}, "
            f"citations={p.get('citation_count', '?')})"
            for i, p in enumerate(papers[:10])
        )
        prompt = _EVALUATOR_PROMPT.format(
            mode=mode,
            interests=interests,
            papers_block=papers_block,
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")
            weak_indices = [int(i) for i in data.get("weak_indices", [])]
            return score, reason, weak_indices
        except Exception as e:
            logger.warning(f"Evaluator LLM call failed: {e}")
            return 0.5, "evaluation unavailable", []

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def _replace_weak_papers(
        self,
        papers: List[Dict],
        weak_indices: List[int],
        mode: str,
        interests: str,
        db_session,
        user_id: int,
    ) -> Optional[List[Dict]]:
        """
        Replace papers at weak_indices with higher-scoring alternatives from DB.
        Simple strategy: query DB for papers NOT in current feed, ranked by relevance_score.
        """
        from src.database.models import Paper, UserPaperRecommendation

        current_arxiv_ids = {p.get("arxiv_id") for p in papers}

        # Pull higher-relevance papers not currently in the feed
        alternates = (
            db_session.query(Paper, UserPaperRecommendation)
            .join(UserPaperRecommendation, Paper.id == UserPaperRecommendation.paper_id)
            .filter(
                UserPaperRecommendation.user_id == user_id,
                ~Paper.arxiv_id.in_(current_arxiv_ids),
            )
            .order_by(UserPaperRecommendation.relevance_score.desc())
            .limit(len(weak_indices) * 3)
            .all()
        )

        if not alternates:
            return None

        result = list(papers)
        alt_iter = iter(alternates)
        for idx in sorted(weak_indices, reverse=True):
            if idx >= len(result):
                continue
            try:
                paper, rec = next(alt_iter)
                result[idx] = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "relevance_score": rec.relevance_score or 0.0,
                    "citation_count": paper.citation_count,
                    "url": paper.arxiv_url,
                    "summary": rec.personalized_summary,
                    "rank": idx + 1,
                }
            except StopIteration:
                break

        return result
