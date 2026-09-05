"""
IterativeRefinementAgent — Evaluator-Optimizer loop for feed quality.

Backs POST /feed/refine: improve an existing feed without re-running the pipeline.

  Round 1:  an LLM scores the feed; below threshold, weak entries are replaced
  Round 2+: re-score the revised feed, up to MAX_ROUNDS

Scoring is a single gpt-4o-mini judgement over the whole list, not a computed
formula — the model is asked to weigh diversity, relevance, novelty and fit with
the reader's profile, and returns one score plus the positions it considers weak.
Replacements come from a candidate_provider supplied by the caller, which goes
back to the corpus; an earlier version searched the user's own recommendation
rows, which are exactly the feed being refined, so it never found anything and
the optimizer half silently did nothing.
"""
import json
import logging
from typing import Callable, Dict, List, Optional

from openai import OpenAI

from src.utils.config import settings

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3
EVAL_THRESHOLD = 0.70

_EVALUATOR_PROMPT = """\
You are an expert ML paper feed curator. Evaluate this list of recommended papers \
for a researcher and return a quality score from 0.0 to 1.0.

Reading profile: {mode}
User interests: {interests}

Papers:
{papers_block}

Score on:
- Diversity (varied topics/methods, not all the same theme)
- Relevance (match to the stated interests)
- Novelty (not all well-known papers the reader likely already knows)
- Profile fit (does the mix of established and recent work match the reading profile above)

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
        candidate_provider: Optional[Callable[[set, int], List[Dict]]] = None,
    ) -> Dict:
        """
        Run up to MAX_ROUNDS of evaluate -> replace weak entries -> re-evaluate.

        candidate_provider(exclude_arxiv_ids, limit) supplies replacements. Without
        one the loop can still score, but has nothing to swap in, so it returns
        after the first evaluation rather than burning two more LLM calls to
        re-score an unchanged list.

        Returns the highest-scoring version seen, with the score that version
        actually earned. Tracking those separately is the point: an earlier version
        updated the paper list on every round but only raised the score when it
        improved, so a refinement that made things worse was returned alongside the
        better round's score.
        """
        interests_str = ", ".join(user_interests[:8]) if user_interests else "ML research"

        current = list(papers)
        best_papers, best_score = list(papers), -1.0
        rounds_run = 0

        for round_num in range(1, MAX_ROUNDS + 1):
            rounds_run = round_num
            logger.info(f"[IterativeRefinement] round {round_num}/{MAX_ROUNDS}")

            score, reason, weak_indices = self._evaluate(current, user_mode, interests_str)
            logger.info(f"  score={score:.2f} reason={reason}")

            # Keep the list and the score it earned together.
            if score > best_score:
                best_score, best_papers = score, list(current)

            if score >= EVAL_THRESHOLD:
                logger.info(f"  PASS (score={score:.2f} >= {EVAL_THRESHOLD})")
                break

            if not weak_indices:
                logger.info("  evaluator flagged nothing to replace — stopping")
                break

            if candidate_provider is None:
                logger.info("  no candidate provider — cannot replace, stopping")
                break

            try:
                revised = self._replace_weak_papers(
                    current, weak_indices, candidate_provider
                )
            except Exception as e:
                logger.warning(f"  refinement failed: {e}")
                break

            if not revised:
                logger.info("  no replacement candidates available — stopping")
                break
            current = revised

        return {
            "papers": best_papers,
            "score": max(best_score, 0.0),
            "rounds": rounds_run,
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
        candidate_provider: Callable[[set, int], List[Dict]],
    ) -> Optional[List[Dict]]:
        """
        Swap the flagged positions for fresh candidates from the corpus.

        Excludes everything currently shown so a replacement is never a paper the
        reader is already looking at. Returns None when nothing could be replaced,
        which the caller treats as a signal to stop rather than loop pointlessly.
        """
        valid = sorted({i for i in weak_indices if 0 <= i < len(papers)})
        if not valid:
            return None

        current_ids = {p.get("arxiv_id") for p in papers if p.get("arxiv_id")}
        candidates = candidate_provider(current_ids, len(valid) * 2) or []
        # The provider excludes by id, but guard here too — it is a callback and
        # this class cannot assume it honoured the contract.
        candidates = [c for c in candidates if c.get("arxiv_id") not in current_ids]
        if not candidates:
            return None

        result = list(papers)
        for idx, cand in zip(valid, candidates):
            result[idx] = {**cand, "rank": idx + 1}

        replaced = min(len(valid), len(candidates))
        logger.info(f"  replaced {replaced} of {len(valid)} weak papers")
        return result
