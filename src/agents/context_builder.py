"""
ContextBuilder — GSSC pipeline for intelligent conversation history selection.

Inspired by Hello-Agents ch9 Context Engineering (Gather → Select → Structure → Compress).

Instead of blindly truncating to the last N messages, ContextBuilder:
  1. Embeds the current query
  2. Computes cosine similarity against all stored messages
  3. Returns the top-k most semantically relevant turns

This means a user asking about "attention mechanisms" will see their earlier
conversation about transformers, even if it was 30 messages ago, while irrelevant
greeting turns are dropped.

Falls back to recency-based selection (last k messages) when no embedding model
is available.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


class ContextBuilder:
    """
    Selects the most semantically relevant conversation turns for a query.

    Usage:
        builder = ContextBuilder(embedding_model=embedding_manager.model)
        relevant_history = builder.build(query, full_history, top_k=8)
        # Pass relevant_history to agent.run()
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def build(
        self,
        query: str,
        history: List[Dict],
        top_k: int = 8,
        always_include_last: int = 2,
    ) -> List[Dict]:
        """
        Return up to top_k messages most relevant to query.

        Args:
            query: The current user message.
            history: Full conversation history (OpenAI message dicts).
            top_k: Maximum messages to return.
            always_include_last: Always keep the N most recent turns regardless
                of score (preserves immediate conversational context).

        Returns:
            Selected messages in original chronological order.
        """
        if not history:
            return []

        if len(history) <= top_k:
            return history

        # Always keep the tail for immediate context
        tail = history[-always_include_last:] if always_include_last > 0 else []
        candidates = history[:-always_include_last] if always_include_last > 0 else history

        remaining_slots = top_k - len(tail)
        if remaining_slots <= 0:
            return tail

        if not candidates:
            return tail

        # Try semantic selection
        selected = self._select_by_relevance(query, candidates, remaining_slots)

        # Merge and restore chronological order
        tail_set = set(id(m) for m in tail)
        merged = [m for m in history if id(m) in set(id(s) for s in selected) or id(m) in tail_set]
        return merged

    def _select_by_relevance(
        self,
        query: str,
        candidates: List[Dict],
        k: int,
    ) -> List[Dict]:
        """
        Rank candidates by cosine similarity to the query embedding.
        Falls back to recency on embedding failure.
        """
        if not self.embedding_model or not hasattr(self.embedding_model, "encode"):
            logger.debug("ContextBuilder: no embedding model — using recency fallback")
            return candidates[-k:]

        try:
            texts = [m.get("content", "") or "" for m in candidates]
            # Encode query + all candidate messages in one batch
            all_texts = [query] + texts
            embeddings = self.embedding_model.encode(all_texts, show_progress_bar=False)
            query_emb = embeddings[0]
            msg_embs = embeddings[1:]

            scores = [_cosine_sim(query_emb, e) for e in msg_embs]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = sorted(ranked[:k])  # restore chronological order
            return [candidates[i] for i in top_indices]

        except Exception as e:
            logger.warning(f"ContextBuilder embedding failed: {e} — using recency fallback")
            return candidates[-k:]
