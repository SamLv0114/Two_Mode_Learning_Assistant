"""
Hybrid intent recognition: keyword rules → embedding similarity → LLM fallback.

Stage 1 (keyword): fast, deterministic, handles obvious cases
Stage 2 (embedding): semantic cosine similarity against labeled examples
Stage 3 (LLM): GPT-4o-mini classification for ambiguous messages
"""
import json
import logging
import re
from enum import Enum
from typing import Tuple, List, Dict, Optional

import numpy as np

from src.utils.config import settings

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    RESEARCH_QA = "research_qa"
    RECOMMENDATION = "recommendation"
    DOCUMENT_MANAGEMENT = "document_management"
    GENERAL_CHAT = "general_chat"


# Confidence returned when two intents both have keyword evidence and the
# leader is not strictly dominant: deliberately below the 0.6 gate in
# recognize(), so the decision falls through to the semantic stages.
CONTESTED_CONFIDENCE = 0.45

# Stage 1: keyword rules
KEYWORD_RULES: Dict[str, List[str]] = {
    Intent.RECOMMENDATION: [
        "recommend", "suggest", "feed", "what should i read", "trending",
        "show me papers", "latest papers", "new papers", "find papers",
        "discover papers", "what's popular", "popular papers", "top papers",
    ],
    Intent.DOCUMENT_MANAGEMENT: [
        "my documents", "my files", "uploaded", "knowledge base",
        "what do i have", "list documents", "list files", "delete document",
        "add document", "my uploads", "how many documents",
    ],
    Intent.RESEARCH_QA: [
        "explain", "what is", "how does", "what are", "define", "describe",
        "difference between", "compare", "tell me about", "summarize",
        "help me understand", "elaborate", "walk me through", "overview of",
        "what does", "break down", "deep dive",
    ],
    Intent.GENERAL_CHAT: [
        "hello", "hi there", "hey", "thanks", "thank you", "bye",
        "goodbye", "how are you", "what can you do", "help me",
    ],
}

# Stage 2: labeled examples for embedding similarity
INTENT_EXAMPLES: Dict[str, List[str]] = {
    Intent.RESEARCH_QA: [
        "explain the transformer architecture",
        "what is the attention mechanism in neural networks",
        "how does BERT work",
        "what is the difference between GPT and BERT",
        "can you summarize the diffusion models paper",
        "help me understand reinforcement learning from human feedback",
        "what are the key ideas in contrastive learning",
        "walk me through how LoRA fine-tuning works",
        "break down the concept of self-supervised learning",
        "give me an overview of mixture of experts",
    ],
    Intent.RECOMMENDATION: [
        "suggest papers on reinforcement learning",
        "what should I read today",
        "generate my personalized feed",
        "show me the latest papers on transformers",
        "recommend something on computer vision",
        "what are the trending topics in NLP this week",
        "find me papers about diffusion models",
        "any good articles on LLM fine-tuning",
    ],
    Intent.DOCUMENT_MANAGEMENT: [
        "what documents do I have in my knowledge base",
        "list my uploaded files",
        "search my documents for information about attention",
        "how many documents have I uploaded",
        "find information in my notes about RLHF",
        "do I have any documents on transformers",
    ],
    Intent.GENERAL_CHAT: [
        "hello there",
        "what can you help me with",
        "thanks for the help",
        "how do I use this system",
        "what are your capabilities",
        "good morning",
        "you're really helpful",
    ],
}


class IntentRecognizer:
    """
    Three-stage hybrid intent classifier.
    Shares the sentence-transformer model with EmbeddingManager when passed in.
    """

    def __init__(self, embedding_model=None):
        self._model = embedding_model
        self._example_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._openai_client = None
        if settings.OPENAI_API_KEY:
            import openai
            self._openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    # ── Model access ──────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _build_example_embeddings(self) -> None:
        """Pre-compute and cache embeddings for all intent examples."""
        if self._example_embeddings is not None:
            return
        model = self._get_model()
        self._example_embeddings = {}
        for intent, examples in INTENT_EXAMPLES.items():
            embs = model.encode(examples, convert_to_numpy=True)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
            self._example_embeddings[intent] = embs / norms

    # ── Stage 1: keyword matching ─────────────────────────────────────────────

    def _keyword_match(self, text: str) -> Tuple[Optional[str], float]:
        """
        Weighted keyword vote, deferring when two intents both have evidence.

        Two things a plain match count got wrong, both measured on
        evaluation/datasets/routing.jsonl:

        1. Every keyword counted the same, so the single generic word "feed"
           in "how does a feed-forward network work" outvoted nothing and
           still won on a tie. Longer phrases are far more decisive than
           single words, so a match is now worth its word count.

        2. A keyword from one intent appearing inside a question *about*
           another intent ("explain what a knowledge base is in RAG systems")
           was treated as confidently as an uncontested match. When a second
           intent also has lexical evidence and the leader is not strictly
           dominant, this stage now returns a low confidence on purpose so the
           decision falls through to the embedding/LLM stages, which read the
           sentence rather than spot words in it.

        Word boundaries matter here for the same reason they do in
        router.py's analytical check: a bare substring test lets "hey" fire on
        "they" and "vs" on "VSA".
        """
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for intent, keywords in KEYWORD_RULES.items():
            weight = sum(
                len(kw.split())
                for kw in keywords
                if re.search(rf"\b{re.escape(kw)}\b", text_lower)
            )
            if weight:
                scores[intent] = weight

        if not scores:
            return None, 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_weight = ranked[0]
        runner_weight = ranked[1][1] if len(ranked) > 1 else 0

        # Contested and not strictly dominant -> hand it to the next stage.
        if runner_weight and best_weight <= 2 * runner_weight:
            return best, CONTESTED_CONFIDENCE

        # 1 word → 0.65, 2 → 0.80, 3 → 0.90, 4+ → 0.95
        confidence = min(0.95, 0.5 + best_weight * 0.15)
        return best, confidence

    # ── Stage 2: embedding cosine similarity ──────────────────────────────────

    def _embedding_match(self, text: str) -> Tuple[str, float]:
        self._build_example_embeddings()
        model = self._get_model()
        query_emb = model.encode([text], convert_to_numpy=True)[0]
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        best_intent = Intent.GENERAL_CHAT
        best_score = 0.0
        for intent, normed_embs in self._example_embeddings.items():
            sims = normed_embs @ query_emb
            score = float(sims.max())
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent, best_score

    # ── Stage 3: LLM classification ───────────────────────────────────────────

    def _llm_classify(self, text: str) -> Tuple[str, float]:
        if not self._openai_client:
            return Intent.GENERAL_CHAT, 0.5

        prompt = (
            'Classify this user message into exactly one intent.\n\n'
            'Intents:\n'
            '- research_qa: explain/define/compare concepts, summarize papers\n'
            '- recommendation: wants paper/article suggestions or a feed\n'
            '- document_management: asking about uploaded documents\n'
            '- general_chat: greetings, meta-questions, off-topic\n\n'
            f'Message: "{text}"\n\n'
            'Reply with JSON only: {"intent": "<category>", "confidence": <0.0-1.0>}'
        )
        try:
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=60,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            intent = data.get("intent", Intent.GENERAL_CHAT)
            confidence = float(data.get("confidence", 0.5))
            valid = {i.value for i in Intent}
            if intent not in valid:
                intent = Intent.GENERAL_CHAT
            return intent, confidence
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            return Intent.GENERAL_CHAT, 0.5

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize(self, text: str) -> Tuple[str, float, str]:
        """
        Classify the intent of a user message.

        Returns:
            (intent, confidence, method_used)
            method_used is one of: "keyword", "embedding", "llm"
        """
        # Stage 1
        kw_intent, kw_conf = self._keyword_match(text)
        if kw_conf >= 0.6:
            logger.debug(f"Intent '{kw_intent}' via keywords (conf={kw_conf:.2f})")
            return kw_intent, kw_conf, "keyword"

        # Stage 2
        emb_intent, emb_score = self._embedding_match(text)
        if emb_score >= 0.65:
            logger.debug(f"Intent '{emb_intent}' via embeddings (score={emb_score:.2f})")
            return emb_intent, emb_score, "embedding"

        # Stage 3
        llm_intent, llm_conf = self._llm_classify(text)
        logger.debug(f"Intent '{llm_intent}' via LLM (conf={llm_conf:.2f})")
        return llm_intent, llm_conf, "llm"
