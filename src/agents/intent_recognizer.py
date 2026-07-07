"""
Hybrid intent recognition: keyword rules → embedding similarity → LLM fallback.

Stage 1 (keyword): fast, deterministic, handles obvious cases
Stage 2 (embedding): semantic cosine similarity against labeled examples
Stage 3 (LLM): GPT-4o-mini classification for ambiguous messages
"""
import json
import logging
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
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for intent, keywords in KEYWORD_RULES.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count:
                scores[intent] = count

        if not scores:
            return None, 0.0

        best = max(scores, key=scores.get)
        # Scale: 1 match → ~0.5 conf, 2 matches → ~0.75, 3+ → ~0.9
        confidence = min(0.95, 0.4 + scores[best] * 0.25)
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
