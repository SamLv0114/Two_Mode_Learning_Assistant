"""
AgentRouter: classifies user intent and dispatches to the right specialized agent.
"""
import logging
from typing import Dict, Any, List, Tuple

from src.agents.intent_recognizer import IntentRecognizer, Intent
from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.document_agent import DocumentAgent

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    """Fallback for greetings and meta-questions — no tools needed."""

    name = "GeneralAgent"
    system_prompt = """\
You are a helpful assistant for a machine learning research platform.

This platform helps researchers:
- Discover personalized paper and article recommendations
- Ask questions about ML concepts (RAG-powered Q&A)
- Manage an uploaded document knowledge base

Answer general questions concisely. For capability questions, explain what the platform \
can do and guide the user toward the right feature."""
    tool_schemas = []


_INTENT_TO_AGENT: Dict[str, type] = {
    Intent.RESEARCH_QA: ResearchAgent,
    Intent.RECOMMENDATION: RecommendationAgent,
    Intent.DOCUMENT_MANAGEMENT: DocumentAgent,
    Intent.GENERAL_CHAT: GeneralAgent,
}


class AgentRouter:
    """
    Singleton-friendly router.
    Lazily instantiates agents and shares the sentence-transformer model
    with the EmbeddingManager to avoid loading it twice.
    """

    def __init__(self, embedding_model=None):
        self.recognizer = IntentRecognizer(embedding_model=embedding_model)
        self._agents: Dict[str, BaseAgent] = {}

    def _get_agent(self, intent: str) -> BaseAgent:
        if intent not in self._agents:
            cls = _INTENT_TO_AGENT.get(intent, GeneralAgent)
            self._agents[intent] = cls()
        return self._agents[intent]

    def route(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Tuple[AgentResult, str, str, float]:
        """
        Classify intent and run the matching agent.

        Returns:
            (result, intent, recognition_method, confidence)
        """
        intent, confidence, method = self.recognizer.recognize(message)
        logger.info(
            f"Routing → intent='{intent}', method='{method}', conf={confidence:.2f}"
        )

        agent = self._get_agent(intent)
        result = agent.run(message, conversation_history, context)
        return result, intent, method, confidence
