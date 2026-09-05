"""
AgentRouter: classifies intent and dispatches to the right specialized agent.

research_qa routing matrix:
  ┌──────────────────────────────┬──────────────────────────────┐
  │  Query type                  │  Agent                       │
  ├──────────────────────────────┼──────────────────────────────┤
  │  analytical (compare/tradeoff│  PlanAndSolveAgent           │
  │  complex research             │  DeepResearchAgent           │
  │  simple                       │  ReflectionAgent             │
  └──────────────────────────────┴──────────────────────────────┘

All three fall back gracefully to ResearchAgent if unavailable.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.intent_recognizer import IntentRecognizer, Intent
from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.document_agent import DocumentAgent
from src.agents.tools import SEARCH_WEB

logger = logging.getLogger(__name__)


# ── Query classifiers ──────────────────────────────────────────────────────────

_ANALYTICAL_KEYWORDS = {
    "compare", "comparison", "difference between", "differences between",
    "pros and cons", "trade-off", "trade off", "tradeoffs", "tradeoff",
    "when should i use", "vs", "versus", "advantages of", "disadvantages of",
    "benefits of", "limitations of", "better than", "worse than",
    "which is better", "should i use",
}

_COMPLEX_KEYWORDS = {
    "how does", "why does", "comprehensive", "overview of", "survey",
    "explain in detail", "walk me through", "what are the implications",
    "multiple", "several", "various", "relationship between",
}

_COMPLEX_LENGTH = 90        # long enough to *consider* the deep path
_VERY_LONG_LENGTH = 220     # long enough that multi-part intent is near-certain


def _contains_phrase(message: str, phrases) -> bool:
    """
    Whole-word phrase match.

    A bare substring test misfires badly on short keywords: "vs" matches
    "VSA", "vsync" and "revision", routing trivial lookups to the expensive
    analytical agent. \\b anchors the match to word boundaries.
    """
    lower = message.lower()
    return any(re.search(rf"\b{re.escape(p)}\b", lower) for p in phrases)


def _is_analytical_query(message: str) -> bool:
    """True for comparison, trade-off, and when-to-use questions."""
    return _contains_phrase(message, _ANALYTICAL_KEYWORDS)


def _is_complex_query(message: str) -> bool:
    """
    True for multi-faceted research questions.

    Length alone is not evidence of complexity — a long "summarise this and
    keep it short" request would otherwise trigger DeepResearchAgent, which
    costs a planner + up to 5 summarisers + a writer. Length is therefore only
    a supporting signal: it must be paired with an actual complexity keyword,
    or be long enough that a multi-part question is genuinely likely.
    """
    if _contains_phrase(message, _COMPLEX_KEYWORDS):
        return True
    # Long *and* multi-clause: several sentences or an explicit conjunction of asks.
    if len(message) >= _COMPLEX_LENGTH:
        clause_markers = message.count("?") + message.count(";") + message.count(" and ")
        return clause_markers >= 2 or len(message) >= _VERY_LONG_LENGTH
    return False


# ── GeneralAgent ───────────────────────────────────────────────────────────────

class GeneralAgent(BaseAgent):
    name = "GeneralAgent"
    system_prompt = """\
You are a helpful assistant for a machine learning research platform.

This platform helps researchers:
- Discover personalized paper and article recommendations
- Ask questions about ML concepts (RAG-powered Q&A)
- Manage an uploaded document knowledge base

You can search the web for current information when needed.
Answer general questions concisely. For capability questions, explain what the platform \
can do and guide the user toward the right feature."""
    tool_schemas = [SEARCH_WEB]


_INTENT_TO_AGENT: Dict[str, type] = {
    Intent.RECOMMENDATION: RecommendationAgent,
    Intent.DOCUMENT_MANAGEMENT: DocumentAgent,
    Intent.GENERAL_CHAT: GeneralAgent,
}


# ── Router ─────────────────────────────────────────────────────────────────────

class AgentRouter:
    """
    Singleton-friendly router. Lazily instantiates agents and shares the
    sentence-transformer model with EmbeddingManager to avoid loading it twice.
    """

    def __init__(self, embedding_model=None):
        self.recognizer = IntentRecognizer(embedding_model=embedding_model)
        self._agents: Dict[str, BaseAgent] = {}
        self._reflection_agent: Optional[BaseAgent] = None
        self._deep_agent: Optional[BaseAgent] = None
        self._plan_agent: Optional[BaseAgent] = None
        self._reflection_loaded = False
        self._deep_loaded = False
        self._plan_loaded = False

    # ── Agent lazy loaders ────────────────────────────────────────────────────

    def _get_agent(self, intent: str) -> BaseAgent:
        if intent not in self._agents:
            cls = _INTENT_TO_AGENT.get(intent, GeneralAgent)
            try:
                self._agents[intent] = cls()
            except ValueError as e:
                logger.warning(f"Could not initialise {cls.__name__}: {e}")
                self._agents[intent] = GeneralAgent()
        return self._agents[intent]

    def _get_research_agent(self) -> BaseAgent:
        if Intent.RESEARCH_QA not in self._agents:
            self._agents[Intent.RESEARCH_QA] = ResearchAgent()
        return self._agents[Intent.RESEARCH_QA]

    def _get_reflection_agent(self) -> BaseAgent:
        if not self._reflection_loaded:
            self._reflection_loaded = True
            try:
                from src.agents.reflection_agent import ReflectionAgent
                self._reflection_agent = ReflectionAgent()
                logger.info("ReflectionAgent loaded")
            except Exception as e:
                logger.warning(f"ReflectionAgent unavailable: {e}")
        return self._reflection_agent or self._get_research_agent()

    def _get_deep_agent(self) -> BaseAgent:
        if not self._deep_loaded:
            self._deep_loaded = True
            try:
                from src.agents.deep_research_agent import DeepResearchAgent
                self._deep_agent = DeepResearchAgent()
                logger.info("DeepResearchAgent loaded")
            except Exception as e:
                logger.warning(f"DeepResearchAgent unavailable: {e}")
        return self._deep_agent or self._get_research_agent()

    def _get_plan_agent(self) -> BaseAgent:
        if not self._plan_loaded:
            self._plan_loaded = True
            try:
                from src.agents.plan_and_solve_agent import PlanAndSolveAgent
                self._plan_agent = PlanAndSolveAgent()
                logger.info("PlanAndSolveAgent loaded")
            except Exception as e:
                logger.warning(f"PlanAndSolveAgent unavailable: {e}")
        return self._plan_agent or self._get_research_agent()

    def _select_research_agent(self, message: str, streaming: bool = False) -> BaseAgent:
        """Pick the right research_qa agent based on query characteristics."""
        if _is_analytical_query(message):
            return self._get_plan_agent()
        if _is_complex_query(message):
            return self._get_deep_agent()
        # Simple path: streaming uses plain ResearchAgent (lower latency);
        # non-streaming uses ReflectionAgent (quality improvement).
        return self._get_research_agent() if streaming else self._get_reflection_agent()

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        message: str,
        conversation_history: List[Dict],
        context: Dict[str, Any],
    ) -> Tuple[AgentResult, str, str, float, str]:
        """
        Classify intent, pick agent, run, return 5-tuple.

        Returns: (result, intent, recognition_method, confidence, agent_name)
        """
        intent, confidence, method = self.recognizer.recognize(message)
        logger.info(f"Routing → intent='{intent}', method='{method}', conf={confidence:.2f}")

        if intent == Intent.RESEARCH_QA:
            agent = self._select_research_agent(message, streaming=False)
        else:
            agent = self._get_agent(intent)

        result = agent.run(message, conversation_history, context)
        return result, intent, method, confidence, agent.name

    def route_stream(
        self,
        message: str,
    ) -> Tuple[BaseAgent, str, str, float]:
        """
        Return (agent, intent, method, confidence) without running the agent.
        Used by the streaming endpoint, which calls agent.stream() directly.
        """
        intent, confidence, method = self.recognizer.recognize(message)

        if intent == Intent.RESEARCH_QA:
            agent = self._select_research_agent(message, streaming=True)
        else:
            agent = self._get_agent(intent)

        return agent, intent, method, confidence
