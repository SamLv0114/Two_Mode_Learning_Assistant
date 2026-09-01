"""
Multi-agent system for ResearchMate.

Entry point: AgentRouter.route(message, history, context) → AgentResult
"""
from src.agents.router import AgentRouter
from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.intent_recognizer import IntentRecognizer, Intent
from src.agents.memory import ConversationMemory, UserFactMemory
from src.agents.critic_agent import CriticAgent, CriticResult
from src.agents.context_builder import ContextBuilder
from src.agents.tool_aware_agent import ToolAwareAgent, ToolTraceEntry
from src.agents.core import ToolRegistry, Message
from src.agents.web_article_agent import WebArticleAgent

__all__ = [
    # Routing
    "AgentRouter",
    "AgentResult",
    "BaseAgent",
    "IntentRecognizer",
    "Intent",
    # Memory
    "ConversationMemory",
    "UserFactMemory",
    # Evaluation
    "CriticAgent",
    "CriticResult",
    # Context
    "ContextBuilder",
    # Observability
    "ToolAwareAgent",
    "ToolTraceEntry",
    # Framework
    "ToolRegistry",
    "Message",
    # Feed
    "WebArticleAgent",
]
