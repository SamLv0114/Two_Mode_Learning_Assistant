"""
Multi-agent system for ResearchMate.

Entry point: AgentRouter.route(message, history, context) → AgentResult
"""
from src.agents.router import AgentRouter
from src.agents.base_agent import AgentResult
from src.agents.intent_recognizer import IntentRecognizer, Intent
from src.agents.memory import ConversationMemory

__all__ = ["AgentRouter", "AgentResult", "IntentRecognizer", "Intent", "ConversationMemory"]
