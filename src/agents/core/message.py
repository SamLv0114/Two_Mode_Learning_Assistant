"""
Message — a typed conversation turn, inspired by Hello-Agents ch7 framework.

Richer than a raw dict: carries timestamp and arbitrary metadata so agents
can log which tool produced a message, attach confidence scores, etc.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> Dict:
        """Convert to the dict format the OpenAI API expects."""
        return {"role": self.role, "content": self.content}

    @staticmethod
    def system(content: str, **meta) -> "Message":
        return Message(role="system", content=content, metadata=meta)

    @staticmethod
    def user(content: str, **meta) -> "Message":
        return Message(role="user", content=content, metadata=meta)

    @staticmethod
    def assistant(content: str, **meta) -> "Message":
        return Message(role="assistant", content=content, metadata=meta)
