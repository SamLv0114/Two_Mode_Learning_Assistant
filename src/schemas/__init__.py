"""
Pydantic schemas for API request/response validation
"""
from src.schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    Token,
    TokenPayload,
)
from src.schemas.feed import (
    FeedRequest,
    FeedResponse,
    PaperResponse,
    ArticleResponse,
)
from src.schemas.interaction import (
    InteractionCreate,
    InteractionResponse,
    InteractionStats,
)

__all__ = [
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "UserUpdate",
    "Token",
    "TokenPayload",
    "FeedRequest",
    "FeedResponse",
    "PaperResponse",
    "ArticleResponse",
    "InteractionCreate",
    "InteractionResponse",
    "InteractionStats",
]
