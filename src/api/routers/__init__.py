"""
API Routers
"""
from src.api.routers.auth import router as auth_router
from src.api.routers.feed import router as feed_router
from src.api.routers.interactions import router as interactions_router
from src.api.routers.qa import router as qa_router
from src.api.routers.chat import router as chat_router
from src.api.routers.whats_hot import router as whats_hot_router

__all__ = ["auth_router", "feed_router", "interactions_router", "qa_router", "chat_router", "whats_hot_router"]
