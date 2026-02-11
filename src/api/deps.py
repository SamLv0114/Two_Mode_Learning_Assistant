"""
FastAPI dependency injection utilities
"""
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError

from src.database.models import get_db, User, UserInteraction
from src.api.security import decode_token
from src.models.embeddings import EmbeddingManager

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False  # Don't auto-raise, we handle it ourselves
)

# Singleton embedding manager
_embedding_manager: Optional[EmbeddingManager] = None


def get_db_session() -> Generator[Session, None, None]:
    """
    Database session dependency

    Yields a database session and ensures it's closed after use.
    """
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db_session)
) -> User:
    """
    Get the current authenticated user from JWT token

    Args:
        token: JWT token from Authorization header
        db: Database session

    Returns:
        User object

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except (TypeError, ValueError):
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db_session)
) -> Optional[User]:
    """
    Get the current user if authenticated, otherwise return None

    Useful for endpoints that work with or without authentication.
    """
    if not token:
        return None

    payload = decode_token(token)
    if payload is None:
        return None

    user_id_str = payload.get("sub")
    if user_id_str is None:
        return None

    try:
        user_id = int(user_id_str)
    except (TypeError, ValueError):
        return None

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        return None

    return user


async def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get the current user and verify they are a superuser

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def get_embedding_manager() -> EmbeddingManager:
    """
    Get the singleton embedding manager

    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


def get_user_interaction_count(user_id: int, db: Session) -> int:
    """
    Get the total number of interactions for a user

    Args:
        user_id: User ID
        db: Database session

    Returns:
        Number of interactions
    """
    return db.query(UserInteraction).filter(
        UserInteraction.user_id == user_id
    ).count()
