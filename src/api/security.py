"""
Security utilities for authentication and authorization
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from jose import jwt, JWTError
import bcrypt
from src.utils.config import settings


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    """Hash a password for storage"""
    salt = bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def create_access_token(
    subject: Union[str, int],
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token

    Args:
        subject: The user ID to encode in the token
        email: The user's email
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "sub": str(subject),
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, int],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token (longer-lived)

    Args:
        subject: The user ID to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT refresh token string
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token

    Args:
        token: The JWT token string

    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


def get_token_expiry_seconds() -> int:
    """Get the token expiration time in seconds"""
    return settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
