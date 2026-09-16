"""
Authentication endpoints
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.database.models import User, UserInteraction, UserModelState
from src.api.deps import get_db_session, get_current_user
from src.api.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry_seconds
)
from src.schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    Token,
    UserWithToken,
    RefreshRequest
)
from src.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

REFRESH_JTI_PREFIX = "refresh_jti:"


def _get_redis():
    if not settings.REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def _issue_tokens(user: User, redis_client) -> tuple[str, str]:
    """
    Create an (access_token, refresh_token) pair and register the refresh
    token's jti in Redis so it can be redeemed exactly once (see
    create_refresh_token's docstring for why). If Redis isn't available,
    the refresh_token is still returned — the client can hold onto it —
    but /auth/refresh will reject it, since there is nowhere to record or
    check "has this jti been redeemed" without Redis. That is a deliberate
    fail-closed choice: silently accepting refresh tokens with no
    revocation tracking would be worse than refresh simply not working
    until Redis is back.
    """
    access_token = create_access_token(user.id, user.email)
    refresh_token, jti = create_refresh_token(user.id)
    if redis_client:
        try:
            redis_client.setex(
                f"{REFRESH_JTI_PREFIX}{jti}",
                settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
                str(user.id),
            )
        except Exception as e:
            logger.warning(f"Could not register refresh token jti: {e}")
    return access_token, refresh_token


@router.post("/register", response_model=UserWithToken, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db_session),
    redis_client=Depends(_get_redis)
):
    """
    Register a new user account

    - **email**: Valid email address (must be unique)
    - **password**: At least 8 characters with uppercase, lowercase, and digit
    - **full_name**: Optional display name
    - **interests**: Optional list of research interests
    - **focus_areas**: Optional list of focus areas (ML, NLP, CV, etc.)
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
    )

    # Set interests if provided
    if user_data.interests:
        user.set_interests_list(user_data.interests)
    else:
        # Default interests
        user.set_interests_list(settings.USER_INTERESTS)

    # Set focus areas if provided
    if user_data.focus_areas:
        user.set_focus_areas_list(user_data.focus_areas)

    db.add(user)
    db.commit()
    db.refresh(user)

    access_token, refresh_token = _issue_tokens(user, redis_client)

    return UserWithToken(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        interests=user.get_interests_list(),
        focus_areas=user.get_focus_areas_list(),
        is_active=user.is_active,
        created_at=user.created_at,
        interaction_count=0,
        model_trained=False,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db_session),
    redis_client=Depends(_get_redis)
):
    """
    Login with email and password to get an access token

    Uses OAuth2 password flow (form data with username/password fields).
    The username field should contain the email address.
    """
    # Find user by email
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    access_token, refresh_token = _issue_tokens(user, redis_client)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=get_token_expiry_seconds()
    )


@router.post("/login/json", response_model=Token)
async def login_json(
    credentials: UserLogin,
    db: Session = Depends(get_db_session),
    redis_client=Depends(_get_redis)
):
    """
    Login with JSON body (alternative to form-based login)

    Useful for JavaScript clients that prefer JSON.
    """
    # Find user by email
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Verify password
    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    access_token, refresh_token = _issue_tokens(user, redis_client)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=get_token_expiry_seconds()
    )


@router.post("/refresh", response_model=Token)
async def refresh(
    body: RefreshRequest,
    db: Session = Depends(get_db_session),
    redis_client=Depends(_get_redis)
):
    """
    Exchange a refresh token for a new access token + refresh token pair.

    One-time use: redeeming a refresh token immediately deletes its jti
    from Redis and issues a brand new refresh token (rotation), so a
    captured refresh token is only good until whichever of the legitimate
    client or the attacker uses it next — after that, the jti is gone and
    the other party's copy of the same token is permanently rejected, even
    though it hasn't reached its 7-day expiry yet.

    Requires Redis: there is nowhere else this endpoint could check "has
    this jti already been redeemed". Returns 401 (not 503) when Redis is
    unavailable — a general client shouldn't be able to tell "Redis is
    down" from "token invalid", so the caller's only correct response
    either way is the same: fall back to a normal login.
    """
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token",
    )

    payload = decode_token(body.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise invalid

    jti = payload.get("jti")
    user_id_str = payload.get("sub")
    if not jti or not user_id_str:
        raise invalid

    if not redis_client:
        raise invalid

    key = f"{REFRESH_JTI_PREFIX}{jti}"
    try:
        stored_user_id = redis_client.get(key)
        if stored_user_id is None:
            raise invalid
        redis_client.delete(key)  # one-time use: gone the instant it's redeemed
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Refresh token lookup failed: {e}")
        raise invalid

    try:
        user_id = int(user_id_str)
    except (TypeError, ValueError):
        raise invalid

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise invalid

    access_token, new_refresh_token = _issue_tokens(user, redis_client)

    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=get_token_expiry_seconds()
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Get the current user's profile

    Requires authentication via Bearer token.
    """
    # Get interaction count
    interaction_count = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id
    ).count()

    # Get model training status
    model_state = db.query(UserModelState).filter(
        UserModelState.user_id == current_user.id
    ).first()
    model_trained = model_state.is_trained if model_state else False

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        interests=current_user.get_interests_list(),
        focus_areas=current_user.get_focus_areas_list(),
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        interaction_count=interaction_count,
        model_trained=model_trained
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Update the current user's profile

    All fields are optional. Only provided fields will be updated.
    """
    # Update full name if provided
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name

    # Update interests if provided
    if user_update.interests is not None:
        current_user.set_interests_list(user_update.interests)

    # Update focus areas if provided
    if user_update.focus_areas is not None:
        current_user.set_focus_areas_list(user_update.focus_areas)

    # Update password if provided
    if user_update.password is not None:
        current_user.hashed_password = get_password_hash(user_update.password)

    db.commit()
    db.refresh(current_user)

    # Get interaction count
    interaction_count = db.query(UserInteraction).filter(
        UserInteraction.user_id == current_user.id
    ).count()

    # Get model training status
    model_state = db.query(UserModelState).filter(
        UserModelState.user_id == current_user.id
    ).first()
    model_trained = model_state.is_trained if model_state else False

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        interests=current_user.get_interests_list(),
        focus_areas=current_user.get_focus_areas_list(),
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        interaction_count=interaction_count,
        model_trained=model_trained
    )


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session)
):
    """
    Delete the current user's account

    This action is irreversible. All user data including interactions
    and model state will be permanently deleted.
    """
    db.delete(current_user)
    db.commit()
    return None
