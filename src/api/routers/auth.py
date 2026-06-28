"""
Authentication endpoints
"""
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.database.models import User, UserInteraction, UserModelState
from src.api.deps import get_db_session, get_current_user
from src.api.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_token_expiry_seconds
)
from src.schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    Token,
    UserWithToken
)
from src.utils.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserWithToken, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db_session)
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

    # Create access token
    access_token = create_access_token(user.id, user.email)

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
        token_type="bearer"
    )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db_session)
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

    # Create access token
    access_token = create_access_token(user.id, user.email)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=get_token_expiry_seconds()
    )


@router.post("/login/json", response_model=Token)
async def login_json(
    credentials: UserLogin,
    db: Session = Depends(get_db_session)
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

    # Create access token
    access_token = create_access_token(user.id, user.email)

    return Token(
        access_token=access_token,
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
