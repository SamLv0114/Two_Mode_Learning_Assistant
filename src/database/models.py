"""
Database models for storing papers, articles, users, and interactions
"""
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime,
    Float, Boolean, ForeignKey, LargeBinary, text, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
from typing import Optional
from src.utils.config import settings

Base = declarative_base()


class User(Base):
    """User model for multi-user support"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)

    # User preferences (moved from global settings)
    interests = Column(Text, nullable=True)  # JSON array of interest strings
    focus_areas = Column(String(255), nullable=True)  # comma-separated: "ML,NLP,CV"

    # Account status
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    interactions = relationship("UserInteraction", back_populates="user", cascade="all, delete-orphan")
    model_state = relationship("UserModelState", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"

    def get_interests_list(self) -> list:
        """Parse interests JSON string to list"""
        if not self.interests:
            return []
        import json
        try:
            return json.loads(self.interests)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_interests_list(self, interests: list):
        """Set interests from list to JSON string"""
        import json
        self.interests = json.dumps(interests)

    def get_focus_areas_list(self) -> list:
        """Parse focus_areas comma-separated string to list"""
        if not self.focus_areas:
            return []
        return [area.strip() for area in self.focus_areas.split(",") if area.strip()]

    def set_focus_areas_list(self, areas: list):
        """Set focus_areas from list to comma-separated string"""
        self.focus_areas = ",".join(areas)


class UserModelState(Base):
    """Store per-user ML model and heuristic weights"""
    __tablename__ = "user_model_states"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)

    # Serialized model data
    model_blob = Column(LargeBinary, nullable=True)  # Pickled ML model
    heuristic_weights_blob = Column(LargeBinary, nullable=True)  # Pickled weights
    model_type = Column(String(50), nullable=True)  # "ltr" or "regressor"

    # Training state
    is_trained = Column(Boolean, default=False)
    interaction_count_at_training = Column(Integer, default=0)
    last_trained_at = Column(DateTime, nullable=True)

    # Metrics from last training
    train_ndcg = Column(Float, nullable=True)
    train_mrr = Column(Float, nullable=True)
    val_ndcg = Column(Float, nullable=True)
    val_mrr = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationship
    user = relationship("User", back_populates="model_state")

    def __repr__(self):
        return f"<UserModelState(user_id={self.user_id}, is_trained={self.is_trained})>"


class Paper(Base):
    """ArXiv paper model"""
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    arxiv_id = Column(String(50), unique=True, index=True)
    title = Column(String(500), index=True)
    authors = Column(Text)
    abstract = Column(Text)
    categories = Column(String(255))
    published_date = Column(DateTime)
    arxiv_url = Column(String(500))
    pdf_url = Column(String(500))
    citation_count = Column(Integer, default=0)
    heuristic_impact_score = Column(Float, nullable=True)
    relevance_score = Column(Float, default=0.0)
    personalized_summary = Column(Text, nullable=True)
    collected_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    recommended = Column(Boolean, default=False)
    recommended_date = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Paper(arxiv_id='{self.arxiv_id}', title='{self.title[:50]}...')>"


class Article(Base):
    """Tech article model"""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), index=True)  # hackernews, devto, medium, etc.
    source_id = Column(String(100), index=True)
    title = Column(String(500), index=True)
    url = Column(String(1000), unique=True)
    content = Column(Text)
    author = Column(String(255), nullable=True)
    published_date = Column(DateTime, nullable=True)
    upvotes = Column(Integer, default=0)
    engagement_score = Column(Float, default=0.0)
    relevance_score = Column(Float, default=0.0)
    personalized_summary = Column(Text, nullable=True)
    collected_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    recommended = Column(Boolean, default=False)
    recommended_date = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Article(source='{self.source}', title='{self.title[:50]}...')>"


class UserInteraction(Base):
    """Track user interactions for personalization"""
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    item_type = Column(String(20), nullable=False)  # "paper" or "article"
    item_id = Column(Integer, nullable=False)
    interaction_type = Column(String(20), nullable=False)  # "viewed", "saved", "dismissed"
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    user = relationship("User", back_populates="interactions")

    # Composite index for faster lookups
    __table_args__ = (
        Index('ix_user_item', 'user_id', 'item_type', 'item_id'),
    )

    def __repr__(self):
        return f"<UserInteraction(user_id={self.user_id}, type='{self.interaction_type}', item_id={self.item_id})>"


class UserDocument(Base):
    """User-uploaded documents for Q&A"""
    __tablename__ = "user_documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    source = Column(String(255), nullable=True)  # filename or URL
    content_hash = Column(String(64), nullable=True)  # SHA-256 hash for deduplication
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<UserDocument(user_id={self.user_id}, title='{self.title[:30]}...')>"


# Database setup
def get_engine():
    """Create database engine based on URL"""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        return create_engine(db_url, connect_args={"check_same_thread": False})
    else:
        # PostgreSQL or other databases
        return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

    # Handle SQLite-specific migrations for legacy tables
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        with engine.connect() as conn:
            # Check and add missing columns to papers table
            result = conn.execute(text("PRAGMA table_info(papers)")).fetchall()
            existing = {row[1] for row in result}
            if "heuristic_impact_score" not in existing:
                conn.execute(text("ALTER TABLE papers ADD COLUMN heuristic_impact_score FLOAT"))
                conn.commit()

            # Check if user_interactions has user_id column
            result = conn.execute(text("PRAGMA table_info(user_interactions)")).fetchall()
            existing = {row[1] for row in result}
            if "user_id" not in existing:
                # Add user_id column with default NULL
                conn.execute(text("ALTER TABLE user_interactions ADD COLUMN user_id INTEGER"))
                conn.commit()


def get_db():
    """Get database session - dependency injection for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Get a database session directly (for non-FastAPI contexts)"""
    return SessionLocal()
