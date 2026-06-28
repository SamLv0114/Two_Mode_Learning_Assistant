"""
Configuration settings for the AI Learning Assistant
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Optional
import secrets


class Settings(BaseSettings):
    """Application settings"""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None

    # LLM Settings
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ArXiv settings
    ARXIV_CATEGORIES: List[str] = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE"]
    MAX_PAPERS_PER_DAY: int = 50

    # Tech article sources
    TECH_SOURCES: List[str] = ["hackernews", "devto", "medium"]

    # Recommendation settings
    TOP_PAPERS_COUNT: int = 5
    TOP_ARTICLES_COUNT: int = 3
    MIN_SIMILARITY_THRESHOLD: float = 0.3

    # Learning-to-rank settings
    USE_LTR: bool = True

    # Diversity settings
    USE_MMR_DIVERSITY: bool = True
    MMR_LAMBDA: float = 0.7
    MMR_CANDIDATE_MULTIPLIER: int = 5

    # Exploration settings
    EXPLORATION_RATE: float = 0.2

    # Novelty settings
    NOVELTY_LOOKBACK_DAYS: int = 14
    NOVELTY_MAX_ITEMS: int = 50

    # Implicit feedback handling
    INCLUDE_IMPLICIT_NEGATIVES: bool = True
    IMPLICIT_NEGATIVE_SAMPLE_RATE: float = 0.15

    # Vector database
    VECTOR_DB_COLLECTION_NAME: str = "ml_knowledge_base"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # User preferences - read as string from .env, parsed to list (for legacy/default)
    USER_INTERESTS_STR: Optional[str] = None

    # Database
    DATABASE_URL: str = "sqlite:///./data/learning_assistant.db"

    # ============================================================
    # NEW: Authentication & Security Settings
    # ============================================================

    # JWT Configuration
    SECRET_KEY: str = secrets.token_urlsafe(32)  # Override in .env for production!
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Password hashing
    BCRYPT_ROUNDS: int = 12

    # ============================================================
    # NEW: API Settings
    # ============================================================

    # CORS - stored as comma-separated string in .env
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        """Parse CORS_ORIGINS from comma-separated string"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # API
    API_V1_PREFIX: str = "/api/v1"
    API_TITLE: str = "Learning Assistant API"
    API_VERSION: str = "1.0.0"

    # Rate limiting (requests per minute)
    RATE_LIMIT_PER_MINUTE: int = 60

    # ============================================================
    # NEW: Redis (for caching, optional)
    # ============================================================
    REDIS_URL: Optional[str] = None  # e.g., "redis://localhost:6379"

    # ============================================================
    # NEW: Model Training Settings
    # ============================================================
    MIN_INTERACTIONS_FOR_TRAINING: int = 50
    AUTO_RETRAIN_INTERVAL_HOURS: int = 24

    @property
    def USER_INTERESTS(self) -> List[str]:
        """Parse USER_INTERESTS from comma-separated string (legacy support)"""
        if self.USER_INTERESTS_STR:
            return [item.strip() for item in self.USER_INTERESTS_STR.split(",") if item.strip()]
        # Default values
        return [
            "machine learning",
            "deep learning",
            "natural language processing",
            "computer vision",
            "reinforcement learning"
        ]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Create directories and update database URL
settings = Settings()
settings.DATA_DIR.mkdir(exist_ok=True)
settings.RAW_DATA_DIR.mkdir(exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
settings.VECTOR_DB_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)

# Update database URL with absolute path for SQLite
if settings.DATABASE_URL.startswith("sqlite"):
    db_path = settings.DATA_DIR / "learning_assistant.db"
    settings.DATABASE_URL = f"sqlite:///{db_path}"
