"""
Configuration settings for the AI Learning Assistant
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional, Union


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
    
    # LLM Settings
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"  # or "gpt-3.5-turbo" for cheaper option
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
    
    # Vector database
    VECTOR_DB_COLLECTION_NAME: str = "ml_knowledge_base"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Citation enrichment
    CITATION_ENRICHMENT_ENABLED: bool = False
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    CITATION_API_TIMEOUT: int = 5  # seconds

    # User preferences - read as string from .env, parsed to list
    USER_INTERESTS_STR: Optional[str] = None
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/learning_assistant.db"
    

    @property
    def USER_INTERESTS(self) -> List[str]:
        """Parse USER_INTERESTS from comma-separated string"""
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
        extra = "ignore"  # Ignore extra fields


# Create directories and update database URL
settings = Settings()
settings.DATA_DIR.mkdir(exist_ok=True)
settings.RAW_DATA_DIR.mkdir(exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
settings.VECTOR_DB_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)

# Update database URL with absolute path
db_path = settings.DATA_DIR / "learning_assistant.db"
settings.DATABASE_URL = f"sqlite:///{db_path}"
