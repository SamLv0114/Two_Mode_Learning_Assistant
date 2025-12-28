"""
Database models for storing papers and articles
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional
from src.utils.config import settings

Base = declarative_base()


class Paper(Base):
    """ArXiv paper model"""
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    arxiv_id = Column(String, unique=True, index=True)
    title = Column(String, index=True)
    authors = Column(Text)
    abstract = Column(Text)
    categories = Column(String)
    published_date = Column(DateTime)
    arxiv_url = Column(String)
    pdf_url = Column(String)
    citation_count = Column(Integer, default=0)
    relevance_score = Column(Float, default=0.0)
    personalized_summary = Column(Text, nullable=True)
    collected_date = Column(DateTime, default=datetime.utcnow)
    recommended = Column(Boolean, default=False)
    recommended_date = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Paper(arxiv_id='{self.arxiv_id}', title='{self.title[:50]}...')>"


class Article(Base):
    """Tech article model"""
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)  # hackernews, devto, medium, etc.
    source_id = Column(String, index=True)
    title = Column(String, index=True)
    url = Column(String, unique=True)
    content = Column(Text)
    author = Column(String, nullable=True)
    published_date = Column(DateTime, nullable=True)
    upvotes = Column(Integer, default=0)
    engagement_score = Column(Float, default=0.0)
    relevance_score = Column(Float, default=0.0)
    personalized_summary = Column(Text, nullable=True)
    collected_date = Column(DateTime, default=datetime.utcnow)
    recommended = Column(Boolean, default=False)
    recommended_date = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Article(source='{self.source}', title='{self.title[:50]}...')>"


class UserInteraction(Base):
    """Track user interactions for personalization"""
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    item_type = Column(String)  # "paper" or "article"
    item_id = Column(Integer)
    interaction_type = Column(String)  # "viewed", "saved", "dismissed"
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserInteraction(type='{self.interaction_type}', item_id={self.item_id})>"


# Database setup
# Handle SQLite connection string
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite"):
    # SQLite needs special connection args
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
else:
    engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

