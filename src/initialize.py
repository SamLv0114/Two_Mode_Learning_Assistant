"""
Initialize the system (database, vector DB, etc.)
"""
import logging
from src.database.models import init_db
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize():
    """Initialize all system components"""
    logger.info("Initializing AI Learning Assistant...")
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    
    # Initialize vector database (will be created on first use)
    logger.info("Vector database will be initialized on first use")
    
    # Create directories
    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.MODELS_DIR.mkdir(exist_ok=True)
    settings.VECTOR_DB_DIR.mkdir(exist_ok=True)
    logger.info("Directories created")
    
    logger.info("Initialization complete!")
    logger.info("\nNext steps:")
    logger.info("1. Make sure your .env file is configured with API keys")
    logger.info("2. Run the daily feed: python -m src.pipelines.daily_feed")
    logger.info("3. Or use Q&A assistant: python -m src.pipelines.qa_assistant")


if __name__ == "__main__":
    initialize()

