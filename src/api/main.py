"""
Main FastAPI application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from apscheduler.schedulers.background import BackgroundScheduler
import logging

from src.utils.config import settings
from src.database.models import init_db
from src.api.routers import auth_router, feed_router, interactions_router, qa_router, chat_router
from src.jobs.nightly_indexer import run_nightly_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Module-level scheduler (shared across lifespan)
_scheduler = BackgroundScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler

    Runs initialization code on startup and cleanup on shutdown.
    """
    # Startup
    logger.info("Starting Learning Assistant API...")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Pre-load embedding manager (optional, improves first request latency)
    try:
        from src.api.deps import get_embedding_manager
        get_embedding_manager()
        logger.info("Embedding manager loaded")
    except Exception as e:
        logger.warning(f"Could not pre-load embedding manager: {e}")

    # Start nightly indexer — runs at 06:00 UTC daily (ArXiv RSS updates ~05:00 UTC)
    try:
        _scheduler.add_job(run_nightly_index, "cron", hour=6, minute=0, id="nightly_index")
        _scheduler.start()
        logger.info("Nightly indexer scheduled at 06:00 UTC daily")
    except Exception as e:
        logger.warning(f"Could not start scheduler: {e}")

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Learning Assistant API...")
    if _scheduler.running:
        _scheduler.shutdown(wait=False)


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    # Learning Assistant API

    A personalized ML research recommendation system with Q&A capabilities.

    ## Features

    - **Authentication**: JWT-based user authentication
    - **Daily Feed**: Personalized paper and article recommendations
    - **Interactions**: Track saved, viewed, and dismissed items
    - **Q&A Assistant**: Ask questions about your knowledge base
    - **Document Upload**: Add your own documents to the knowledge base

    ## Authentication

    Most endpoints require authentication. To authenticate:

    1. Register a new account at `/api/v1/auth/register`
    2. Login at `/api/v1/auth/login` to get an access token
    3. Include the token in the `Authorization` header: `Bearer <token>`
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with more detail"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )


# Include routers
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(feed_router, prefix=settings.API_V1_PREFIX)
app.include_router(interactions_router, prefix=settings.API_V1_PREFIX)
app.include_router(qa_router, prefix=settings.API_V1_PREFIX)
app.include_router(chat_router, prefix=settings.API_V1_PREFIX)


# Prometheus metrics endpoint (scraped by Prometheus every 15s)
@app.get("/metrics", tags=["Observability"], include_in_schema=False)
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Manual trigger for the nightly indexer — useful for testing and first-run seeding
@app.post("/api/v1/admin/index-now", tags=["Admin"])
async def trigger_index():
    """Run the nightly paper indexer immediately (for testing/manual refresh)."""
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_nightly_index)
    return result


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns the API status and version.
    """
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "api_title": settings.API_TITLE
    }


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint

    Provides links to documentation.
    """
    return {
        "message": "Welcome to the Learning Assistant API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


# Development server entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
