"""RAG Business Analytics Assistant - FastAPI Application Entry Point.

This is the main application server. Run with:
    uvicorn app.main:app --reload
"""

import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.security import SecureErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown.
    
    Future: Initialize embeddings model, vector DB, database connections here.
    """
    logger.info(f"Starting {settings.app_name}")
    
    # Startup
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI assistant that answers questions from company data using RAG",
    version="0.1.0",
    lifespan=lifespan
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: HTTPException(status_code=429))

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/api/health")
async def health():
    """Health check endpoint.
    
    Returns:
        dict: Status of the service
    """
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "0.1.0"
    }


@app.get("/api/config")
async def get_config():
    """Get public configuration (non-sensitive).
    
    Returns:
        dict: Public config values (model names, limits, etc.)
    """
    return {
        "app_name": settings.app_name,
        "openai_model": settings.openai_model,
        "chunk_size": settings.chunk_size,
        "retrieval_top_k": settings.retrieval_top_k,
        "temperature": settings.openai_temperature,
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Secure HTTP exception handler.
    
    Returns user-friendly errors without internal details.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.error(f"[{request_id}] HTTP Error {exc.status_code}: {exc.detail}")
    
    # Map status codes to user-friendly messages
    messages = {
        400: "Invalid request. Please check your input.",
        401: "Authentication required.",
        403: "Access denied.",
        404: "Resource not found.",
        429: "Too many requests. Please wait a moment.",
        500: "An unexpected error occurred. Please try again later.",
    }
    
    user_message = messages.get(exc.status_code, "An error occurred.")
    
    return {
        "error": user_message,
        "error_code": f"HTTP_{exc.status_code}",
        "request_id": request_id
    }


# Import routers
from app.api.routes import data as data_routes

# Include routers
app.include_router(data_routes.router)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
