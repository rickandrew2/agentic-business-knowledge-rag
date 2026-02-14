"""Application configuration management.

Use environment variables via .env file.
Example:
    OPENAI_API_KEY=sk-...
    CHROMA_PATH=./chroma_data
    DATABASE_URL=sqlite:///./rag.db
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(env_file=".env", case_sensitive=False)
    
    # OpenAI API Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.3  # Low temp = deterministic, less hallucination
    
    # Vector Database Configuration
    chroma_path: str = "./chroma_data"
    
    # SQL Database Configuration
    database_url: str = "sqlite:///./rag.db"
    
    # Application Settings
    app_name: str = "RAG Business Analytics Assistant"
    debug: bool = False
    
    # RAG Parameters
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 128  # tokens
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.5
    
    # Rate Limiting
    rate_limit_chat: str = "10/minute"
    rate_limit_upload: str = "5/minute"


# Singleton instance
settings = Settings()
