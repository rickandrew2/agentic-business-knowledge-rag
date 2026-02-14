"""API request/response schemas (Pydantic models)."""

from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================
# Data Ingestion Schemas
# ============================================================

class DataUploadRequest(BaseModel):
    """Request to upload data file."""
    filename: str
    file_type: str  # "csv", "markdown", "text"
    
    @field_validator('file_type')
    @classmethod
    def validate_file_type(cls, v):
        allowed = ['csv', 'markdown', 'text']
        if v not in allowed:
            raise ValueError(f"file_type must be one of {allowed}")
        return v


class DataUploadResponse(BaseModel):
    """Response from data upload."""
    file_id: str
    filename: str
    chunks_created: int
    documents_created: int
    status: str  # "success", "processing", "error"
    message: str
    timestamp: datetime


class DataStatusRequest(BaseModel):
    """Request to check data ingestion status."""
    file_id: Optional[str] = None  # If None, return all


class DataStatusResponse(BaseModel):
    """Response with data ingestion status."""
    total_files: int
    total_documents: int
    total_chunks: int
    files: List[Dict[str, Any]]


# ============================================================
# Search/Retrieval Schemas
# ============================================================

class SearchRequest(BaseModel):
    """Request to search documents."""
    query: str
    top_k: int = 5
    min_score: float = 0.0  # Minimum relevance score
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if len(v) < 1:
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        if v < 1 or v > 50:
            raise ValueError("top_k must be between 1 and 50")
        return v


class SearchResult(BaseModel):
    """Single search result."""
    rank: int
    text: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response from search."""
    query: str
    result_count: int
    results: List[SearchResult]
    context: str  # Formatted context from all results
    search_time_ms: float


# ============================================================
# Chat/RAG Schemas
# ============================================================

class ChatMessage(BaseModel):
    """A chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    """Request for chat with RAG."""
    message: str
    top_k: int = 5
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if len(v) < 1:
            raise ValueError("Message cannot be empty")
        if len(v) > 2000:
            raise ValueError("Message too long")
        return v.strip()


class ChatResponse(BaseModel):
    """Response from chat/RAG."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float  # 0.0 to 1.0
    retrieval_time_ms: float
    generation_time_ms: float
    model: str


# ============================================================
# Evaluation Schemas
# ============================================================

class MetricsResponse(BaseModel):
    """Evaluation metrics."""
    precision_at_5: float
    mrr: float  # Mean Reciprocal Rank
    avg_retrieval_time_ms: float
    total_queries: int
    last_updated: datetime
