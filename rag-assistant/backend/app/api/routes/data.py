"""API routes for data management and RAG operations."""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.schemas import (
    DataUploadResponse,
    DataStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResult
)
from app.rag.ingestion import RAGIngestionPipeline
from app.core.config import settings
from app.core.security import detect_pii, mask_pii
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Global state (in production, use database)
_uploaded_files: Dict[str, Dict[str, Any]] = {}
_rag_pipeline = None
_embeddings = None


def init_rag_system():
    """Initialize RAG components."""
    global _rag_pipeline, _embeddings
    
    _rag_pipeline = RAGIngestionPipeline()
    _embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key
    )
    
    logger.info("RAG system initialized with Qdrant")


# ============================================================
# Data Upload Endpoint
# ============================================================

@router.post("/api/data/upload")
@limiter.limit("5/minute")
async def upload_data(request: Request, file: UploadFile = File(...)):
    """Upload and ingest a data file.
    
    Supports: CSV, Markdown, Text files
    
    Args:
        request: HTTP request object (for rate limiting)
        file: File to upload
        
    Returns:
        DataUploadResponse with ingestion status
    """
    
    # Initialize RAG if needed
    if _rag_pipeline is None:
        init_rag_system()
    
    file_id = str(uuid.uuid4())[:8]
    
    try:
        # Validate file type
        allowed_extensions = ('.csv', '.md', '.txt')
        if not any(file.filename.endswith(ext) for ext in allowed_extensions):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {allowed_extensions}"
            )
        
        # Check file size (max 50MB for MVP)
        file_size = 0
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            temp_path = tmp_file.name
            
            # Read and scan file content
            content = await file.read()
            file_size = len(content)
            
            if file_size > 50 * 1024 * 1024:  # 50MB
                raise HTTPException(status_code=413, detail="File too large (max 50MB)")
            
            tmp_file.write(content)
        
        logger.info(f"Received file {file.filename} ({file_size} bytes)")
        
        # Check for PII in filename
        pii_in_filename = detect_pii(file.filename)
        if any(pii_in_filename.values()):
            logger.warning(f"PII detected in filename: {pii_in_filename}")
            # Don't block, just warn and mask
            safe_filename = mask_pii(file.filename)
            logger.info(f"Masked filename: {file.filename} → {safe_filename}")
        
        # Ingest file using pipeline (handles chunking, embedding, and storage)
        start_time = time.time()
        result = _rag_pipeline.ingest_file(temp_path)
        ingestion_time = time.time() - start_time
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('error', 'Ingestion failed'))
        
        logger.info(f"Created {result['total_chunks']} chunks")
        
        # Track uploaded file
        _uploaded_files[file_id] = {
            "filename": file.filename,
            "file_size": file_size,
            "chunks_created": result['total_chunks'],
            "documents_created": result['original_documents'],
            "uploaded_at": datetime.now(),
            "status": "completed"
        }
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        logger.info(f"File {file_id} ingested successfully in {ingestion_time:.2f}s")
        
        return DataUploadResponse(
            file_id=file_id,
            filename=file.filename,
            chunks_created=result['total_chunks'],
            documents_created=result['original_documents'],
            status="success",
            message=f"File ingested successfully. Created {result['total_chunks']} chunks.",
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Data Status Endpoint
# ============================================================

@router.get("/api/data/status")
async def get_data_status():
    """Get status of uploaded data and ingestion.
    
    Returns:
        DataStatusResponse with current stats
    """
    
    if _rag_pipeline is None:
        init_rag_system()
    
    try:
        # Get collection info from Qdrant
        collection_info = _rag_pipeline.vector_store.get_collection("business_data")
        document_count = collection_info.points_count
        
        return DataStatusResponse(
            total_files=len(_uploaded_files),
            total_documents=len(_uploaded_files),
            total_chunks=document_count,
            files=list(_uploaded_files.values())
        )
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Search Endpoint
# ============================================================

@router.post("/api/data/search")
@limiter.limit("10/minute")
async def search_documents(request: Request, search_request: SearchRequest):
    """Search for documents similar to query.
    
    Args:
        request: HTTP request object (for rate limiting)
        search_request: SearchRequest with query and parameters
        
    Returns:
        SearchResponse with results
    """
    
    if _rag_pipeline is None:
        init_rag_system()
    
    try:
        start_time = time.time()
        
        # Validate query
        query = search_request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Searching for: {query}")
        
        # Embed query
        query_embedding = _embeddings.embed_query(query)
        
        # Search Qdrant
        response = _rag_pipeline.vector_store.query_points(
            collection_name="business_data",
            query=query_embedding,
            limit=search_request.top_k,
            score_threshold=search_request.min_score
        )
        
        # Format results
        results = []
        for rank, hit in enumerate(response.points, 1):
            results.append(SearchResult(
                rank=rank,
                text=hit.payload.get('content', ''),
                source=hit.payload.get('source', 'Unknown'),
                relevance_score=float(hit.score),
                metadata=hit.payload
            ))
        
        # Build context
        context = "\n\n---\n\n".join([
            f"[Source: {r.source}]\n{r.text}"
            for r in results
        ])
        
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SearchResponse(
            query=query,
            result_count=len(results),
            results=results,
            context=context,
            search_time_ms=search_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
