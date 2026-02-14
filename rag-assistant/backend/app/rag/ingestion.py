"""RAG Data Ingestion Pipeline.

Handles:
- Loading CSV and text documents
- Chunking with RecursiveCharacterTextSplitter (from Context7 best practices)
- Embedding with OpenAI API  
- Storing in Qdrant vector database (Python 3.14 compatible)
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import csv
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings
from app.core.security import detect_pii, mask_pii

logger = logging.getLogger(__name__)


class RAGIngestionPipeline:
    """Handles data ingestion, chunking, embedding, and storage using Qdrant.
    
    Uses best practices from Context7:
    - RecursiveCharacterTextSplitter for intelligent chunking
    - OpenAI embeddings for semantic encoding
    - Qdrant for vector storage (Python 3.14 compatible)
    """
    
    def __init__(self):
        """Initialize the ingestion pipeline."""
        
        # Initialize text splitter with RecursiveCharacterTextSplitter
        # This recursively splits on various separators for better semantic chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings model (OpenAI API)
        logger.info(f"Initializing OpenAI embeddings: {settings.openai_embedding_model}")
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key
        )
        
        # Initialize Qdrant client (local persistent storage)
        logger.info(f"Initializing Qdrant client at: {settings.chroma_path}")
        self.vector_store = QdrantClient(path=settings.chroma_path)
        
        self.collection_name = "business_data"
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            self.vector_store.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            logger.info(f"Creating new collection: {self.collection_name}")
            
            # Get embedding dimension from a sample embedding
            sample_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)
            
            self.vector_store.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection created with embedding dimension: {embedding_dim}")
    
    def load_csv(self, file_path: str) -> List[Document]:
        """Load and parse CSV file into LangChain Documents.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading CSV file: {file_path}")
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    # Convert row to readable text
                    row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                    
                    # Check for PII
                    pii_detected = detect_pii(row_text)
                    if any(pii_detected.values()):
                        logger.warning(f"PII detected in row {row_idx}: {pii_detected}")
                        row_text = mask_pii(row_text)
                    
                    doc = Document(
                        page_content=row_text,
                        metadata={
                            "source": Path(file_path).name,
                            "source_type": "csv",
                            "row_index": row_idx,
                            "pii_masked": any(pii_detected.values())
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} rows from CSV")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_text(self, file_path: str) -> List[Document]:
        """Load and parse text/markdown file into LangChain Documents.
        
        Args:
            file_path: Path to text/markdown file
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for PII
            pii_detected = detect_pii(content)
            if any(pii_detected.values()):
                logger.warning(f"PII detected in file: {pii_detected}")
                content = mask_pii(content)
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": Path(file_path).name,
                    "source_type": "text",
                    "pii_masked": any(pii_detected.values())
                }
            )
            
            logger.info(f"Loaded {len(content)} characters from text file")
            return [doc]
            
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def embed_and_store(self, chunks: List[Document], batch_size: int = 100) -> Dict[str, int]:
        """Embed documents and store in Qdrant vector database.
        
        Args:
            chunks: List of document chunks
            batch_size: Number of documents to embed in each batch
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Embedding and storing {len(chunks)} chunks")
        
        stats = {
            "total_chunks": len(chunks),
            "stored_chunks": 0,
            "failed_chunks": 0
        }
        
        try:
            # Get embeddings for all chunks
            texts = [chunk.page_content for chunk in chunks]
            
            # Embed all texts (OpenAI handles batching internally)
            logger.info("Generating embeddings with OpenAI API")
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Prepare points for Qdrant
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Use UUID for unique IDs
                    vector=embedding,
                    payload={
                        "content": chunk.page_content,
                        "source": chunk.metadata.get("source"),
                        "source_type": chunk.metadata.get("source_type"),
                        "start_index": chunk.metadata.get("start_index", 0),
                        "pii_masked": chunk.metadata.get("pii_masked", False)
                    }
                )
                points.append(point)
            
            # Store in Qdrant
            logger.info(f"Storing {len(points)} points in Qdrant collection")
            self.vector_store.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            stats["stored_chunks"] = len(points)
            logger.info(f"Successfully stored {stats['stored_chunks']} chunks")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error embedding and storing: {e}")
            stats["failed_chunks"] = len(chunks)
            raise
    
    def ingest_file(self, file_path: str) -> Dict[str, any]:
        """Full ingestion pipeline: load → chunk → embed → store.
        
        Args:
            file_path: Path to file to ingest
            
        Returns:
            Dictionary with ingestion results and statistics
        """
        logger.info(f"Starting ingestion pipeline for: {file_path}")
        
        try:
            # Determine file type and load
            if file_path.lower().endswith('.csv'):
                documents = self.load_csv(file_path)
            else:
                documents = self.load_text(file_path)
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Embed and store
            stats = self.embed_and_store(chunks)
            
            stats.update({
                "file": Path(file_path).name,
                "status": "success",
                "original_documents": len(documents)
            })
            
            logger.info(f"Ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {
                "file": Path(file_path).name,
                "status": "failed",
                "error": str(e)
            }
