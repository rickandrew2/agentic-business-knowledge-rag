"""Chroma vector database integration for RAG.

Handles:
- Storing embeddings in Chroma
- Similarity search
- Metadata filtering
"""

import logging
from typing import List, Dict, Any, Tuple
import chromadb

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store using Chroma."""
    
    def __init__(self, persist_directory: str = "./chroma_data", collection_name: str = "documents"):
        """Initialize Chroma vector store.
        
        Args:
            persist_directory: Directory to persist vector DB
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize Chroma client (use simple persistent client)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Initialized Chroma with collection: {collection_name}")
    
    def add_documents(self, chunks: List[str], embeddings: List[List[float]], 
                     metadata_list: List[Dict[str, Any]]) -> int:
        """Add documents with their embeddings to Chroma.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata_list: List of metadata dicts (one per chunk)
            
        Returns:
            Number of documents added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        if len(chunks) != len(embeddings) or len(chunks) != len(metadata_list):
            raise ValueError("Chunks, embeddings, and metadata must have same length")
        
        try:
            # Generate IDs
            ids = [f"doc_{i}" for i in range(len(chunks))]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata_list,
                documents=chunks
            )
            
            logger.info(f"Added {len(chunks)} documents to Chroma")
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
              where_filter: Dict[str, Any] = None) -> Tuple[List[str], List[Dict], List[float]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where_filter: Metadata filter (Chroma where syntax)
            
        Returns:
            Tuple of (documents, metadatas, distances)
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            # Extract results
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Convert distances to similarity scores (1 - distance for cosine)
            similarities = [1 - d for d in distances]
            
            logger.debug(f"Search returned {len(documents)} results")
            return documents, metadatas, similarities
        
        except Exception as e:
            logger.error(f"Error searching Chroma: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """Delete all documents in collection.
        
        WARNING: This is destructive and cannot be undone easily.
        """
        try:
            # Get all IDs
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.warning(f"Deleted {len(all_docs['ids'])} documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def delete_by_metadata(self, where_filter: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter.
        
        Args:
            where_filter: Chroma where filter
            
        Returns:
            Number of documents deleted
        """
        try:
            # Find matching documents
            results = self.collection.get(where=where_filter)
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents matching filter")
                return len(results['ids'])
            
            return 0
        
        except Exception as e:
            logger.error(f"Error deleting by metadata: {e}")
            raise


class RAGRetriever:
    """High-level retriever combining embeddings + vector store."""
    
    def __init__(self, embedding_service, vector_store: ChromaVectorStore):
        """Initialize retriever.
        
        Args:
            embedding_service: EmbeddingService instance
            vector_store: ChromaVectorStore instance
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        """Retrieve documents for a query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            Tuple of (documents, metadatas, relevance_scores)
        """
        try:
            # Embed query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Search vector store
            documents, metadatas, similarities = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            
            return documents, metadatas, similarities
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def build_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """Build context string from retrieved documents.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for doc, meta in zip(documents, metadatas):
            source = meta.get('source', 'Unknown')
            context_parts.append(f"[Source: {source}]\n{doc}")
        
        context = "\n\n---\n\n".join(context_parts)
        return context
