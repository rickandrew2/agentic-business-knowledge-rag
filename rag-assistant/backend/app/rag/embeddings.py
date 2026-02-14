"""Embeddings integration with OpenAI API.

Handles:
- Generating embeddings for text
- Caching embeddings to avoid redundant API calls
- Batch embedding generation
"""

import logging
import json
from typing import List, Dict, Any
from pathlib import Path
import hashlib

from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, api_key: str = None, model: str = None, cache_dir: str = None):
        """Initialize embedding service.
        
        Args:
            api_key: OpenAI API key (uses settings if not provided)
            model: Embedding model (uses settings if not provided)
            cache_dir: Directory for caching embeddings
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./embedding_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized EmbeddingService with model: {self.model}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.
        
        Args:
            text: Text to cache
            
        Returns:
            Hash-based cache key
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str):
        """Retrieve cached embedding if available.
        
        Args:
            text: Text to retrieve embedding for
            
        Returns:
            Embedding vector or None if not cached
        """
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"Retrieved cached embedding for text: {text[:30]}...")
                    return data.get('embedding')
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding.
        
        Args:
            text: Original text
            embedding: Embedding vector
        """
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'text': text[:100],  # Store first 100 chars for reference
                    'embedding': embedding,
                    'model': self.model
                }, f)
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector
            
        Raises:
            Exception: If API call fails
        """
        # Clean text
        text = text.strip()
        if not text:
            raise ValueError("Cannot embed empty text")
        
        # Check cache
        if use_cache:
            cached = self._get_cached_embedding(text)
            if cached:
                return cached
        
        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            # Cache it
            if use_cache:
                self._cache_embedding(text, embedding)
            
            logger.debug(f"Generated embedding for text: {text[:30]}...")
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str], use_cache: bool = True, batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Note: OpenAI API supports batch requests up to 2048 embeddings.
        For MVP, we'll use sequential calls with caching for efficiency.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            batch_size: Batch size for API calls (max 2048)
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        text_indices = {}
        
        # First pass: check cache
        if use_cache:
            for idx, text in enumerate(texts):
                cached = self._get_cached_embedding(text)
                if cached:
                    embeddings.append(cached)
                else:
                    uncached_texts.append(text)
                    text_indices[len(embeddings)] = idx
                    embeddings.append(None)  # Placeholder
        else:
            uncached_texts = texts
            embeddings = [None] * len(texts)
            text_indices = {i: i for i in range(len(texts))}
        
        # Generate uncached embeddings
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
            
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                    
                    # Sort by index to maintain order
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    
                    for j, embedding_obj in enumerate(sorted_data):
                        embedding = embedding_obj.embedding
                        text = batch[j]
                        
                        # Cache it
                        if use_cache:
                            self._cache_embedding(text, embedding)
                        
                        # Place in correct position
                        original_idx = i + j
                        if use_cache:
                            # Find where this text was placed in embeddings list
                            for pos, idx in text_indices.items():
                                if idx == original_idx:
                                    embeddings[pos] = embedding
                                    break
                        else:
                            embeddings[original_idx] = embedding
                    
                    logger.info(f"Processed batch {i // batch_size + 1}")
                
                except Exception as e:
                    logger.error(f"Error generating batch embeddings: {e}")
                    raise
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings from this model.
        
        Returns:
            Embedding dimension
        """
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)
