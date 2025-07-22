"""Google Gemini embedding client for semantic deduplication."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from google import genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

from ..config import get_settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingClient:
    """Client for generating embeddings using Google Gemini API."""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google AI client."""
        if not GOOGLE_AI_AVAILABLE:
            logger.warning("Google AI SDK not available - embedding functionality disabled")
            return
            
        if not self.settings.google_ai_api_key:
            logger.warning("Google AI API key not configured - embedding functionality disabled")
            return
            
        try:
            self._client = genai.Client(api_key=self.settings.google_ai_api_key)
            logger.info("Google AI embedding client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google AI client: {e}")
            self._client = None
    
    def is_available(self) -> bool:
        """Check if embedding client is available."""
        return self._client is not None
    
    async def generate_embedding(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Optional[np.ndarray]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            task_type: Task type for the embedding (SEMANTIC_SIMILARITY, CLASSIFICATION, etc.)
            
        Returns:
            NumPy array of embedding vector, or None if failed
        """
        if not self.is_available():
            return None
            
        try:
            # Run in thread pool to avoid blocking async event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.models.embed_content(
                    model="models/text-embedding-004",  # Updated model name
                    contents=[text],  # Contents is a list
                    config={"output_dimensionality": 768}  # Config object
                )
            )
            
            # Extract embedding from response
            if hasattr(result, 'embeddings') and result.embeddings:
                embedding_obj = result.embeddings[0]
                
                # Handle different response formats
                if hasattr(embedding_obj, 'values'):
                    # New format with .values attribute
                    embedding = np.array(embedding_obj.values, dtype=np.float32)
                elif isinstance(embedding_obj, (list, tuple)):
                    # Direct list/tuple format
                    embedding = np.array(embedding_obj, dtype=np.float32)
                else:
                    # Try to convert the object to array
                    try:
                        embedding = np.array(embedding_obj, dtype=np.float32)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Cannot convert embedding object to array: {type(embedding_obj)}")
                        logger.error(f"Embedding object: {embedding_obj}")
                        logger.error(f"Conversion error: {e}")
                        return None
            else:
                logger.error(f"Unexpected response format: {type(result)}")
                logger.error(f"Response: {result}")
                return None
            logger.debug(f"Generated embedding of size {embedding.shape} for text length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str], task_type: str = "SEMANTIC_SIMILARITY") -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            task_type: Task type for the embeddings
            
        Returns:
            List of embedding arrays (same order as input texts)
        """
        if not self.is_available():
            return [None] * len(texts)
        
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.generate_embedding(text, task_type) for text in batch_texts]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for emb in batch_embeddings:
                if isinstance(emb, Exception):
                    logger.error(f"Embedding generation failed: {emb}")
                    embeddings.append(None)
                else:
                    embeddings.append(emb)
            
            # Brief pause to avoid rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        logger.info(f"Generated {sum(1 for e in embeddings if e is not None)} embeddings out of {len(texts)} texts")
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0


# Global embedding client instance
_embedding_client: Optional[GeminiEmbeddingClient] = None


def get_embedding_client() -> GeminiEmbeddingClient:
    """Get or create global embedding client instance."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = GeminiEmbeddingClient()
    return _embedding_client


def is_embedding_available() -> bool:
    """Check if embedding functionality is available."""
    client = get_embedding_client()
    return client.is_available()