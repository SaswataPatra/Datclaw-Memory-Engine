"""
Embedding Service
Handles text embedding generation using OpenAI's embedding models
"""

from typing import Optional, List
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings
    
    Uses OpenAI's text-embedding-3-small model (1536 dimensions)
    for semantic similarity and vector search operations.
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize EmbeddingService
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        self.embedding_dim = 1536 if "small" in model else 3072
        
        logger.info(f"EmbeddingService initialized with model: {model} ({self.embedding_dim}D)")
    
    async def generate(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text: '{text[:50]}...' (dim: {len(embedding)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text[:50]}...': {e}", exc_info=True)
            return None
    
    async def generate_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in a single API call
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (or None for failed generations)
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings in batch")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}", exc_info=True)
            # Return None for all texts on batch failure
            return [None] * len(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service
        
        Returns:
            Embedding dimension (1536 for small, 3072 for large)
        """
        return self.embedding_dim

