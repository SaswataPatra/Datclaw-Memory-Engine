from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import asyncio
import logging

from ml.component_scorers.base import ComponentScorer, ScorerResult

logger = logging.getLogger(__name__)


class NoveltyScorer(ComponentScorer):
    """
    Calculates novelty score based on semantic similarity to existing memories in Qdrant.
    A higher score means the memory is more novel (less similar to existing ones).
    """
    
    def __init__(self, config: Dict[str, Any], qdrant_client: QdrantClient):
        super().__init__(config)
        self.qdrant = qdrant_client
        self.collection_name = config.get('qdrant', {}).get('collection_name', 'memories')
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2-3 (per-user, context-aware)
        # Controls how similar a memory must be to existing ones to be considered "not novel"
        # Current: 0.7 = strict (only very similar memories reduce novelty)
        # Future: Learned per-user (some want strict, others want broad)
        self.similarity_threshold = config.get('ego_scoring', {}).get('novelty_similarity_threshold', 0.7)
        
        # ⚙️ STAYS - Computational limit for vector search
        # Number of similar memories to retrieve for comparison
        self.top_k = config.get('ego_scoring', {}).get('novelty_top_k', 5)
        
        # ⚙️ STAYS - Infrastructure parameter for async indexing race condition
        # Retries to handle delay between memory creation and Qdrant indexing
        self.max_retries = config.get('ego_scoring', {}).get('novelty_max_retries', 3)
        self.retry_delay = config.get('ego_scoring', {}).get('novelty_retry_delay', 0.8)  # seconds
    
    async def score(self, memory: Dict[str, Any], **kwargs) -> ScorerResult:
        """
        Score novelty of a memory.
        Requires 'embedding' and 'user_id' in memory.
        
        Note: Includes retry logic to handle async indexing delay from Qdrant consumer.
        """
        embedding = memory.get('embedding')
        user_id = memory.get('user_id')
        
        if embedding is None or user_id is None:
            return ScorerResult(score=0.5, metadata={"reason": "missing_embedding_or_user_id"})
        
        # Filter by user_id and optionally by memory type
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
        
        # Retry logic to handle async indexing delay
        search_result = []
        for attempt in range(self.max_retries + 1):
            try:
                # Search for similar memories (Qdrant client is synchronous)
                search_result = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    query_filter=query_filter,
                    limit=self.top_k,
                    score_threshold=self.similarity_threshold  # Only consider above threshold
                )
                
                # If we found results or this is the last attempt, break
                if search_result or attempt == self.max_retries:
                    break
                
                # Wait before retrying (to allow Qdrant consumer to index recent memories)
                if attempt < self.max_retries:
                    logger.debug(f"Novelty scorer: No results on attempt {attempt + 1}, retrying after {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Novelty scorer error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    raise
        
        if not search_result:
            # No similar memories found above threshold, very novel
            logger.debug(f"Novelty scorer: no similar memories found (threshold={self.similarity_threshold})")
            return ScorerResult(score=1.0, metadata={"reason": "no_similar_memories", "retries": self.max_retries})
        
        # Max similarity found
        max_similarity = max([hit.score for hit in search_result])
        
        # Novelty is inverse of similarity (1 - similarity)
        # Scale to ensure it's between 0 and 1
        novelty_score = 1.0 - max_similarity
        
        logger.debug(f"Novelty scorer: max_similarity={max_similarity:.3f}, novelty_score={novelty_score:.3f}")
        
        return ScorerResult(
            score=novelty_score,
            metadata={
                "max_similarity": max_similarity,
                "similar_memories_count": len(search_result),
                "top_similar_node_id": search_result[0].id if search_result else None,
                "retries": attempt if search_result else self.max_retries
            }
        )
