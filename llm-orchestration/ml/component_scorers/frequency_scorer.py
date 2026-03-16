from typing import Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from datetime import datetime, timedelta
import asyncio
import logging
import os
from arango import ArangoClient

from ml.component_scorers.base import ComponentScorer, ScorerResult

logger = logging.getLogger(__name__)


class FrequencyScorer(ComponentScorer):
    """
    Calculates frequency score based on how many times similar memories have been stored
    within a recent time window.
    A higher score means the memory is more frequent.
    """
    
    def __init__(self, config: Dict[str, Any], qdrant_client: QdrantClient, event_bus=None):
        super().__init__(config)
        self.qdrant = qdrant_client
        self.event_bus = event_bus  # Optional: for event-driven indexing wait
        self.collection_name = config.get('qdrant', {}).get('collection_name', 'memories')
        
        # Initialize ArangoDB client for fetching memory content
        arango_config = config.get('arangodb', {})
        arango_url = arango_config.get('url', 'http://localhost:8529')
        # Replace env vars if present
        arango_url = arango_url.replace('${ARANGO_HOST:localhost}', os.getenv('ARANGO_HOST', 'localhost'))
        arango_url = arango_url.replace('${ARANGO_PORT:8529}', os.getenv('ARANGO_PORT', '8529'))
        
        try:
            arango_client = ArangoClient(hosts=arango_url)
            self.arango_db = arango_client.db(
                arango_config.get('database', 'dappy_memories'),
                username=arango_config.get('username', 'root'),
                password=os.getenv('ARANGO_PASSWORD', arango_config.get('password', 'dappy_dev_password'))
            )
            self.memory_collection = self.arango_db.collection(
                arango_config.get('memory_collection', 'memories')
            )
        except Exception as e:
            logger.warning(f"Could not connect to ArangoDB for memory content fetching: {e}")
            self.arango_db = None
            self.memory_collection = None
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2-3 (user-specific memory window)
        # How far back to look for similar memories when calculating frequency
        # Current: 30 days = consider last month of conversations
        # Future: Learned per-user (some have longer memory context, others shorter)
        self.lookback_days = config.get('ego_scoring', {}).get('frequency_lookback_days', 30)
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2-3 (per-user, context-aware)
        # Lower than novelty (0.7) to capture topic-level similarity, not just exact matches
        # Current: 0.6 = captures related topics (e.g., "stakes" and "fillet mignon")
        # Future: Learned per-user and per-topic (family: 0.5, work: 0.7, etc.)
        self.similarity_threshold = config.get('ego_scoring', {}).get('frequency_similarity_threshold', 0.6)
        
        # ⚙️ STAYS - Normalization constant for frequency score
        # Maximum number of similar memories to count (caps frequency at 1.0)
        self.max_frequency_count = config.get('ego_scoring', {}).get('max_frequency_count', 10)
        
        # ⚙️ STAYS - Infrastructure parameter for async indexing race condition
        # Timeout for waiting for memory.indexed event (event-driven approach)
        self.indexing_timeout = config.get('ego_scoring', {}).get('frequency_indexing_timeout', 2.0)  # seconds
    
    async def score(self, memory: Dict[str, Any], **kwargs) -> ScorerResult:
        """
        Score frequency of a memory.
        Requires 'embedding' and 'user_id' in memory.
        
        Note: Uses event-driven approach to wait for memory indexing completion.
        If event_bus is available, waits for 'memory.indexed' event before querying.
        """
        embedding = memory.get('embedding')
        user_id = memory.get('user_id')
        memory_id = memory.get('memory_id') or memory.get('node_id')
        
        if embedding is None or user_id is None:
            return ScorerResult(score=0.0, metadata={"reason": "missing_embedding_or_user_id"})
        
        # Calculate time window
        now = datetime.utcnow()
        lookback_date = now - timedelta(days=self.lookback_days)
        
        # Filter by user_id only (time filtering would require timestamp field in Qdrant)
        # In production, you'd add a timestamp field and filter by it
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
        
        # Event-driven approach: Wait for memory.indexed event if event_bus is available
        # This eliminates the race condition between memory creation and indexing
        # NOTE: This only works for EXISTING memories being re-scored, not new memories
        # For new memories (no memory_id), we skip the wait since they haven't been published yet
        if self.event_bus and memory_id:
            logger.debug(f"Waiting for memory {memory_id} to be indexed...")
            
            # Wait for the specific memory to be indexed
            indexed_event = await self.event_bus.wait_for_event(
                topic="memory.indexed",
                filter_fn=lambda e: e.payload.get('node_id') == memory_id,
                timeout=self.indexing_timeout
            )
            
            if indexed_event:
                logger.debug(f"Memory {memory_id} indexed, proceeding with frequency scoring")
            else:
                logger.debug(f"Timeout waiting for memory {memory_id} indexing, proceeding anyway")
        elif not memory_id:
            # New memory being scored before publication - can't wait for indexing
            # This is expected and normal for the first-time scoring flow
            logger.debug("New memory (no ID yet) - skipping indexing wait")
        
        # Query Qdrant for similar memories
        try:
            all_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter=query_filter,
                limit=self.max_frequency_count + 1,  # Get one more to check if capped
                score_threshold=0.0  # Get all results for debugging
            )
            
            # Filter by threshold
            search_result = [r for r in all_results if r.score >= self.similarity_threshold]
            all_results_with_scores = [(r.id, r.score) for r in all_results[:5]]  # Top 5 for debugging
            
            # Log what we found
            if all_results:
                logger.info(f"Frequency scorer: Found {len(all_results)} total memories, {len(search_result)} above threshold {self.similarity_threshold}")
                logger.debug(f"  Top scores: {all_results_with_scores}")
                
                # Log the similar memories that passed the threshold
                if search_result:
                    logger.info(f"  📋 Similar memories found:")
                    for i, result in enumerate(search_result[:5], 1):  # Show top 5 similar
                        # Get memory content from payload if available
                        created_at = result.payload.get('created_at', 'N/A')[:19] if result.payload else 'N/A'
                        tier = result.payload.get('tier', 'N/A') if result.payload else 'N/A'
                        ego_score = result.payload.get('ego_score', 0) if result.payload else 0
                        
                        # Fetch actual memory content from ArangoDB
                        memory_content = "N/A"
                        if self.memory_collection:
                            try:
                                memory_doc = self.memory_collection.get(result.id)
                                if memory_doc:
                                    memory_content = memory_doc.get('content', memory_doc.get('summary', 'N/A'))
                                    # Truncate for logging
                                    if len(memory_content) > 80:
                                        memory_content = memory_content[:80] + "..."
                            except Exception as e:
                                logger.debug(f"Could not fetch memory content for {result.id}: {e}")
                        
                        logger.info(
                            f"    {i}. Similarity: {result.score:.3f} | "
                            f"ID: {result.id} | "
                            f"Tier: {tier} | "
                            f"Ego: {ego_score:.3f} | "
                            f"Created: {created_at}"
                        )
                        logger.info(f"       Content: \"{memory_content}\"")
                    
        except Exception as e:
            logger.error(f"Frequency scorer error: {e}")
            raise
        
        frequency_count = len(search_result)
        
        # Normalize frequency score
        frequency_score = min(1.0, frequency_count / self.max_frequency_count)
        
        logger.info(f"Frequency scorer: Final result = {frequency_count} similar memories (threshold={self.similarity_threshold}, score={frequency_score:.3f})")
        
        return ScorerResult(
            score=frequency_score,
            metadata={
                "frequency_count": frequency_count,
                "lookback_days": self.lookback_days,
                "similarity_threshold": self.similarity_threshold,
                "all_scores": all_results_with_scores,
                "event_driven": self.event_bus is not None
            }
        )
