"""
DAPPY - Qdrant Consumer
Consumes memory.upsert events and indexes vectors in Qdrant
"""

import asyncio
import uuid
from typing import Dict, Any, List
from datetime import datetime
import logging
import openai

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


class QdrantConsumer:
    """
    Qdrant Consumer for memory.upsert events
    
    Responsibilities:
    - Generate embeddings for memory content
    - Index vectors in Qdrant (fast vector store)
    - Store minimal payload (node_id, entities, version, etc.)
    - Handle versioning for idempotency
    - Update entity tags when memory.entities_extracted events arrive
    
    Note: Entities are extracted AFTER initial indexing (async KG consolidation),
    so we handle both initial indexing and entity updates.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Connect to Qdrant
        qdrant_config = config.get('qdrant', {})
        self.client = QdrantClient(
            url=qdrant_config['url'],
            timeout=qdrant_config.get('timeout', 30)
        )
        
        self.collection_name = qdrant_config.get('collection_name', 'memories')
        self.vector_size = qdrant_config.get('vector_size', 1536)
        self.distance = qdrant_config.get('distance', 'Cosine')
        
        # Embedding config
        embeddings_config = config.get('embeddings', {})
        self.embedding_model = embeddings_config.get('model', 'text-embedding-3-small')
        openai.api_key = embeddings_config.get('api_key')
        
        # Initialize collection
        self._init_collection()
        
        # Track processing stats
        self.processed_count = 0
        self.error_count = 0
        self.embedding_errors = 0
        
        self._running = False
        
        logger.info("Qdrant Consumer initialized", extra={"collection": self.collection_name})
    
    def _init_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE if self.distance == 'Cosine' else Distance.EUCLID
                    )
                )
                
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}", exc_info=True)
            raise
    
    async def start(self):
        """Start consuming memory.upsert events"""
        self._running = True
        
        # Subscribe to memory.upsert topic (for initial indexing)
        await self.event_bus.subscribe(
            topic="memory.upsert",
            handler=self._handle_memory_upsert,
            consumer_group="qdrant-consumer"
        )
        
        logger.info("Qdrant Consumer started")
    
    async def stop(self):
        """Stop consuming events"""
        self._running = False
        
        await self.event_bus.unsubscribe("memory.upsert")
        
        logger.info("Qdrant Consumer stopped")
    
    async def _handle_memory_upsert(self, event: Event):
        """
        Handle memory.upsert event
        
        Generates embedding and indexes in Qdrant.
        
        Note: If event_type is 'memory.entities_extracted', only updates the entities
        field without re-embedding (entities are extracted after initial indexing).
        """
        
        try:
            payload = event.payload
            
            node_id = payload.get('node_id')
            version = payload.get('version', 1)
            event_type = event.event_type
            
            # Check if this version already exists (idempotency)
            existing_point = self._get_existing_point(node_id)
            
            # Special handling for entity extraction events (update entities only)
            if event_type == "memory.entities_extracted":
                if not existing_point:
                    logger.warning(f"Cannot update entities for non-existent point: {node_id}")
                    return
                
                # Update only the entities field in the existing point
                updated_payload = existing_point.payload
                updated_payload["entities"] = payload.get('entities', [])
                updated_payload["version"] = version
                
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=updated_payload,
                    points=[existing_point.id]
                )
                
                logger.info(f"Updated entities in Qdrant: {node_id} ({len(payload.get('entities', []))} entities)")
                return
            
            if existing_point and existing_point.payload.get('version', 0) >= version:
                logger.debug(
                    f"Skipping older version: {node_id} v{version}",
                    extra={"node_id": node_id, "existing_version": existing_point.payload.get('version')}
                )
                return
            
            # Generate embedding
            content = payload.get('content', '') or payload.get('summary', '')
            
            if not content:
                logger.warning(f"No content to embed for {node_id}")
                return
            
            embedding = await self._generate_embedding(content)
            
            if embedding is None:
                self.embedding_errors += 1
                logger.error(f"Failed to generate embedding for {node_id}")
                return
            
            # Build Qdrant payload (minimal - only IDs and version)
            # ArangoDB is canonical store for full metadata
            qdrant_payload = {
                "node_id": node_id,
                "user_id": payload.get('user_id'),
                "tier": payload.get('tier'),
                "ego_score": payload.get('ego_score'),
                "created_at": payload.get('created_at'),
                "observed_at": payload.get('observed_at'),
                "recency_score": payload.get('time_features', {}).get('recency_score', 1.0),
                "entities": payload.get('entities', []),  # Entity tags for localized KG
                "version": version
            }
            
            # Reuse existing point UUID if updating, otherwise generate a new one
            if existing_point:
                point_id = existing_point.id
            else:
                point_id = str(uuid.uuid4())
            
            # Qdrant requires IDs to be either integers or valid UUIDs.
            # node_id has a "mem_" prefix so we use a proper UUID as the point ID
            # and store node_id in the payload for reference.
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=qdrant_payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            self.processed_count += 1
            
            try:
                from core.metrics import STORAGE_OPERATIONS
                STORAGE_OPERATIONS.labels(store='qdrant', operation='upsert', outcome='success').inc()
            except Exception:
                pass
            
            logger.info(
                f"Indexed memory in Qdrant: {node_id}",
                extra={
                    "node_id": node_id,
                    "tier": payload.get('tier'),
                    "ego_score": payload.get('ego_score'),
                    "version": version
                }
            )
            
            # Publish memory.indexed event for downstream consumers
            # This allows frequency scorer to wait for indexing completion
            from core.event_bus import Event
            await self.event_bus.publish(
                Event(
                    topic="memory.indexed",
                    event_type="memory.indexed",
                    payload={
                        "node_id": node_id,
                        "user_id": payload.get('user_id'),
                        "version": version,
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
            )
            
        except Exception as e:
            self.error_count += 1
            try:
                from core.metrics import STORAGE_OPERATIONS
                STORAGE_OPERATIONS.labels(store='qdrant', operation='upsert', outcome='error').inc()
            except Exception:
                pass
            logger.error(
                f"Failed to index memory in Qdrant: {e}",
                extra={"event_id": event.event_id},
                exc_info=True
            )
            raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            return None
    
    def _get_existing_point(self, node_id: str):
        """Get existing point by node_id in payload."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="node_id", match=MatchValue(value=node_id))]
                ),
                limit=1
            )
            points, _ = results
            return points[0] if points else None
        except Exception as e:
            logger.warning(f"Failed to retrieve point: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "processed_count": self.processed_count,
                "error_count": self.error_count,
                "embedding_errors": self.embedding_errors,
                "error_rate": self.error_count / max(1, self.processed_count),
                "total_points": collection_info.points_count
            }
        
        except Exception as e:
            logger.error(f"Failed to get Qdrant stats: {e}")
            return {
                "processed_count": self.processed_count,
                "error_count": self.error_count,
                "embedding_errors": self.embedding_errors
            }

