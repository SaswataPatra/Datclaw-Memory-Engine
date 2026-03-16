"""
DAPPY - ArangoDB Consumer
Consumes memory.upsert events and writes to ArangoDB (canonical store)
"""

import asyncio
from typing import Dict, Any
from datetime import datetime
import logging

from arango import ArangoClient
from core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


class ArangoDBConsumer:
    """
    ArangoDB Consumer for memory.upsert events
    
    Responsibilities:
    - Write memory nodes to ArangoDB (canonical source of truth)
    - Maintain full metadata and provenance
    - Create/update graph edges
    - Handle versioning for idempotency
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Connect to ArangoDB
        arango_config = config.get('arangodb', {})
        client = ArangoClient(hosts=arango_config['url'])
        self.db = client.db(
            arango_config['database'],
            username=arango_config['username'],
            password=arango_config['password']
        )
        
        self.memory_collection_name = arango_config.get('memory_collection', 'memories')
        self.edges_collection_name = arango_config.get('memory_edges', 'memory_edges')
        
        # Initialize collections
        self._init_collections()
        
        # Track processing stats
        self.processed_count = 0
        self.error_count = 0
        
        self._running = False
        self._consumer_task = None
        
        logger.info("ArangoDB Consumer initialized")
    
    def _init_collections(self):
        """Create collections and indexes if they don't exist"""
        
        # Memories collection
        if not self.db.has_collection(self.memory_collection_name):
            self.memories = self.db.create_collection(self.memory_collection_name)
            logger.info(f"Created collection: {self.memory_collection_name}")
        else:
            self.memories = self.db.collection(self.memory_collection_name)
        
        # Edges collection
        if not self.db.has_collection(self.edges_collection_name):
            self.edges = self.db.create_collection(self.edges_collection_name, edge=True)
            logger.info(f"Created edge collection: {self.edges_collection_name}")
        else:
            self.edges = self.db.collection(self.edges_collection_name)
        
        # Create indexes
        try:
            # User ID index
            self.memories.add_persistent_index(
                fields=['user_id', 'created_at'],
                name='user_time_idx',
                unique=False
            )
            
            # Tier index
            self.memories.add_hash_index(
                fields=['tier'],
                name='tier_idx',
                unique=False
            )
            
            # Ego score index
            self.memories.add_persistent_index(
                fields=['ego_score'],
                name='ego_score_idx',
                unique=False
            )
            
            # Note: node_id uniqueness is enforced by ArangoDB's _key (set by Go service)
            # No need for additional version index since Go service doesn't use versioning
            
            logger.info("ArangoDB indexes created")
            
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")
    
    async def start(self):
        """Start consuming memory.upsert events"""
        self._running = True
        
        # Subscribe to memory.upsert topic
        await self.event_bus.subscribe(
            topic="memory.upsert",
            handler=self._handle_memory_upsert,
            consumer_group="arango-consumer"
        )
        
        logger.info("ArangoDB Consumer started")
    
    async def stop(self):
        """Stop consuming events"""
        self._running = False
        
        await self.event_bus.unsubscribe("memory.upsert")
        
        logger.info("ArangoDB Consumer stopped")
    
    async def _handle_memory_upsert(self, event: Event):
        """
        Handle memory.upsert event
        
        Writes memory node to ArangoDB with full metadata
        """
        
        try:
            payload = event.payload
            
            node_id = payload.get('node_id')
            version = payload.get('version', 1)
            
            # Check if this version already exists (idempotency)
            existing = self._get_existing_node(node_id)
            
            if existing and existing.get('version', 0) >= version:
                logger.debug(
                    f"Skipping older version: {node_id} v{version}",
                    extra={"node_id": node_id, "existing_version": existing.get('version')}
                )
                return
            
            # Build ArangoDB document
            doc = {
                "_key": node_id,
                "node_id": node_id,
                "user_id": payload.get('user_id'),
                "tier": payload.get('tier'),
                "summary": payload.get('summary'),
                "content": payload.get('content'),
                "ego_score": payload.get('ego_score'),
                "source": payload.get('source'),  # Add source field
                "components": payload.get('components', {}),
                "observed_at": payload.get('observed_at'),
                "created_at": payload.get('created_at'),
                "last_accessed_at": payload.get('last_accessed_at'),
                "last_reconsolidated_at": None,
                "sequence": payload.get('sequence'),
                "session_sequence": payload.get('session_sequence'),
                "timezone": payload.get('timezone'),
                "time_features": payload.get('time_features', {}),
                "metadata": payload.get('metadata', {}),
                "version": version,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Upsert document (idempotent)
            self.memories.insert(doc, overwrite=True, sync=True)
            
            self.processed_count += 1
            
            logger.info(
                f"Wrote memory to ArangoDB: {node_id}",
                extra={
                    "node_id": node_id,
                    "tier": payload.get('tier'),
                    "ego_score": payload.get('ego_score'),
                    "version": version
                }
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                f"Failed to write memory to ArangoDB: {e}",
                extra={"event_id": event.event_id},
                exc_info=True
            )
            raise
    
    def _get_existing_node(self, node_id: str) -> Dict[str, Any]:
        """Get existing memory node by ID"""
        
        try:
            return self.memories.get(node_id)
        except:
            return None
    
    async def create_edge(
        self,
        from_node: str,
        to_node: str,
        relationship: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Create edge between memory nodes
        
        Used for:
        - Parent-child relationships (consolidation)
        - Related memories
        - Contradiction relationships
        """
        
        edge = {
            "_from": f"{self.memory_collection_name}/{from_node}",
            "_to": f"{self.memory_collection_name}/{to_node}",
            "relationship": relationship,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        try:
            self.edges.insert(edge, sync=True)
            
            logger.debug(
                f"Created edge: {from_node} --{relationship}--> {to_node}",
                extra={"from": from_node, "to": to_node, "relationship": relationship}
            )
            
        except Exception as e:
            # Ignore duplicate edge errors
            if "unique constraint" not in str(e).lower():
                logger.error(f"Failed to create edge: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics"""
        
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.processed_count),
            "total_memories": self.memories.count()
        }

