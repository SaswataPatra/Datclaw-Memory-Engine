"""
DAPPY - ArangoDB Message Store Adapter
Long-term message storage with provenance linking
"""

from arango import ArangoClient
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

from core.messages.message_store import Message, MessageStore

logger = logging.getLogger(__name__)


class ArangoMessageStore:
    """
    Long-term message storage in ArangoDB
    
    Features:
    - Persistent storage for archived messages
    - Provenance edges linking messages to memory nodes
    - Time-based indexing for efficient queries
    - Sequence indexing for ordered retrieval
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Connect to ArangoDB
        client = ArangoClient(hosts=config['url'])
        self.db = client.db(
            config['database'],
            username=config['username'],
            password=config['password']
        )
        
        self.message_collection_name = config.get('collection', 'messages')
        self.edge_collection_name = config.get('edges', 'message_to_memory')
        
        # Initialize collections and indexes
        self._init_collections()
    
    def _init_collections(self):
        """Create collections and indexes if they don't exist"""
        
        # Messages collection
        if not self.db.has_collection(self.message_collection_name):
            self.messages = self.db.create_collection(self.message_collection_name)
            logger.info(f"Created collection: {self.message_collection_name}")
        else:
            self.messages = self.db.collection(self.message_collection_name)
        
        # Provenance edges collection
        if not self.db.has_collection(self.edge_collection_name):
            self.edges = self.db.create_collection(self.edge_collection_name, edge=True)
            logger.info(f"Created edge collection: {self.edge_collection_name}")
        else:
            self.edges = self.db.collection(self.edge_collection_name)
        
        # Create indexes for efficient queries
        try:
            # Composite index for timeline queries
            self.messages.add_persistent_index(
                fields=['user_id', 'session_id', 'created_at'],
                name='user_session_time_idx',
                unique=False
            )
            
            # Sequence index for ordered retrieval
            self.messages.add_persistent_index(
                fields=['user_id', 'sequence'],
                name='user_sequence_idx',
                unique=False
            )
            
            # Message ID index (for lookups and provenance)
            self.messages.add_hash_index(
                fields=['message_id'],
                name='message_id_idx',
                unique=True
            )
            
            logger.info("Message store indexes created")
            
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")
    
    async def append(
        self,
        message_id: str,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        observed_at: datetime,
        sequence: int,
        metadata: Optional[Dict] = None
    ) -> None:
        """Append message to ArangoDB (called by archiver)"""
        
        doc = {
            "_key": message_id,
            "message_id": message_id,
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
            "observed_at": observed_at.isoformat(),
            "sequence": sequence,
            "metadata": metadata or {},
            "archived": False,
            "s3_key": None
        }
        
        try:
            # Upsert (idempotent)
            self.messages.insert(doc, overwrite=True, sync=True)
            
            logger.debug(
                f"Archived message to ArangoDB",
                extra={"message_id": message_id, "user_id": user_id}
            )
            
        except Exception as e:
            logger.error(f"Failed to archive message: {e}", exc_info=True)
            raise
    
    async def archive_message(self, message: Dict[str, Any]):
        """
        Archive a message from Redis (called by worker)
        
        Accepts dict to avoid conversion overhead
        """
        await self.append(
            message_id=message['message_id'],
            user_id=message['user_id'],
            session_id=message['session_id'],
            role=message['role'],
            content=message['content'],
            observed_at=datetime.fromisoformat(message['observed_at']),
            sequence=int(message['sequence']),
            metadata=message.get('metadata', {})
        )
    
    async def get_recent(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50
    ) -> List[Message]:
        """Get recent messages for a session"""
        
        aql = """
        FOR msg IN @@collection
            FILTER msg.user_id == @user_id
            FILTER msg.session_id == @session_id
            SORT msg.created_at DESC
            LIMIT @limit
            RETURN msg
        """
        
        try:
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "@collection": self.message_collection_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "limit": limit
                }
            )
            
            messages = []
            for doc in cursor:
                messages.append(Message(
                    message_id=doc['message_id'],
                    user_id=doc['user_id'],
                    session_id=doc['session_id'],
                    role=doc['role'],
                    content=doc['content'],
                    observed_at=datetime.fromisoformat(doc['observed_at']),
                    sequence=doc['sequence'],
                    created_at=datetime.fromisoformat(doc['created_at']),
                    metadata=doc.get('metadata', {})
                ))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}", exc_info=True)
            return []
    
    async def get_range(
        self,
        user_id: str,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Message]:
        """Query messages by time range"""
        
        aql = """
        FOR msg IN @@collection
            FILTER msg.user_id == @user_id
            FILTER msg.session_id == @session_id
            FILTER msg.created_at >= @start
            FILTER msg.created_at <= @end
            SORT msg.sequence ASC
            RETURN msg
        """
        
        try:
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "@collection": self.message_collection_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            )
            
            messages = []
            for doc in cursor:
                # Check if archived to S3
                if doc.get('archived') and doc.get('s3_key'):
                    # TODO: Return S3 signed URL or lazy load
                    logger.debug(f"Message {doc['message_id']} is archived to S3: {doc['s3_key']}")
                
                messages.append(Message(
                    message_id=doc['message_id'],
                    user_id=doc['user_id'],
                    session_id=doc['session_id'],
                    role=doc['role'],
                    content=doc['content'],
                    observed_at=datetime.fromisoformat(doc['observed_at']),
                    sequence=doc['sequence'],
                    created_at=datetime.fromisoformat(doc['created_at']),
                    metadata=doc.get('metadata', {})
                ))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get message range: {e}", exc_info=True)
            return []
    
    async def link_to_memory(
        self,
        message_id: str,
        memory_node_id: str
    ) -> None:
        """Create provenance edge linking message to memory node"""
        
        edge = {
            "_from": f"{self.message_collection_name}/{message_id}",
            "_to": f"memories/{memory_node_id}",
            "relationship": "SOURCE_OF",
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            self.edges.insert(edge, sync=True)
            
            logger.debug(
                f"Linked message {message_id} to memory {memory_node_id}",
                extra={"message_id": message_id, "memory_node_id": memory_node_id}
            )
            
        except Exception as e:
            # Ignore duplicate edge errors
            if "unique constraint" not in str(e).lower():
                logger.error(f"Failed to create provenance edge: {e}", exc_info=True)
    
    async def get_message_sources(self, memory_node_id: str) -> List[Message]:
        """Get all messages that contributed to a memory node"""
        
        aql = """
        FOR edge IN @@edges
            FILTER edge._to == @memory_node
            FOR msg IN @@messages
                FILTER msg._id == edge._from
                RETURN msg
        """
        
        try:
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "@edges": self.edge_collection_name,
                    "@messages": self.message_collection_name,
                    "memory_node": f"memories/{memory_node_id}"
                }
            )
            
            messages = []
            for doc in cursor:
                messages.append(Message(
                    message_id=doc['message_id'],
                    user_id=doc['user_id'],
                    session_id=doc['session_id'],
                    role=doc['role'],
                    content=doc['content'],
                    observed_at=datetime.fromisoformat(doc['observed_at']),
                    sequence=doc['sequence'],
                    created_at=datetime.fromisoformat(doc['created_at']),
                    metadata=doc.get('metadata', {})
                ))
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get message sources: {e}", exc_info=True)
            return []
    
    async def mark_as_archived(self, message_id: str, s3_key: str) -> None:
        """Mark message as archived to S3"""
        
        try:
            self.messages.update({
                "_key": message_id,
                "archived": True,
                "s3_key": s3_key,
                "archived_at": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Marked message {message_id} as archived to {s3_key}")
            
        except Exception as e:
            logger.error(f"Failed to mark message as archived: {e}", exc_info=True)

