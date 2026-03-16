"""
DAPPY - Redis Message Bus Adapter
Hot tier message storage using Redis Streams
"""

import redis.asyncio as redis
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import logging

from core.messages.message_store import Message, MessageStore

logger = logging.getLogger(__name__)


class RedisMessageBus:
    """
    Hot message storage using Redis Streams
    
    Features:
    - 14-day retention by default
    - Fast timeline queries
    - Automatic TTL expiration
    - Event publishing for archiver
    """
    
    def __init__(self, redis_client, config: Dict[str, Any]):
        self.redis = redis_client
        self.retention_days = config.get('retention_days', 14)
        self.retention_seconds = self.retention_days * 24 * 3600
        self.stream_prefix = config.get('stream_prefix', 'messages:')
    
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
        """Append message to Redis Stream"""
        
        stream_key = f"{self.stream_prefix}{user_id}:{session_id}"
        
        data = {
            "message_id": message_id,
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "observed_at": observed_at.isoformat(),
            "sequence": str(sequence),
            "created_at": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        try:
            # Append to stream
            message_stream_id = await self.redis.xadd(stream_key, data)
            
            # Set TTL on stream
            await self.redis.expire(stream_key, self.retention_seconds)
            
            # Publish event for archiver
            await self.redis.publish(
                "messages.ingest",
                json.dumps({
                    "user_id": user_id,
                    "session_id": session_id,
                    "message_id": message_id,
                    "stream_key": stream_key
                })
            )
            
            logger.debug(
                f"Appended message to {stream_key}",
                extra={
                    "message_id": message_id,
                    "stream_id": message_stream_id,
                    "sequence": sequence
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to append message: {e}", exc_info=True)
            raise
    
    async def get_recent(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50
    ) -> List[Message]:
        """Read recent messages from Redis Stream (newest first)"""
        
        stream_key = f"{self.stream_prefix}{user_id}:{session_id}"
        
        try:
            # Read last N messages (XREVRANGE for newest first)
            messages = await self.redis.xrevrange(stream_key, count=limit)
            
            result = []
            for stream_id, msg_data in messages:
                try:
                    message = Message(
                        message_id=msg_data['message_id'],
                        user_id=msg_data['user_id'],
                        session_id=msg_data['session_id'],
                        role=msg_data['role'],
                        content=msg_data['content'],
                        observed_at=datetime.fromisoformat(msg_data['observed_at']),
                        sequence=int(msg_data['sequence']),
                        created_at=datetime.fromisoformat(msg_data['created_at']),
                        metadata=json.loads(msg_data.get('metadata', '{}'))
                    )
                    result.append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}", extra={"stream_id": stream_id})
                    continue
            
            return result
            
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
        """
        Get messages in time range
        
        Note: Redis Streams don't support time-based queries efficiently.
        This method reads all messages and filters by time.
        For large ranges, use long-term store instead.
        """
        stream_key = f"{self.stream_prefix}{user_id}:{session_id}"
        
        try:
            # Read all messages
            messages = await self.redis.xrange(stream_key)
            
            result = []
            for stream_id, msg_data in messages:
                try:
                    observed_at = datetime.fromisoformat(msg_data['observed_at'])
                    
                    # Filter by time range
                    if start_time <= observed_at <= end_time:
                        message = Message(
                            message_id=msg_data['message_id'],
                            user_id=msg_data['user_id'],
                            session_id=msg_data['session_id'],
                            role=msg_data['role'],
                            content=msg_data['content'],
                            observed_at=observed_at,
                            sequence=int(msg_data['sequence']),
                            created_at=datetime.fromisoformat(msg_data['created_at']),
                            metadata=json.loads(msg_data.get('metadata', '{}'))
                        )
                        result.append(message)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}")
                    continue
            
            # Sort by sequence
            result.sort(key=lambda m: m.sequence)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get message range: {e}", exc_info=True)
            return []
    
    async def link_to_memory(
        self,
        message_id: str,
        memory_node_id: str
    ) -> None:
        """
        Link message to memory node
        
        Note: Provenance edges are maintained in ArangoDB, not Redis.
        This is a no-op for Redis adapter.
        """
        # Redis doesn't maintain provenance edges
        # This is handled by ArangoMessageStore
        pass
    
    async def get_stream_info(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get stream statistics"""
        stream_key = f"{self.stream_prefix}{user_id}:{session_id}"
        
        try:
            info = await self.redis.xinfo_stream(stream_key)
            return {
                "length": info.get('length', 0),
                "first_entry": info.get('first-entry'),
                "last_entry": info.get('last-entry'),
                "stream_key": stream_key
            }
        except Exception as e:
            logger.warning(f"Failed to get stream info: {e}")
            return {"length": 0, "stream_key": stream_key}

