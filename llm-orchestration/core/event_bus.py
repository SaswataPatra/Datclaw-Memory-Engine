"""
DAPPY LLM Orchestration Service - Event Bus Abstraction
Decoupled event-driven architecture with Redis Pub/Sub (Phase 1) and Kafka-ready (Phase 3)
"""

from typing import Protocol, Dict, Any, Callable, Awaitable, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import asyncio
from abc import abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """
    Standard event schema for DAPPY
    
    All events flowing through the system use this structure
    """
    topic: str
    event_type: str
    payload: Dict[str, Any]
    event_id: Optional[str] = None
    timestamp: Optional[str] = None
    source_service: str = "llm-orchestrator"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.event_id is None:
            import uuid
            self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Deserialize from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))


class EventBus(Protocol):
    """
    Abstract Event Bus interface
    
    Implementations: RedisEventBus (Phase 1), KafkaEventBus (Phase 3)
    """
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus"""
        ...
    
    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Awaitable[None]],
        consumer_group: Optional[str] = None
    ) -> None:
        """Subscribe to a topic and process events with handler"""
        ...
    
    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic"""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close all connections and cleanup"""
        ...


class RedisEventBus:
    """
    Redis-based Event Bus using Pub/Sub and Streams
    
    Features:
    - Pub/Sub for real-time notifications
    - Streams for reliable delivery with consumer groups
    - Automatic reconnection
    - Dead letter queue for failed messages
    """
    
    def __init__(self, redis_client, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.stream_prefix = config.get('stream_prefix', 'events:')
        self.consumer_group = config.get('consumer_group', 'llm-orchestrator')
        self.max_len = config.get('max_len', 10000)
        
        # Track active subscriptions
        self._subscriptions: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def publish(self, event: Event) -> None:
        """
        Publish event to Redis Stream
        
        Uses XADD with MAXLEN to limit stream size
        """
        stream_key = f"{self.stream_prefix}{event.topic}"
        
        try:
            # Serialize the entire event to a single JSON string
            # This is simpler and avoids Redis XADD dict serialization issues
            event_json = event.to_json()
            
            # Add to stream with max length limit
            message_id = await self.redis.xadd(
                stream_key,
                {"event": event_json},  # Single field with JSON string
                maxlen=self.max_len,
                approximate=True
            )
            
            logger.debug(
                f"Published event to {stream_key}",
                extra={
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "message_id": message_id
                }
            )
            
            # Also publish to Pub/Sub for real-time listeners
            await self.redis.publish(event.topic, event.to_json())
            
        except Exception as e:
            logger.error(
                f"Failed to publish event to {stream_key}: {e}",
                extra={"event": event.to_dict()}
            )
            raise
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Awaitable[None]],
        consumer_group: Optional[str] = None
    ) -> None:
        """
        Subscribe to topic using Redis Streams with consumer groups
        
        Ensures at-least-once delivery and load balancing
        """
        stream_key = f"{self.stream_prefix}{topic}"
        group_name = consumer_group or self.consumer_group
        consumer_name = f"{group_name}-{id(handler)}"
        
        # Create consumer group if not exists
        try:
            await self.redis.xgroup_create(
                stream_key,
                group_name,
                id='0',
                mkstream=True
            )
            logger.info(f"Created consumer group {group_name} for {stream_key}")
        except Exception as e:
            # Group already exists
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Failed to create consumer group: {e}")
        
        # Start consumer task
        task = asyncio.create_task(
            self._consume_stream(stream_key, group_name, consumer_name, handler)
        )
        
        self._subscriptions[topic] = task
        self._running = True
        
        logger.info(
            f"Subscribed to {topic}",
            extra={"stream_key": stream_key, "consumer_group": group_name}
        )
    
    async def _consume_stream(
        self,
        stream_key: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[Event], Awaitable[None]]
    ):
        """
        Consume messages from Redis Stream
        
        Processes messages in batches and ACKs successfully processed ones
        """
        logger.info(f"Consumer {consumer_name} started for {stream_key}")
        
        while self._running:
            try:
                # Read from stream (blocking with timeout)
                messages = await self.redis.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={stream_key: '>'},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                if not messages:
                    continue
                
                # Process each message
                for stream, message_list in messages:
                    for message_id, message_data in message_list:
                        try:
                            # Deserialize event from JSON string
                            event_json = message_data.get(b'event') or message_data.get('event')
                            if isinstance(event_json, bytes):
                                event_json = event_json.decode('utf-8')
                            event_dict = json.loads(event_json)
                            event = Event.from_dict(event_dict)
                            
                            # Call handler
                            await handler(event)
                            
                            # ACK message
                            await self.redis.xack(stream_key, group_name, message_id)
                            
                            logger.debug(
                                f"Processed event {event.event_id}",
                                extra={"message_id": message_id}
                            )
                            
                        except Exception as e:
                            logger.error(
                                f"Error processing message {message_id}: {e}",
                                extra={"message_data": message_data},
                                exc_info=True
                            )
                            
                            # Move to dead letter queue
                            await self._move_to_dlq(stream_key, message_id, message_data, str(e))
                            
                            # ACK to remove from pending
                            await self.redis.xack(stream_key, group_name, message_id)
            
            except asyncio.CancelledError:
                logger.info(f"Consumer {consumer_name} cancelled")
                break
            
            except Exception as e:
                logger.error(f"Consumer error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Backoff before retry
        
        logger.info(f"Consumer {consumer_name} stopped")
    
    async def _move_to_dlq(
        self,
        stream_key: str,
        message_id: str,
        message_data: Dict[str, Any],
        error: str
    ):
        """Move failed message to dead letter queue"""
        dlq_key = f"{stream_key}:dlq"
        
        dlq_entry = {
            **message_data,
            "original_message_id": message_id,
            "error": error,
            "failed_at": datetime.utcnow().isoformat()
        }
        
        await self.redis.xadd(dlq_key, dlq_entry, maxlen=1000)
        
        logger.warning(
            f"Moved message to DLQ: {dlq_key}",
            extra={"message_id": message_id, "error": error}
        )
    
    async def wait_for_event(
        self,
        topic: str,
        filter_fn: Callable[[Event], bool],
        timeout: float = 5.0
    ) -> Optional[Event]:
        """
        Wait for a specific event matching the filter function.
        
        This is useful for synchronous operations that need to wait for
        async event completion (e.g., waiting for memory indexing).
        
        Args:
            topic: Event topic to listen to
            filter_fn: Function to filter events (returns True for matching event)
            timeout: Maximum time to wait in seconds
        
        Returns:
            Matching Event or None if timeout
        """
        event_future = asyncio.Future()
        
        async def handler(event: Event):
            if filter_fn(event):
                if not event_future.done():
                    event_future.set_result(event)
        
        # Subscribe to topic using Pub/Sub for real-time notification
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(topic)
        
        try:
            # Wait for matching event with timeout
            async def listen():
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            event_json = message['data']
                            if isinstance(event_json, bytes):
                                event_json = event_json.decode('utf-8')
                            event_dict = json.loads(event_json)
                            event = Event.from_dict(event_dict)
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Error parsing event: {e}")
            
            listen_task = asyncio.create_task(listen())
            
            try:
                result = await asyncio.wait_for(event_future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                logger.debug(f"Timeout waiting for event on topic {topic}")
                return None
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
        
        finally:
            await pubsub.unsubscribe(topic)
            await pubsub.close()
    
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from topic"""
        if topic in self._subscriptions:
            task = self._subscriptions[topic]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._subscriptions[topic]
            logger.info(f"Unsubscribed from {topic}")
    
    async def close(self) -> None:
        """Close all subscriptions"""
        self._running = False
        
        # Cancel all subscription tasks
        for topic in list(self._subscriptions.keys()):
            await self.unsubscribe(topic)
        
        logger.info("Event bus closed")


class KafkaEventBus:
    """
    Kafka-based Event Bus (Phase 3)
    
    Placeholder for future Kafka implementation
    Same interface as RedisEventBus for zero-code migration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        raise NotImplementedError("Kafka Event Bus will be implemented in Phase 3")
    
    async def publish(self, event: Event) -> None:
        raise NotImplementedError()
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Awaitable[None]],
        consumer_group: Optional[str] = None
    ) -> None:
        raise NotImplementedError()
    
    async def unsubscribe(self, topic: str) -> None:
        raise NotImplementedError()
    
    async def close(self) -> None:
        raise NotImplementedError()


class EventBusFactory:
    """Factory for creating Event Bus instances based on config"""
    
    @staticmethod
    def create(provider: str, redis_client=None, config: Dict[str, Any] = None) -> EventBus:
        """
        Create Event Bus instance
        
        Args:
            provider: 'redis' or 'kafka'
            redis_client: Redis client instance (required for redis provider)
            config: Provider-specific configuration
        
        Returns:
            EventBus implementation
        """
        config = config or {}
        
        if provider == 'redis':
            if redis_client is None:
                raise ValueError("redis_client is required for Redis Event Bus")
            return RedisEventBus(redis_client, config)
        
        elif provider == 'kafka':
            return KafkaEventBus(config)
        
        else:
            raise ValueError(f"Unknown event bus provider: {provider}")

