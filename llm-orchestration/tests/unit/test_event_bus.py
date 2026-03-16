"""
Unit tests for Event Bus

Tests Redis event bus implementation, Kafka-ready abstraction,
and event handling.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.event_bus import Event, RedisEventBus, EventBusFactory


class TestEvent:
    """Test Event dataclass"""
    
    def test_event_creation(self):
        """Test creating an event"""
        event = Event(
            topic="memory.updates",
            event_type="memory.upsert",
            payload={"node_id": "mem123", "tier": 1},
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id="evt123"
        )
        
        assert event.topic == "memory.updates"
        assert event.event_type == "memory.upsert"
        assert event.payload["node_id"] == "mem123"
        assert event.event_id == "evt123"
    
    def test_event_to_dict(self):
        """Test event serialization"""
        event = Event(
            topic="memory.updates",
            event_type="memory.upsert",
            payload={"node_id": "mem123"},
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id="evt123"
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["topic"] == "memory.updates"
        assert event_dict["event_type"] == "memory.upsert"
        assert event_dict["payload"]["node_id"] == "mem123"
        assert "timestamp" in event_dict
    
    def test_event_from_dict(self):
        """Test event deserialization"""
        data = {
            "topic": "memory.updates",
            "event_type": "memory.upsert",
            "payload": {"node_id": "mem123"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": "evt123"
        }
        
        event = Event.from_dict(data)
        
        assert event.topic == "memory.updates"
        assert event.event_type == "memory.upsert"
        assert event.payload["node_id"] == "mem123"
        assert event.event_id == "evt123"


class TestRedisEventBus:
    """Test suite for Redis Event Bus"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis = AsyncMock()
        redis.xadd = AsyncMock(return_value=b"1234567890-0")
        redis.xread = AsyncMock(return_value=[])
        redis.xreadgroup = AsyncMock(return_value=[])
        redis.xack = AsyncMock()
        redis.xdel = AsyncMock()
        redis.xgroup_create = AsyncMock()
        redis.publish = AsyncMock()
        redis.ping = AsyncMock(return_value=True)
        return redis
    
    @pytest.fixture
    def config(self):
        """Event bus configuration"""
        return {
            'stream_prefix': 'events:',
            'consumer_group': 'test-group',
            'max_len': 10000
        }
    
    @pytest.fixture
    def event_bus(self, mock_redis, config):
        """Create event bus instance"""
        return RedisEventBus(mock_redis, config)
    
    def test_initialization(self, event_bus, config):
        """Test event bus initialization"""
        assert event_bus.stream_prefix == 'events:'
        assert event_bus.consumer_group == 'test-group'
        assert event_bus.max_len == 10000
        assert event_bus._running == False
        assert event_bus._subscriptions == {}
    
    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus, mock_redis):
        """Test publishing an event"""
        event = Event(
            topic="memory.updates",
            event_type="memory.upsert",
            payload={"node_id": "mem123"},
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id="evt123"
        )
        
        await event_bus.publish(event)
        
        # Should call xadd
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "events:memory.updates"
        
        # Should also publish to Pub/Sub
        mock_redis.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_with_custom_topic(self, event_bus, mock_redis):
        """Test publishing to custom topic"""
        event = Event(
            topic="tier.promoted",
            event_type="tier.change",
            payload={"node_id": "mem456", "to_tier": 1},
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_id="evt456"
        )
        
        await event_bus.publish(event)
        
        # Should use correct stream key
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "events:tier.promoted"
    
    @pytest.mark.asyncio
    async def test_subscribe_creates_consumer_group(self, event_bus, mock_redis):
        """Test that subscribe creates consumer group"""
        async def test_handler(event):
            pass
        
        # Start subscribe (but don't await the consumer task)
        await event_bus.subscribe("memory", test_handler)
        
        # Should create consumer group
        mock_redis.xgroup_create.assert_called_once()
        call_args = mock_redis.xgroup_create.call_args
        assert call_args[0][0] == "events:memory"
        assert call_args[0][1] == "test-group"
        
        # Should track subscription
        assert "memory" in event_bus._subscriptions
        assert event_bus._running == True
        
        # Cleanup: cancel the task
        event_bus._subscriptions["memory"].cancel()
        try:
            await event_bus._subscriptions["memory"]
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_subscribe_handles_existing_group(self, event_bus, mock_redis):
        """Test subscribe handles existing consumer group"""
        # Mock group already exists error
        mock_redis.xgroup_create = AsyncMock(
            side_effect=Exception("BUSYGROUP Consumer Group name already exists")
        )
        
        async def test_handler(event):
            pass
        
        # Should not raise exception
        await event_bus.subscribe("memory", test_handler)
        
        assert "memory" in event_bus._subscriptions
        
        # Cleanup
        event_bus._subscriptions["memory"].cancel()
        try:
            await event_bus._subscriptions["memory"]
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus, mock_redis):
        """Test unsubscribing from topic"""
        async def test_handler(event):
            pass
        
        # Subscribe first
        await event_bus.subscribe("memory", test_handler)
        assert "memory" in event_bus._subscriptions
        
        # Unsubscribe
        await event_bus.unsubscribe("memory")
        
        # Should remove subscription
        assert "memory" not in event_bus._subscriptions
    
    @pytest.mark.asyncio
    async def test_close(self, event_bus, mock_redis):
        """Test closing event bus"""
        async def test_handler(event):
            pass
        
        # Subscribe to multiple topics
        await event_bus.subscribe("memory", test_handler)
        await event_bus.subscribe("tier", test_handler)
        
        assert len(event_bus._subscriptions) == 2
        
        # Close
        await event_bus.close()
        
        # Should cancel all subscriptions
        assert len(event_bus._subscriptions) == 0
        assert event_bus._running == False
    
    @pytest.mark.asyncio
    async def test_consume_stream_processes_messages(self, event_bus, mock_redis):
        """Test that consumer processes messages"""
        # Track handler calls
        handler_calls = []
        
        async def test_handler(event):
            handler_calls.append(event)
        
        # Mock xreadgroup to return a message once, then raise CancelledError
        message_data = {
            'topic': 'memory.updates',
            'event_type': 'memory.upsert',
            'payload': {'node_id': 'mem123'},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_id': 'evt123'
        }
        
        call_count = [0]
        async def mock_xreadgroup_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [('events:memory', [('1234567890-0', message_data)])]
            # After first call, raise CancelledError to stop loop
            raise asyncio.CancelledError()
        
        mock_redis.xreadgroup = AsyncMock(side_effect=mock_xreadgroup_side_effect)
        
        # Start consumer
        await event_bus.subscribe("memory", test_handler)
        
        # Give it time to process
        await asyncio.sleep(0.2)
        
        # Stop consumer
        await event_bus.close()
        
        # Should have processed the message
        assert len(handler_calls) == 1
        assert handler_calls[0].event_id == 'evt123'
        
        # Should have ACKed the message
        mock_redis.xack.assert_called()
    
    @pytest.mark.asyncio
    async def test_consume_stream_handles_errors(self, event_bus, mock_redis):
        """Test consumer handles handler errors"""
        async def failing_handler(event):
            raise Exception("Handler error")
        
        # Mock xreadgroup to return a message once, then raise CancelledError
        message_data = {
            'topic': 'memory.updates',
            'event_type': 'memory.upsert',
            'payload': {'node_id': 'mem123'},
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_id': 'evt123'
        }
        
        call_count = [0]
        async def mock_xreadgroup_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [('events:memory', [('1234567890-0', message_data)])]
            raise asyncio.CancelledError()
        
        mock_redis.xreadgroup = AsyncMock(side_effect=mock_xreadgroup_side_effect)
        
        # Start consumer
        await event_bus.subscribe("memory", failing_handler)
        
        # Give it time to process
        await asyncio.sleep(0.2)
        
        # Stop consumer
        await event_bus.close()
        
        # Should still ACK the message (after moving to DLQ)
        assert mock_redis.xack.called
        
        # Should have added to DLQ
        assert mock_redis.xadd.call_count >= 1  # DLQ entry


class TestEventBusFactory:
    """Test EventBusFactory"""
    
    def test_create_redis_bus(self):
        """Test creating Redis event bus"""
        mock_redis = AsyncMock()
        config = {'stream_prefix': 'events:'}
        
        bus = EventBusFactory.create('redis', redis_client=mock_redis, config=config)
        
        assert isinstance(bus, RedisEventBus)
        assert bus.redis == mock_redis
    
    def test_create_kafka_bus_not_implemented(self):
        """Test Kafka bus raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            EventBusFactory.create('kafka')
    
    def test_create_unknown_provider(self):
        """Test unknown provider raises ValueError"""
        with pytest.raises(ValueError, match="Unknown event bus provider"):
            EventBusFactory.create('unknown')


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
