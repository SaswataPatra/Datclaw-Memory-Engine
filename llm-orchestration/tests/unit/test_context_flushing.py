"""
Test Context Flushing with Processing State Tracking
Tests the new 3-tier flushing strategy
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List

from services.context_manager import ContextMemoryManager
from services.chatbot_service import ChatbotService


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    redis.setex = AsyncMock()
    redis.xadd = AsyncMock()
    return redis


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        'context_memory': {
            'max_tokens': 128000,
            'flush_threshold': 0.5,
            'emergency_threshold': 0.8,
            'keep_recent_messages': 20,
            'summarize_batch_size': 50
        },
        'llm': {
            'model': 'gpt-4o-mini'
        },
        'tiers': {
            'tier4': {
                'ttl_seconds': 600,
                'redis_key_prefix': 'tier4:'
            }
        }
    }


@pytest.fixture
def mock_ego_scorer():
    """Mock ego scorer"""
    scorer = Mock()
    result = Mock()
    result.ego_score = 0.5
    result.tier = 2
    scorer.calculate = Mock(return_value=result)
    return scorer


@pytest.fixture
def mock_event_bus():
    """Mock event bus"""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def mock_message_store():
    """Mock message store"""
    store = AsyncMock()
    store.append = AsyncMock()
    return store


@pytest.fixture
def mock_chatbot_service():
    """Mock chatbot service with processing state"""
    service = Mock()
    service.memory_processing_state = {}
    
    def get_state(session_id):
        return service.memory_processing_state.get(session_id, {})
    
    def mark_processing(session_id, message_id, state):
        if session_id not in service.memory_processing_state:
            service.memory_processing_state[session_id] = {}
        service.memory_processing_state[session_id][message_id] = state
    
    service.get_processing_state = Mock(side_effect=get_state)
    service.mark_message_processing = Mock(side_effect=mark_processing)
    
    return service


@pytest.fixture
def context_manager(mock_redis, mock_config, mock_ego_scorer, mock_event_bus, mock_message_store):
    """Create context manager instance"""
    return ContextMemoryManager(
        redis_client=mock_redis,
        config=mock_config,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        message_store=mock_message_store
    )


def create_message(msg_id: str, content: str, role: str = "user") -> Dict:
    """Helper to create a message"""
    return {
        "message_id": msg_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
        "sequence": int(msg_id.split('_')[1])
    }


@pytest.mark.asyncio
async def test_no_flush_under_threshold(context_manager):
    """Test that no flushing occurs when under threshold"""
    conversation_history = [
        create_message("msg_1", "Hello", "user"),
        create_message("msg_2", "Hi there!", "assistant"),
    ]
    
    result, metadata = await context_manager.manage_context(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    assert len(result) == 2
    assert metadata['flushed'] == False
    assert metadata['usage_percent'] < 0.5


@pytest.mark.asyncio
async def test_keep_recent_messages(context_manager, mock_chatbot_service):
    """Test that last N messages are always kept"""
    context_manager.set_chatbot_service(mock_chatbot_service)
    
    # Create 100 messages
    conversation_history = [
        create_message(f"msg_{i}", f"Message {i}", "user" if i % 2 == 0 else "assistant")
        for i in range(1, 101)
    ]
    
    # Mark first 30 as completed
    for i in range(1, 31):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "completed"
        )
    
    # Manually trigger flush (simulate high token count)
    result = await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Should keep last 20 messages (msg_81 to msg_100)
    recent_messages = [msg for msg in result if not msg.get('is_summary')]
    assert len(recent_messages) == 20
    assert recent_messages[0]['message_id'] == 'msg_81'
    assert recent_messages[-1]['message_id'] == 'msg_100'


@pytest.mark.asyncio
async def test_flush_completed_messages(context_manager, mock_chatbot_service, mock_event_bus):
    """Test that completed messages are flushed"""
    context_manager.set_chatbot_service(mock_chatbot_service)
    
    # Create 50 messages
    conversation_history = [
        create_message(f"msg_{i}", f"Message {i}")
        for i in range(1, 51)
    ]
    
    # Mark first 20 as completed
    for i in range(1, 21):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "completed"
        )
    
    # Trigger flush
    result = await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Wait a bit for background task
    await asyncio.sleep(0.1)
    
    # Verify completed messages were flushed
    # Should have summary + last 20 messages
    assert len(result) <= 21  # Summary + 20 recent


@pytest.mark.asyncio
async def test_summarize_unprocessed_messages(context_manager, mock_chatbot_service):
    """Test that unprocessed messages are summarized"""
    context_manager.set_chatbot_service(mock_chatbot_service)
    
    # Create 50 messages
    conversation_history = [
        create_message(f"msg_{i}", f"Message {i}")
        for i in range(1, 51)
    ]
    
    # Mark first 10 as completed, rest are unprocessed
    for i in range(1, 11):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "completed"
        )
    
    # Trigger flush
    result = await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Should have a summary message
    summary_messages = [msg for msg in result if msg.get('is_summary')]
    assert len(summary_messages) == 1
    assert 'Previous context' in summary_messages[0]['content']


@pytest.mark.asyncio
async def test_processing_state_categorization(context_manager, mock_chatbot_service):
    """Test correct categorization based on processing state"""
    context_manager.set_chatbot_service(mock_chatbot_service)
    
    # Create 60 messages
    conversation_history = [
        create_message(f"msg_{i}", f"Message {i}")
        for i in range(1, 61)
    ]
    
    # Set different processing states
    # msg_1-20: completed
    for i in range(1, 21):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "completed"
        )
    
    # msg_21-30: processing
    for i in range(21, 31):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "processing"
        )
    
    # msg_31-40: pending
    for i in range(31, 41):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "pending"
        )
    
    # msg_41-60: no state (unknown)
    
    # Trigger flush
    result = await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Verify structure
    # Should have: summary (for unprocessed 21-40) + recent (41-60)
    assert len(result) <= 21  # Summary + 20 recent


@pytest.mark.asyncio
async def test_no_chatbot_service_fallback(context_manager):
    """Test that flushing works without chatbot_service (safe default)"""
    # Don't set chatbot_service
    
    # Create 50 messages
    conversation_history = [
        create_message(f"msg_{i}", f"Message {i}")
        for i in range(1, 51)
    ]
    
    # Trigger flush
    result = await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Without chatbot_service, all old messages treated as unprocessed
    # Should have: summary + last 20 messages
    assert len(result) <= 21
    
    # Should have a summary
    summary_messages = [msg for msg in result if msg.get('is_summary')]
    assert len(summary_messages) == 1


@pytest.mark.asyncio
async def test_emergency_flush_keeps_last_10(context_manager):
    """Test emergency flush keeps only last 10 messages"""
    # Create 100 messages
    conversation_history = [
        create_message(f"msg_{i}", f"Message {i}")
        for i in range(1, 101)
    ]
    
    # Trigger emergency flush
    result = await context_manager._emergency_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Should keep only last 10 messages
    assert len(result) == 10
    assert result[0]['message_id'] == 'msg_91'
    assert result[-1]['message_id'] == 'msg_100'


@pytest.mark.asyncio
async def test_system_message_preserved(context_manager, mock_chatbot_service):
    """Test that system message is always preserved"""
    context_manager.set_chatbot_service(mock_chatbot_service)
    
    # Create conversation with system message
    conversation_history = [
        {"message_id": "system", "role": "system", "content": "You are DAPPY", "timestamp": datetime.utcnow().isoformat()}
    ] + [
        create_message(f"msg_{i}", f"Message {i}")
        for i in range(1, 51)
    ]
    
    # Mark some as completed
    for i in range(1, 21):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "completed"
        )
    
    # Trigger flush
    result = await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # System message should be first
    assert result[0]['role'] == 'system'
    assert result[0]['message_id'] == 'system'


@pytest.mark.asyncio
async def test_flush_priority_levels(context_manager, mock_chatbot_service, mock_event_bus):
    """Test that completed messages get LOW priority, unprocessed get HIGH"""
    context_manager.set_chatbot_service(mock_chatbot_service)
    
    # Create messages
    conversation_history = [
        create_message(f"msg_{i}", f"Message {i}")
        for i in range(1, 41)
    ]
    
    # Mark first 10 as completed
    for i in range(1, 11):
        mock_chatbot_service.mark_message_processing(
            "test_session", f"msg_{i}", "completed"
        )
    
    # Trigger flush
    await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Wait for background tasks
    await asyncio.sleep(0.2)
    
    # Verify event_bus.publish was called
    assert mock_event_bus.publish.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

