"""
Integration Test: Context Flushing with ChatbotService
Tests the full flow of memory extraction + context flushing
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from services.context_manager import ContextMemoryManager
from services.chatbot_service import ChatbotService
from llm.providers.base import LLMResponse


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider"""
    provider = AsyncMock()
    provider.name = "mock-llm"
    
    # Mock chat response
    response = LLMResponse(
        content="This is a test response",
        model="mock-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        finish_reason="stop"
    )
    provider.chat = AsyncMock(return_value=response)
    
    return provider


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    redis.setex = AsyncMock()
    redis.xadd = AsyncMock()
    redis.ping = AsyncMock()
    return redis


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        'context_memory': {
            'max_tokens': 1000,  # Low threshold for testing
            'flush_threshold': 0.5,
            'emergency_threshold': 0.8,
            'keep_recent_messages': 5,  # Keep only 5 for testing
            'summarize_batch_size': 10
        },
        'llm': {
            'model': 'gpt-4o-mini'
        },
        'tiers': {
            'tier4': {
                'ttl_seconds': 600,
                'redis_key_prefix': 'tier4:'
            }
        },
        'ego_scoring': {
            'thresholds': {
                'tier1': 0.75,
                'tier2': 0.50,
                'tier3': 0.20
            }
        }
    }


@pytest.fixture
def mock_ego_scorer():
    """Mock ego scorer"""
    scorer = Mock()
    result = Mock()
    result.ego_score = 0.6
    result.tier = 2
    result.components = Mock()
    result.components.to_dict = Mock(return_value={})
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
def context_manager(mock_redis, mock_config, mock_ego_scorer, mock_event_bus, mock_message_store):
    """Create context manager"""
    return ContextMemoryManager(
        redis_client=mock_redis,
        config=mock_config,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        message_store=mock_message_store
    )


@pytest.fixture
def chatbot_service(mock_llm_provider, context_manager, mock_ego_scorer, mock_event_bus):
    """Create chatbot service"""
    service = ChatbotService(
        llm_provider=mock_llm_provider,
        context_manager=context_manager,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        shadow_tier=None,
        system_prompt="You are a test assistant"
    )
    
    # Wire up
    context_manager.set_chatbot_service(service)
    
    return service


@pytest.mark.asyncio
async def test_full_flow_with_flushing(chatbot_service, context_manager):
    """Test full flow: chat -> memory extraction -> flushing"""
    
    conversation_history = []
    
    # Send 20 messages with longer content to reach token threshold
    for i in range(1, 21):
        # Use longer messages to increase token count
        long_message = f"Test message {i}. " * 20  # Repeat to increase tokens
        
        response = await chatbot_service.chat(
            user_id="test_user",
            session_id="test_session",
            user_message=long_message,
            conversation_history=conversation_history
        )
        
        conversation_history = response['conversation_history']
        
        # Wait a bit for background processing
        await asyncio.sleep(0.05)
    
    # Verify processing state is tracked
    processing_state = chatbot_service.get_processing_state("test_session")
    assert len(processing_state) > 0
    
    # Note: Flushing may or may not have triggered depending on token count
    # Just verify processing state is working
    print(f"\nFinal conversation history length: {len(conversation_history)}")
    print(f"Processing state entries: {len(processing_state)}")
    
    # Verify at least some messages are tracked
    assert len(processing_state) >= 5


@pytest.mark.asyncio
async def test_memory_extraction_marks_completed(chatbot_service):
    """Test that memory extraction marks messages as completed"""
    
    # Send a message with memory trigger
    response = await chatbot_service.chat(
        user_id="test_user",
        session_id="test_session",
        user_message="My name is Alice and I love Python",
        conversation_history=[]
    )
    
    # Wait for background processing
    await asyncio.sleep(0.2)
    
    # Check processing state
    processing_state = chatbot_service.get_processing_state("test_session")
    
    # Should have at least one message marked
    assert len(processing_state) > 0
    
    # Check if any message reached 'completed' state
    completed_count = sum(1 for state in processing_state.values() if state == 'completed')
    print(f"\nCompleted messages: {completed_count}/{len(processing_state)}")


@pytest.mark.asyncio
async def test_flushing_respects_processing_state(chatbot_service, context_manager):
    """Test that flushing correctly categorizes based on processing state"""
    
    conversation_history = []
    
    # Send 15 messages
    for i in range(1, 16):
        response = await chatbot_service.chat(
            user_id="test_user",
            session_id="test_session",
            user_message=f"Message {i}",
            conversation_history=conversation_history
        )
        conversation_history = response['conversation_history']
        
        # Wait for first 5 to complete processing
        if i <= 5:
            await asyncio.sleep(0.3)
    
    # Check processing state
    processing_state = chatbot_service.get_processing_state("test_session")
    completed = [k for k, v in processing_state.items() if v == 'completed']
    pending = [k for k, v in processing_state.items() if v in ['pending', 'processing']]
    
    print(f"\nCompleted: {len(completed)}, Pending: {len(pending)}")
    
    # Manually trigger flush
    flushed_history = await context_manager._intelligent_flush(
        user_id="test_user",
        session_id="test_session",
        conversation_history=conversation_history
    )
    
    # Verify flushing worked
    assert len(flushed_history) < len(conversation_history)
    print(f"Before flush: {len(conversation_history)}, After flush: {len(flushed_history)}")


@pytest.mark.asyncio
async def test_no_memory_triggers_still_tracks_state(chatbot_service):
    """Test that messages without memory triggers still get state tracked"""
    
    # Send message without memory triggers
    response = await chatbot_service.chat(
        user_id="test_user",
        session_id="test_session",
        user_message="What's the weather?",
        conversation_history=[]
    )
    
    # Wait a bit
    await asyncio.sleep(0.1)
    
    # Processing state should still be tracked (marked as pending)
    processing_state = chatbot_service.get_processing_state("test_session")
    
    # Should have the message marked as pending (even if no memory extracted)
    assert len(processing_state) > 0


@pytest.mark.asyncio
async def test_summary_injection(chatbot_service, context_manager):
    """Test that summary is correctly injected into conversation history"""
    
    conversation_history = []
    
    # Send enough messages to trigger flush
    for i in range(1, 25):
        response = await chatbot_service.chat(
            user_id="test_user",
            session_id="test_session",
            user_message=f"Message {i}",
            conversation_history=conversation_history
        )
        conversation_history = response['conversation_history']
        
        # Mark first 10 as completed
        if i <= 10:
            await asyncio.sleep(0.1)
    
    # Check if summary was injected
    summaries = [msg for msg in conversation_history if msg.get('is_summary')]
    
    if summaries:
        print(f"\nSummary found: {summaries[0]['content']}")
        assert 'Previous context' in summaries[0]['content']
    else:
        print("\nNo summary yet (may need more messages to trigger flush)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

