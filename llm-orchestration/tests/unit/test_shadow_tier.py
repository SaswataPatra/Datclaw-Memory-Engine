"""
Unit tests for Shadow Tier Manager

Tests shadow tier workflow, user confirmation, auto-promotion,
and clarification generation.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import uuid

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.shadow_tier import ShadowTier


class TestShadowTierManager:
    """Test suite for Shadow Tier Manager"""
    
    @pytest.fixture
    def mock_arango(self):
        """Mock ArangoDB client"""
        arango = Mock()
        collection = AsyncMock()
        collection.insert = AsyncMock()
        collection.get = AsyncMock()
        collection.delete = AsyncMock()
        
        aql = AsyncMock()
        aql.execute = AsyncMock(return_value=[])
        
        arango.collection = Mock(return_value=collection)
        arango.aql = aql
        
        return arango
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis = AsyncMock()
        redis.setex = AsyncMock()
        redis.get = AsyncMock()
        redis.delete = AsyncMock()
        redis.sadd = AsyncMock()
        redis.srem = AsyncMock()
        redis.smembers = AsyncMock(return_value=[])
        
        # Mock scan_iter to return an async iterator
        async def async_iter_mock():
            return
            yield  # Make it an async generator
        
        redis.scan_iter = Mock(return_value=async_iter_mock())
        return redis
    
    @pytest.fixture
    def mock_event_bus(self):
        """Mock event bus"""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client"""
        llm = AsyncMock()
        
        # Mock chat completions
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "Is this information correct?"
        
        llm.chat.completions.create = AsyncMock(return_value=response)
        
        return llm
    
    @pytest.fixture
    def config(self):
        """Shadow tier configuration"""
        return {
            'shadow_tier_days': 7,
            'ego_threshold': 0.75,
            'confidence_threshold': 0.85
        }
    
    @pytest.fixture
    def manager(self, mock_arango, mock_redis, mock_event_bus, mock_llm, config):
        """Create shadow tier manager instance"""
        mgr = ShadowTier(
            redis_client=mock_redis,
            config=config,
            event_bus=mock_event_bus
        )
        mgr.arango = mock_arango
        mgr.llm = mock_llm
        return mgr
    
    def test_initialization(self, manager, config):
        """Test shadow tier manager initialization"""
        assert manager.config == config
        assert manager.auto_promote_days == 7
    
    @pytest.mark.asyncio
    async def test_propose_core_memory(self, manager, mock_arango, mock_redis, mock_llm):
        """Test proposing a shadow tier memory"""
        memory = {
            'node_id': 'mem123',
            'summary': 'User prefers green tea',
            'ego_score': 0.80,
            'confidence': 0.75,
            'sources': [{'text': 'I prefer green tea now', 'timestamp': datetime.now().isoformat()}]
        }
        
        clarification_id, question = await manager.propose_core_memory(memory)
        
        # Should create shadow tier entry
        shadow_collection = mock_arango.collection('shadow_memories')
        shadow_collection.insert.assert_called_once()
        
        # Should generate clarification question
        mock_llm.chat.completions.create.assert_called_once()
        assert isinstance(question, str)
        assert len(question) > 0
        
        # Should store in Redis
        mock_redis.setex.assert_called_once()
        
        # Should return clarification ID
        assert clarification_id is not None
    
    @pytest.mark.asyncio
    async def test_handle_user_confirmation_approved(self, manager, mock_redis, mock_event_bus):
        """Test handling user approval"""
        clarification_data = {
            'node_id': 'mem123',
            'question': 'Is this correct?',
            'created_at': datetime.now().isoformat()
        }
        
        mock_redis.get = AsyncMock(return_value=json.dumps(clarification_data))
        
        await manager.handle_user_confirmation('clarif123', confirmed=True)
        
        # Should promote to Tier 1
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args[0]
        event = call_args[0]
        
        assert event['payload']['to_tier'] == 1
        assert event['payload']['actor'] == 'user'
        
        # Should cleanup Redis
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_user_confirmation_rejected(self, manager, mock_redis, mock_event_bus):
        """Test handling user rejection"""
        clarification_data = {
            'node_id': 'mem123',
            'question': 'Is this correct?',
            'created_at': datetime.now().isoformat()
        }
        
        mock_redis.get = AsyncMock(return_value=json.dumps(clarification_data))
        
        await manager.handle_user_confirmation('clarif123', confirmed=False)
        
        # Should demote to Tier 2
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args[0]
        event = call_args[0]
        
        assert event['payload']['to_tier'] == 2
        assert event['payload']['actor'] == 'user'
    
    @pytest.mark.asyncio
    async def test_handle_expired_clarification(self, manager, mock_redis):
        """Test handling expired clarification"""
        mock_redis.get = AsyncMock(return_value=None)  # Expired
        
        await manager.handle_user_confirmation('clarif123', confirmed=True)
        
        # Should handle gracefully (no events published)
        assert manager.event_bus.publish.call_count == 0
    
    @pytest.mark.asyncio
    async def test_auto_promote_pending_memories(self, manager, mock_redis, mock_event_bus):
        """Test auto-promotion of expired shadow memories"""
        from datetime import datetime, timedelta, timezone
        import json
        
        # Mock Redis scan_iter to return user lists
        async def mock_scan_iter():
            yield "shadow_tier:pending:user123:list"
        
        manager.redis.scan_iter = Mock(return_value=mock_scan_iter())
        
        # Mock smembers to return shadow IDs
        manager.redis.smembers = AsyncMock(return_value=['shadow_mem123', 'shadow_mem456'])
        
        # Mock get to return expired shadow memories
        expired_memory_1 = {
            'shadow_id': 'shadow_mem123',
            'node_id': 'mem123',
            'user_id': 'user123',
            'content': 'Old memory',
            'summary': 'Old memory',
            'ego_score': 0.8,
            'confidence': 0.75,
            'sources': [],
            'model_version': 'v1',
            'created_at': (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            'expires_at': (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            'status': 'pending',
            'auto_promote_after_days': 7
        }
        
        expired_memory_2 = {
            'shadow_id': 'shadow_mem456',
            'node_id': 'mem456',
            'user_id': 'user123',
            'content': 'Recent memory',
            'summary': 'Recent memory',
            'ego_score': 0.85,
            'confidence': 0.8,
            'sources': [],
            'model_version': 'v1',
            'created_at': (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
            'expires_at': (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            'status': 'pending',
            'auto_promote_after_days': 7
        }
        
        # Mock get to return these memories
        def mock_get_side_effect(key):
            if 'shadow_mem123' in key:
                return json.dumps(expired_memory_1)
            elif 'shadow_mem456' in key:
                return json.dumps(expired_memory_2)
            return None
        
        manager.redis.get = AsyncMock(side_effect=mock_get_side_effect)
        
        # Run auto-promotion
        await manager._process_auto_promote_batch()
        
        # Should promote both memories
        assert mock_event_bus.publish.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_promote_to_tier1(self, manager, mock_event_bus):
        """Test promoting to Tier 1"""
        await manager._promote_to_tier1('mem123', actor='auto')
        
        # Should publish tier.promoted event
        mock_event_bus.publish.assert_called_once()
        
        call_args = mock_event_bus.publish.call_args[0]
        event = call_args[0]
        
        assert event['event_type'] == 'tier.promoted'
        assert event['payload']['from_tier'] == 0.5
        assert event['payload']['to_tier'] == 1
        assert event['payload']['actor'] == 'auto'
    
    @pytest.mark.asyncio
    async def test_demote_to_tier2(self, manager, mock_event_bus):
        """Test demoting to Tier 2"""
        await manager._demote_to_tier2('mem123', actor='user')
        
        # Should publish tier.demoted event
        mock_event_bus.publish.assert_called_once()
        
        call_args = mock_event_bus.publish.call_args[0]
        event = call_args[0]
        
        assert event['event_type'] == 'tier.demoted'
        assert event['payload']['from_tier'] == 0.5
        assert event['payload']['to_tier'] == 2
        assert event['payload']['actor'] == 'user'
    
    @pytest.mark.asyncio
    async def test_generate_clarification_question(self, manager, mock_llm):
        """Test clarification question generation"""
        memory = {
            'summary': 'User is allergic to peanuts',
            'ego_score': 0.85,
            'confidence': 0.75
        }
        
        question = await manager._generate_clarification(memory)
        
        # Should call LLM
        mock_llm.chat.completions.create.assert_called_once()
        
        # Should return a string
        assert isinstance(question, str)
        assert len(question) > 0
    
    @pytest.mark.asyncio
    async def test_get_pending_memories(self, manager, mock_arango):
        """Test retrieving pending shadow memories for a user"""
        pending_memories = [
            {
                '_key': 'mem123',
                'user_id': 'user123',
                'summary': 'Pending memory 1',
                'status': 'pending',
                'proposed_at': datetime.now().isoformat()
            },
            {
                '_key': 'mem456',
                'user_id': 'user123',
                'summary': 'Pending memory 2',
                'status': 'pending',
                'proposed_at': datetime.now().isoformat()
            }
        ]
        
        mock_arango.aql.execute = AsyncMock(return_value=pending_memories)
        
        result = await manager.get_pending_for_user('user123')
        
        assert len(result) == 2
        assert result[0]['user_id'] == 'user123'
    
    @pytest.mark.asyncio
    async def test_get_shadow_tier_stats(self, manager, mock_arango):
        """Test getting shadow tier statistics"""
        stats_data = [
            {'status': 'pending', 'count': 10},
            {'status': 'promoted', 'count': 50},
            {'status': 'rejected', 'count': 5}
        ]
        
        mock_arango.aql.execute = AsyncMock(return_value=stats_data)
        
        stats = await manager.get_stats()
        
        assert 'pending' in stats
        assert 'promoted' in stats
        assert stats['pending'] == 10
        assert stats['promoted'] == 50


class TestShadowTierEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def manager(self, mock_arango, mock_redis, mock_event_bus, config):
        mgr = ShadowTier(
            redis_client=mock_redis,
            config=config,
            event_bus=mock_event_bus
        )
        mgr.arango = mock_arango
        mgr.llm = AsyncMock()
        return mgr
    
    @pytest.fixture
    def mock_arango(self):
        arango = Mock()
        arango.collection = Mock(return_value=AsyncMock())
        arango.aql = AsyncMock()
        return arango
    
    @pytest.fixture
    def mock_redis(self):
        return AsyncMock()
    
    @pytest.fixture
    def mock_event_bus(self):
        return AsyncMock()
    
    @pytest.fixture
    def config(self):
        return {
            'shadow_tier_days': 7,
            'ego_threshold': 0.75
        }
    
    @pytest.mark.asyncio
    async def test_propose_without_node_id(self, manager):
        """Test proposing memory without node_id"""
        memory = {
            # Missing node_id
            'summary': 'Test memory',
            'ego_score': 0.80,
            'confidence': 0.75
        }
        
        # Should generate node_id automatically
        clarification_id, question = await manager.propose_core_memory(memory)
        assert clarification_id is not None
        assert 'mem_' in clarification_id  # Auto-generated ID
    
    @pytest.mark.asyncio
    async def test_llm_failure_during_clarification(self, manager):
        """Test handling LLM failure during question generation"""
        manager.llm.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM API error")
        )
        
        memory = {
            'summary': 'Test',
            'ego_score': 0.80,
            'confidence': 0.75
        }
        
        # Should gracefully fallback to default question
        question = await manager._generate_clarification(memory)
        assert isinstance(question, str)
        assert 'Test' in question  # Contains the summary
    
    @pytest.mark.asyncio
    async def test_redis_failure_during_propose(self, manager, mock_redis):
        """Test handling Redis failure during propose"""
        mock_redis.setex = AsyncMock(side_effect=Exception("Redis connection error"))
        
        manager.llm.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="Question?"))]
        ))
        
        memory = {
            'node_id': 'mem123',
            'summary': 'Test',
            'ego_score': 0.80,
            'confidence': 0.75,
            'sources': []
        }
        
        # Should propagate exception
        with pytest.raises(Exception):
            await manager.propose_core_memory(memory)
    
    @pytest.mark.asyncio
    async def test_duplicate_clarification_id(self, manager, mock_redis):
        """Test handling duplicate clarification IDs (rare UUID collision)"""
        # First call succeeds, second call with same ID
        mock_redis.get = AsyncMock(return_value=json.dumps({
            'node_id': 'mem123',
            'question': 'Existing question'
        }))
        
        # Should handle gracefully (get existing clarification)
        await manager.handle_user_confirmation('existing_id', confirmed=True)
        
        # Should still work
        assert manager.event_bus.publish.call_count > 0
    
    @pytest.mark.asyncio
    async def test_auto_promote_with_no_expired(self, manager, mock_redis):
        """Test auto-promote when no memories are expired"""
        # Mock Redis scan_iter to return empty results
        async def mock_scan_iter():
            return
            yield  # Make it an async generator that yields nothing
        
        manager.redis.scan_iter = Mock(return_value=mock_scan_iter())
        
        await manager._process_auto_promote_batch()
        
        # Should not publish any events
        assert manager.event_bus.publish.call_count == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

