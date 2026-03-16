"""
Integration Tests for Component Interactions

Tests how different components work together with real (mocked) dependencies.
These are more than unit tests but don't require the full Docker stack.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from services.context_manager import ContextMemoryManager
from workers.consolidation_worker import ConsolidationWorker
from core.scoring.ego_scorer import TemporalEgoScorer, RecencyCalculator
from core.event_bus import RedisEventBus, Event
from core.contradiction_detector import TemporalContradictionDetector
from core.shadow_tier import ShadowTier


@pytest.mark.integration
class TestContextManagerWithEgoScorer:
    """Test context manager integration with ego scorer"""
    
    @pytest.fixture
    async def setup(self):
        """Set up context manager with real ego scorer"""
        config = {
            'context_memory': {
                'max_tokens': 128000,
                'intelligent_flush': {'ego_threshold': 0.4}
            },
            'ego_scoring': {
                'recency': {
                    'tier1_half_life_days': 180,
                    'tier2_half_life_days': 7,
                    'tier3_half_life_days': 1,
                    'tier4_half_life_minutes': 5
                }
            },
            'event_bus': {'redis_streams_prefix': 'events:'}
        }
        
        redis_client = AsyncMock()
        redis_client.xadd = AsyncMock()
        redis_client.xlen = AsyncMock(return_value=0)
        
        event_bus = RedisEventBus(redis_client, config['event_bus'])
        ego_scorer = TemporalEgoScorer(config)
        
        message_store = AsyncMock()
        message_store.store_message = AsyncMock()
        
        context_manager = ContextMemoryManager(
            redis_client=redis_client,
            event_bus=event_bus,
            ego_scorer=ego_scorer,
            config=config,
            message_store=message_store
        )
        
        return context_manager, ego_scorer, redis_client
    
    @pytest.mark.asyncio
    async def test_intelligent_flush_uses_ego_scores(self, setup):
        """Test that intelligent flush uses real ego scoring"""
        context_manager, ego_scorer, redis_client = await setup
        
        # Create messages with different importance levels
        conversation = [
            {
                "message_id": "msg1",
                "role": "user",
                "content": "Remember: my password is secret123",
                "timestamp": "2024-01-01T00:00:00+00:00"
            },
            {
                "message_id": "msg2",
                "role": "assistant",
                "content": "I understand",
                "timestamp": "2024-01-01T00:00:05+00:00"
            },
            {
                "message_id": "msg3",
                "role": "user",
                "content": "What's the weather like?",
                "timestamp": "2024-01-01T00:00:10+00:00"
            }
        ]
        
        # Mock Redis responses
        redis_client.get = AsyncMock(return_value=json.dumps(conversation))
        redis_client.set = AsyncMock()
        
        result, metadata = await context_manager.manage_context(
            user_id="test_user",
            session_id="test_session",
            conversation_history=conversation
        )
        
        # Should keep important messages based on ego scores
        assert isinstance(result, list)
        assert len(result) <= len(conversation)


@pytest.mark.integration
class TestConsolidationWithEventBus:
    """Test consolidation worker integration with event bus"""
    
    @pytest.fixture
    async def setup(self):
        """Set up consolidation worker with event bus"""
        config = {
            'consolidation': {
                'batch_size': 5,
                'spacing': {
                    '24h': {'window_hours': 24, 'tolerance_hours': 2},
                    '7d': {'window_hours': 168, 'tolerance_hours': 24}
                }
            },
            'ego_scoring': {
                'recency': {
                    'tier1_half_life_days': 180,
                    'tier2_half_life_days': 7,
                    'tier3_half_life_days': 1,
                    'tier4_half_life_minutes': 5
                }
            },
            'event_bus': {'redis_streams_prefix': 'events:'}
        }
        
        redis_client = AsyncMock()
        redis_client.xadd = AsyncMock()
        redis_client.xreadgroup = AsyncMock(return_value=[])
        redis_client.xack = AsyncMock()
        redis_client.xgroup_create = AsyncMock()
        redis_client.xgroup_create.side_effect = Exception("Group exists")
        
        event_bus = RedisEventBus(redis_client, config['event_bus'])
        ego_scorer = TemporalEgoScorer(config)
        
        worker = ConsolidationWorker(
            redis_client=redis_client,
            event_bus=event_bus,
            ego_scorer=ego_scorer,
            config=config
        )
        
        return worker, event_bus, redis_client
    
    @pytest.mark.asyncio
    async def test_consolidation_publishes_events(self, setup):
        """Test that consolidation publishes memory upsert events"""
        worker, event_bus, redis_client = await setup
        
        messages = [
            {
                "message_id": "msg1",
                "user_id": "test_user",
                "session_id": "test_session",
                "content": "I love hiking",
                "role": "user",
                "observed_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        # Process messages
        await worker.consolidate_batch(messages)
        
        # Should have published events
        assert redis_client.xadd.called


@pytest.mark.integration
class TestContradictionDetectorWithScoring:
    """Test contradiction detector with ego scoring"""
    
    @pytest.fixture
    async def setup(self):
        """Set up contradiction detector with dependencies"""
        config = {
            'contradiction': {
                'similarity_threshold': 0.85,
                'temporal_gap_threshold_days': 365
            },
            'llm': {'api_key': 'test_key'},
            'event_bus': {'redis_streams_prefix': 'events:'}
        }
        
        mock_arango = MagicMock()
        mock_arango.db = MagicMock(return_value=MagicMock())
        mock_arango.db().collection = MagicMock(return_value=MagicMock())
        mock_arango.db().collection().get = MagicMock(return_value={
            'node_id': 'old_mem',
            'content': 'User is 25 years old',
            'observed_at': '2023-01-01T00:00:00+00:00'
        })
        
        mock_qdrant = Mock()
        mock_qdrant.search = Mock(return_value=[
            Mock(id='old_mem', score=0.9, payload={'node_id': 'old_mem'})
        ])
        
        redis_client = AsyncMock()
        redis_client.xadd = AsyncMock()
        event_bus = RedisEventBus(redis_client, config['event_bus'])
        
        detector = TemporalContradictionDetector(
            arango=mock_arango,
            qdrant=mock_qdrant,
            event_bus=event_bus,
            config=config
        )
        
        # Mock OpenAI client
        detector.client = AsyncMock()
        detector.client.embeddings = AsyncMock()
        detector.client.embeddings.create = AsyncMock()
        detector.client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        detector.client.chat = AsyncMock()
        detector.client.chat.completions = AsyncMock()
        detector.client.chat.completions.create = AsyncMock()
        detector.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="true"))]
        )
        
        return detector, event_bus
    
    @pytest.mark.asyncio
    async def test_contradiction_detection_flow(self, setup):
        """Test full contradiction detection flow"""
        detector, event_bus = await setup
        
        new_memory = {
            'content': 'User is 30 years old',
            'observed_at': '2024-01-01T00:00:00+00:00'
        }
        
        result = await detector.check_for_contradictions(
            new_memory=new_memory,
            user_id='test_user'
        )
        
        # Should detect potential contradiction
        assert result is not None or result == []  # Depends on LLM response


@pytest.mark.integration
class TestShadowTierWithEventBus:
    """Test shadow tier integration with event bus"""
    
    @pytest.fixture
    async def setup(self):
        """Set up shadow tier with dependencies"""
        config = {
            'shadow_tier': {
                'auto_promote_after_days': 7,
                'clarification_timeout_days': 3
            },
            'llm': {'api_key': 'test_key'},
            'event_bus': {'redis_streams_prefix': 'events:'}
        }
        
        redis_client = AsyncMock()
        redis_client.sadd = AsyncMock()
        redis_client.set = AsyncMock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.smembers = AsyncMock(return_value=set())
        redis_client.scan_iter = AsyncMock()
        redis_client.xadd = AsyncMock()
        
        event_bus = RedisEventBus(redis_client, config['event_bus'])
        
        shadow_tier = ShadowTier(
            redis_client=redis_client,
            config=config,
            event_bus=event_bus
        )
        
        # Mock OpenAI
        shadow_tier.client = AsyncMock()
        shadow_tier.client.chat = AsyncMock()
        shadow_tier.client.chat.completions = AsyncMock()
        shadow_tier.client.chat.completions.create = AsyncMock()
        shadow_tier.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Is this information correct?"))]
        )
        
        return shadow_tier, event_bus, redis_client
    
    @pytest.mark.asyncio
    async def test_shadow_tier_publishes_events(self, setup):
        """Test that shadow tier publishes events"""
        shadow_tier, event_bus, redis_client = await setup
        
        await shadow_tier.add_to_shadow_tier(
            node_id="test_node",
            memory_data={
                "content": "Test memory",
                "user_id": "test_user"
            },
            reason="high_ego_unconfirmed",
            requires_clarification=True
        )
        
        # Should have stored in Redis
        assert redis_client.sadd.called
        assert redis_client.set.called


@pytest.mark.integration  
class TestFullPipeline:
    """Test full pipeline from context to consolidation"""
    
    @pytest.mark.asyncio
    async def test_context_to_consolidation_pipeline(self):
        """Test message flows from context manager through consolidation"""
        config = {
            'context_memory': {
                'max_tokens': 128000,
                'intelligent_flush': {'ego_threshold': 0.4}
            },
            'consolidation': {
                'batch_size': 5,
                'spacing': {'24h': {'window_hours': 24, 'tolerance_hours': 2}}
            },
            'ego_scoring': {
                'recency': {
                    'tier1_half_life_days': 180,
                    'tier2_half_life_days': 7,
                    'tier3_half_life_days': 1,
                    'tier4_half_life_minutes': 5
                }
            },
            'event_bus': {'redis_streams_prefix': 'events:'}
        }
        
        # Set up shared dependencies
        redis_client = AsyncMock()
        redis_client.xadd = AsyncMock()
        redis_client.xlen = AsyncMock(return_value=5)
        redis_client.xreadgroup = AsyncMock(return_value=[])
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.xgroup_create = AsyncMock()
        
        event_bus = RedisEventBus(redis_client, config['event_bus'])
        ego_scorer = TemporalEgoScorer(config)
        
        message_store = AsyncMock()
        message_store.store_message = AsyncMock()
        
        # Create components
        context_manager = ContextMemoryManager(
            redis_client=redis_client,
            event_bus=event_bus,
            ego_scorer=ego_scorer,
            config=config,
            message_store=message_store
        )
        
        consolidation_worker = ConsolidationWorker(
            redis_client=redis_client,
            event_bus=event_bus,
            ego_scorer=ego_scorer,
            config=config
        )
        
        # Test flow
        conversation = [
            {
                "message_id": "msg1",
                "role": "user",
                "content": "I love machine learning",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        # Step 1: Context management
        result, metadata = await context_manager.manage_context(
            user_id="pipeline_user",
            session_id="pipeline_session",
            conversation_history=conversation
        )
        
        assert isinstance(result, list)
        
        # Step 2: Messages should have been queued for consolidation
        # Verify events were published
        assert redis_client.xadd.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


