"""
Unit tests for Consolidation Worker

Tests async consolidation, priority queues, spaced repetition,
and memory tier assignment logic.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workers.consolidation_worker import ConsolidationWorker


class TestConsolidationWorker:
    """Test suite for Consolidation Worker"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis = AsyncMock()
        redis.xread = AsyncMock(return_value=[])
        redis.xrange = AsyncMock(return_value=[])
        redis.xdel = AsyncMock()
        redis.get = AsyncMock(return_value='{"content": "test", "embedding": [0.1]}')
        redis.zadd = AsyncMock()
        return redis
    
    @pytest.fixture
    def mock_arango(self):
        """Mock ArangoDB client"""
        return AsyncMock()
    
    @pytest.fixture
    def mock_qdrant(self):
        """Mock Qdrant client"""
        qdrant = AsyncMock()
        qdrant.search = AsyncMock(return_value=[])
        qdrant.batch_search = AsyncMock(return_value=[[]])
        return qdrant
    
    @pytest.fixture
    def mock_event_bus(self):
        """Mock event bus"""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def mock_summarizer(self):
        """Mock summarization service"""
        summarizer = AsyncMock()
        summarizer.summarize = AsyncMock(return_value="Consolidated summary")
        return summarizer
    
    @pytest.fixture
    def config(self):
        """Worker configuration"""
        return {
            'consolidation': {
                'similarity_threshold': 0.80,
                'spacing_windows': [24, 72, 168, 720],  # 1d, 3d, 7d, 30d
                'window_tolerance_hours': 2,
                'batch_size': 128,
                'batch_wait_seconds': 30
            },
            'ego_scoring': {
                'tier1_threshold': 0.75,
                'tier2_threshold': 0.45,
                'tier3_threshold': 0.25
            }
        }
    
    @pytest.fixture
    def mock_ego_scorer(self):
        """Mock ego scorer"""
        scorer = Mock()
        scorer.calculate = Mock(return_value=Mock(
            ego_score=0.65,
            tier='tier2',
            components=Mock()
        ))
        return scorer
    
    @pytest.fixture
    def worker(
        self,
        mock_redis,
        mock_arango,
        mock_qdrant,
        mock_event_bus,
        mock_ego_scorer,
        mock_summarizer,
        config
    ):
        """Create worker instance"""
        worker = ConsolidationWorker(
            redis_client=mock_redis,
            arango_client=mock_arango,
            qdrant_client=mock_qdrant,
            config=config,
            ego_scorer=mock_ego_scorer,
            event_bus=mock_event_bus
        )
        worker.summarizer = mock_summarizer
        return worker
    
    def test_initialization(self, worker, config):
        """Test worker initialization"""
        assert worker.config == config
        assert worker.redis is not None
        assert worker.arango is not None
        assert worker.qdrant is not None
    
    def test_determine_tier_core_memory(self, worker):
        """Test tier determination for core memory (Tier 1)"""
        tier = worker._determine_tier(ego_score=0.85, confidence=0.90)
        
        # High ego + high confidence = Tier 1 (autopromote)
        assert tier == 1
    
    def test_determine_tier_shadow(self, worker):
        """Test tier determination for shadow tier (0.5)"""
        tier = worker._determine_tier(ego_score=0.80, confidence=0.75)
        
        # High ego + medium confidence = Shadow tier
        assert tier == 0.5
    
    def test_determine_tier_long_term(self, worker):
        """Test tier determination for long-term memory (Tier 2)"""
        tier = worker._determine_tier(ego_score=0.55, confidence=0.80)
        
        # Medium ego = Tier 2
        assert tier == 2
    
    def test_determine_tier_short_term(self, worker):
        """Test tier determination for short-term memory (Tier 3)"""
        tier = worker._determine_tier(ego_score=0.35, confidence=0.80)
        
        # Low-medium ego = Tier 3
        assert tier == 3
    
    def test_determine_tier_forget(self, worker):
        """Test tier determination for forgettable memory"""
        tier = worker._determine_tier(ego_score=0.15, confidence=0.80)
        
        # Very low ego = Forget (None)
        assert tier is None
    
    @pytest.mark.asyncio
    async def test_check_spacing_trigger_within_window(self, worker):
        """Test spacing trigger detection within window"""
        memory = {
            'observed_at': datetime.now(timezone.utc) - timedelta(hours=24)
        }
        
        # Should match 24-hour window (±2 hours)
        result = await worker._check_spacing_trigger(memory)
        
        assert result is not None  # Should return window name like "24h"
        assert '24' in str(result)  # Should be 24-hour window
    
    @pytest.mark.asyncio
    async def test_check_spacing_trigger_outside_window(self, worker):
        """Test spacing trigger detection outside window"""
        memory = {
            'observed_at': datetime.now(timezone.utc) - timedelta(hours=40)
        }
        
        # 40 hours is between 24h window (24±12 = 12-36h) and 72h window (72±12 = 60-84h)
        # Should not match any window
        result = await worker._check_spacing_trigger(memory)
        
        assert result is None  # No window match
    
    @pytest.mark.asyncio
    async def test_check_spacing_trigger_72_hour_window(self, worker):
        """Test spacing trigger for 3-day window"""
        memory = {
            'observed_at': datetime.now(timezone.utc) - timedelta(hours=73)
        }
        
        # Should match 72-hour window (±2 hours)
        result = await worker._check_spacing_trigger(memory)
        
        assert result is not None  # Should return window name
        assert '72' in str(result)  # Should be 72-hour window
    
    @pytest.mark.asyncio
    async def test_consolidate_single_high_priority(self, worker, mock_redis):
        """Test single consolidation for HIGH priority item"""
        data = {
            'user_id': 'user123',
            'tier4_key': 'tier4:user123:session456:msg123',
            'ego_score': '0.85'
        }
        
        memory_data = {
            'content': 'Important information',
            'embedding': [0.1] * 768,
            'confidence': 0.9,
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'sources': []
        }
        
        mock_redis.get = AsyncMock(return_value=json.dumps(memory_data))
        
        await worker._consolidate_single(data)
        
        # Should publish memory.upsert event
        worker.event_bus.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consolidate_single_expired_key(self, worker, mock_redis):
        """Test handling expired Tier 4 key"""
        data = {
            'user_id': 'user123',
            'tier4_key': 'tier4:expired',
            'ego_score': '0.75'
        }
        
        mock_redis.get = AsyncMock(return_value=None)  # Key expired
        
        # Should handle gracefully
        await worker._consolidate_single(data)
        
        # Should not publish event
        worker.event_bus.publish.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_consolidate_with_related_memories(self, worker, mock_qdrant, mock_summarizer):
        """Test consolidation with related memories (Phase 1.5 feature)"""
        data = {
            'user_id': 'user123',
            'tier4_key': 'tier4:user123:msg123',
            'ego_score': '0.85'
        }
        
        memory_data = {
            'content': 'New memory',
            'embedding': [0.1] * 768,
            'confidence': 0.9,
            'observed_at': datetime.now(timezone.utc).isoformat()
        }
        
        worker.redis.get = AsyncMock(return_value=json.dumps(memory_data))
        
        # Mock related memories
        mock_qdrant.search = AsyncMock(return_value=[
            {'content': 'Related memory 1', 'score': 0.85},
            {'content': 'Related memory 2', 'score': 0.82}
        ])
        
        await worker._consolidate_single(data)
        
        # Phase 1: Just publishes event (related memory clustering is Phase 1.5)
        assert worker.event_bus.publish.call_count == 1
    
    @pytest.mark.asyncio
    async def test_consolidate_batch(self, worker, mock_redis):
        """Test batch consolidation"""
        items = [
            {
                'user_id': 'user123',
                'tier4_key': 'tier4:user123:msg1',
                'ego_score': '0.5'
            },
            {
                'user_id': 'user123',
                'tier4_key': 'tier4:user123:msg2',
                'ego_score': '0.6'
            }
        ]
        
        memory_data = {
            'content': 'Batch memory',
            'embedding': [0.1] * 768,
            'confidence': 0.8,
            'observed_at': datetime.now(timezone.utc).isoformat()
        }
        
        mock_redis.get = AsyncMock(return_value=json.dumps(memory_data))
        
        await worker._consolidate_batch(items)
        
        # Should publish events for both items
        assert worker.event_bus.publish.call_count == 2
    
    @pytest.mark.asyncio
    async def test_publish_memory_upsert_event(self, worker, mock_event_bus):
        """Test publishing memory upsert event"""
        await worker._publish_memory_upsert_event(
            node_id='mem123',
            user_id='user123',
            tier=1,
            ego_score=0.85,
            content='Test memory content',
            metadata={'confidence': 0.9, 'sources': []}
        )
        
        # Verify event structure
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        
        # Check the Event object
        event = call_args[0][0]
        assert event.topic == "memory.upsert"
        assert event.payload['tier'] == 1
        assert event.payload['node_id'] == 'mem123'
        assert event.payload['user_id'] == 'user123'


class TestConsolidationWorkerEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def mock_ego_scorer(self):
        """Mock ego scorer"""
        scorer = Mock()
        scorer.calculate = Mock(return_value=Mock(
            ego_score=0.65,
            tier='tier2',
            components=Mock()
        ))
        return scorer
    
    @pytest.fixture
    def worker(self, mock_redis, mock_arango, mock_qdrant, mock_event_bus, mock_ego_scorer, config):
        return ConsolidationWorker(
            redis_client=mock_redis,
            arango_client=mock_arango,
            qdrant_client=mock_qdrant,
            config=config,
            ego_scorer=mock_ego_scorer,
            event_bus=mock_event_bus
        )
    
    @pytest.fixture
    def mock_redis(self):
        redis = AsyncMock()
        redis.get = AsyncMock()
        redis.xread = AsyncMock()
        return redis
    
    @pytest.fixture
    def mock_arango(self):
        return AsyncMock()
    
    @pytest.fixture
    def mock_qdrant(self):
        return AsyncMock()
    
    @pytest.fixture
    def mock_event_bus(self):
        return AsyncMock()
    
    @pytest.fixture
    def config(self):
        return {
            'consolidation': {
                'spacing_windows': [24, 72, 168, 720],
                'window_tolerance_hours': 2
            },
            'ego_scoring': {
                'tier1_threshold': 0.75,
                'tier2_threshold': 0.45,
                'tier3_threshold': 0.25
            }
        }
    
    def test_tier_boundary_values(self, worker):
        """Test tier determination at boundary values"""
        # Exactly at tier1 threshold
        assert worker._determine_tier(0.75, 0.85) in [0.5, 1]
        
        # Exactly at tier2 threshold
        assert worker._determine_tier(0.45, 0.85) in [2, 3]
        
        # Exactly at tier3 threshold
        assert worker._determine_tier(0.25, 0.85) in [3, None]
    
    def test_extreme_ego_scores(self, worker):
        """Test handling extreme ego scores"""
        # Maximum ego
        tier = worker._determine_tier(1.0, 0.95)
        assert tier in [0.5, 1]
        
        # Minimum ego
        tier = worker._determine_tier(0.0, 0.85)
        assert tier is None
    
    @pytest.mark.asyncio
    async def test_malformed_memory_data(self, worker, mock_redis):
        """Test handling malformed memory data"""
        data = {
            'user_id': 'user123',
            'tier4_key': 'tier4:malformed',
            'ego_score': '0.75'
        }
        
        # Malformed JSON
        mock_redis.get = AsyncMock(return_value="not valid json")
        
        # Should handle error gracefully (no exception raised, just logged)
        await worker._consolidate_single(data)
        
        # Should not have published any events
        assert worker.event_bus.publish.call_count == 0
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, worker, mock_redis):
        """Test handling missing required fields"""
        data = {
            'user_id': 'user123',
            'tier4_key': 'tier4:incomplete'
            # Missing ego_score
        }
        
        # Return memory without observed_at (will cause issues in spacing check)
        memory_data = {
            'content': 'Test',
            'embedding': [0.1] * 768
            # Missing observed_at
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(memory_data))
        
        # Should handle missing fields gracefully
        await worker._consolidate_single(data)
        
        # Should still publish event (observed_at is optional)
        assert worker.event_bus.publish.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_event_bus_failure(self, worker, mock_event_bus):
        """Test handling event bus publication failure"""
        mock_event_bus.publish = AsyncMock(side_effect=Exception("Event bus down"))
        
        memory = {
            'node_id': 'mem123',
            'content': 'Test',
            'embedding': [0.1],
            'ego_score': 0.75,
            'confidence': 0.9,
            'sources': [],
            'version': 1
        }
        
        # Should propagate exception
        with pytest.raises(Exception):
            await worker._publish_memory_upsert_event(
                user_id='user123',
                memory=memory,
                tier=1,
                summary='Test'
            )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

