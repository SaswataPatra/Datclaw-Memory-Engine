"""
Pytest configuration and shared fixtures

Provides common test fixtures and configuration for all tests.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Standard test configuration"""
    return {
        'redis': {
            'url': 'redis://localhost:6379',
            'max_connections': 10
        },
        'arango': {
            'url': 'http://localhost:8529',
            'database': 'dappy_test',
            'username': 'root',
            'password': 'test'
        },
        'qdrant': {
            'url': 'http://localhost:6333',
            'collection': 'memories_test'
        },
        'event_bus': {
            'provider': 'redis',
            'stream_prefix': 'test:events:',
            'consumer_group': 'test-workers'
        },
        'ego_scoring': {
            'tier1_threshold': 0.75,
            'tier2_threshold': 0.45,
            'tier3_threshold': 0.25,
            'weights': {
                'explicit_user_importance': 0.35,
                'recency_decay': 0.20,
                'frequency': 0.10,
                'sentiment_intensity': 0.10,
                'engagement': 0.10,
                'reference_count': 0.05,
                'confidence': 0.10
            }
        },
        'consolidation': {
            'similarity_threshold': 0.80,
            'spacing_windows': [24, 72, 168, 720],
            'window_tolerance_hours': 2
        },
        'shadow_tier_days': 7,
        'temporal_gap_days': 365
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value="600")
    redis.set = AsyncMock()
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    redis.xadd = AsyncMock(return_value=b"1234567890-0")
    redis.xread = AsyncMock(return_value=[])
    redis.xreadgroup = AsyncMock(return_value=[])
    redis.xack = AsyncMock()
    redis.xdel = AsyncMock()
    redis.xrange = AsyncMock(return_value=[])
    redis.xrevrange = AsyncMock(return_value=[])
    redis.publish = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.zadd = AsyncMock()
    redis.llen = AsyncMock(return_value=0)
    return redis


@pytest.fixture
def mock_arango():
    """Mock ArangoDB client"""
    arango = Mock()
    
    collection = AsyncMock()
    collection.insert = AsyncMock()
    collection.get = AsyncMock(return_value=None)
    collection.update = AsyncMock()
    collection.delete = AsyncMock()
    collection.upsert = AsyncMock()
    
    aql = AsyncMock()
    aql.execute = AsyncMock(return_value=[])
    
    arango.collection = Mock(return_value=collection)
    arango.aql = aql
    
    return arango


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    qdrant = AsyncMock()
    qdrant.search = AsyncMock(return_value=[])
    qdrant.retrieve = AsyncMock(return_value=[])
    qdrant.upsert = AsyncMock()
    qdrant.delete = AsyncMock()
    qdrant.batch_search = AsyncMock(return_value=[[]])
    return qdrant


@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    openai = AsyncMock()
    
    # Mock chat completions
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Generated response"
    
    openai.chat.completions.create = AsyncMock(return_value=response)
    
    return openai


@pytest.fixture
def mock_event_bus():
    """Mock event bus"""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.start_consumer = AsyncMock()
    bus.is_connected = AsyncMock(return_value=True)
    return bus


@pytest.fixture
def mock_ego_scorer():
    """Mock TemporalEgoScorer"""
    from core.scoring.ego_scorer import TemporalEgoScorer
    
    scorer = Mock(spec=TemporalEgoScorer)
    scorer.calculate = Mock(return_value=Mock(
        ego_score=0.65,
        tier='tier2',
        components=Mock(),
        timestamp=datetime.now(timezone.utc).isoformat()
    ))
    return scorer


@pytest.fixture
def sample_memory():
    """Sample memory node"""
    return {
        'node_id': 'mem123',
        'user_id': 'user123',
        'tier': 2,
        'summary': 'User prefers green tea',
        'embedding': [0.1] * 768,
        'ego_score': 0.65,
        'confidence': 0.85,
        'observed_at': datetime.now(timezone.utc),
        'created_at': datetime.now(timezone.utc),
        'version': 1,
        'sources': [
            {
                'message_id': 'msg123',
                'text': 'I prefer green tea now',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]
    }


@pytest.fixture
def sample_message():
    """Sample conversation message"""
    return {
        'message_id': 'msg123',
        'role': 'user',
        'content': 'I prefer green tea now',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'sequence': 1,
        'metadata': {}
    }


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom settings"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add 'unit' marker to all tests in tests/unit/
        if "tests/unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to all tests in tests/integration/
        elif "tests/integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

