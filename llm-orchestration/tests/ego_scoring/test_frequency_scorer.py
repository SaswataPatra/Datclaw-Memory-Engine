"""
Comprehensive test suite for FrequencyScorer
Tests frequency detection based on how often similar memories appear
"""

import pytest
from unittest.mock import Mock, AsyncMock
from ml.component_scorers import FrequencyScorer, ScorerResult


@pytest.fixture
def config():
    """Configuration for FrequencyScorer"""
    return {
        'qdrant': {
            'collection_name': 'test_memories'
        },
        'ego_scoring': {
            'frequency_lookback_days': 30,
            'frequency_similarity_threshold': 0.8,
            'max_frequency_count': 10
        }
    }


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    return AsyncMock()


@pytest.fixture
def frequency_scorer(config, mock_qdrant):
    """Create FrequencyScorer instance"""
    return FrequencyScorer(config, mock_qdrant)


@pytest.mark.asyncio
async def test_frequency_scorer_no_similar_memories(frequency_scorer, mock_qdrant):
    """Test frequency when no similar memories exist (first mention)"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    assert result.score == 0.0
    assert result.metadata['frequency_count'] == 0


@pytest.mark.asyncio
async def test_frequency_scorer_low_frequency(frequency_scorer, mock_qdrant):
    """Test low frequency (mentioned 2 times)"""
    # Mock: 2 similar memories found
    mock_results = [
        Mock(score=0.85),
        Mock(score=0.82)
    ]
    mock_qdrant.search.return_value = mock_results
    
    memory = {
        'content': 'I love wagyu steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    # Frequency = 2 / 10 = 0.2
    assert result.score == 0.2
    assert result.metadata['frequency_count'] == 2


@pytest.mark.asyncio
async def test_frequency_scorer_medium_frequency(frequency_scorer, mock_qdrant):
    """Test medium frequency (mentioned 5 times)"""
    mock_results = [Mock(score=0.85) for _ in range(5)]
    mock_qdrant.search.return_value = mock_results
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    # Frequency = 5 / 10 = 0.5
    assert result.score == 0.5
    assert result.metadata['frequency_count'] == 5


@pytest.mark.asyncio
async def test_frequency_scorer_high_frequency(frequency_scorer, mock_qdrant):
    """Test high frequency (mentioned 10+ times, capped at max)"""
    # Mock: 15 similar memories found (exceeds max_frequency_count)
    mock_results = [Mock(score=0.85) for _ in range(15)]
    mock_qdrant.search.return_value = mock_results
    
    memory = {
        'content': 'I love Python',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    # Frequency = min(1.0, 15 / 10) = 1.0 (capped)
    assert result.score == 1.0
    assert result.metadata['frequency_count'] == 15


@pytest.mark.asyncio
async def test_frequency_scorer_missing_embedding(frequency_scorer):
    """Test handling of missing embedding"""
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123'
        # No embedding
    }
    
    result = await frequency_scorer.score(memory)
    
    assert result.score == 0.0
    assert result.metadata['reason'] == 'missing_embedding_or_user_id'


@pytest.mark.asyncio
async def test_frequency_scorer_missing_user_id(frequency_scorer):
    """Test handling of missing user_id"""
    memory = {
        'content': 'I love steaks',
        'embedding': [0.1] * 1536
        # No user_id
    }
    
    result = await frequency_scorer.score(memory)
    
    assert result.score == 0.0
    assert result.metadata['reason'] == 'missing_embedding_or_user_id'


@pytest.mark.asyncio
async def test_frequency_scorer_filters_by_user_id(frequency_scorer, mock_qdrant):
    """Test that frequency scorer filters by user_id"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user789',
        'embedding': [0.1] * 1536
    }
    
    await frequency_scorer.score(memory)
    
    # Verify filter was applied
    call_args = mock_qdrant.search.call_args
    query_filter = call_args.kwargs['query_filter']
    
    assert query_filter.must[0].key == 'user_id'
    assert query_filter.must[0].match.value == 'user789'


@pytest.mark.asyncio
async def test_frequency_scorer_respects_similarity_threshold(frequency_scorer, mock_qdrant):
    """Test that similarity threshold is passed to Qdrant"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    await frequency_scorer.score(memory)
    
    # Verify threshold was passed
    call_args = mock_qdrant.search.call_args
    assert call_args.kwargs['score_threshold'] == 0.8


@pytest.mark.asyncio
async def test_frequency_scorer_respects_max_count_limit(frequency_scorer, mock_qdrant):
    """Test that max_frequency_count + 1 limit is used in query"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    await frequency_scorer.score(memory)
    
    # Verify limit was passed (max_frequency_count + 1 = 11)
    call_args = mock_qdrant.search.call_args
    assert call_args.kwargs['limit'] == 11


@pytest.mark.asyncio
async def test_frequency_scorer_exact_max_count(frequency_scorer, mock_qdrant):
    """Test frequency when exactly at max_frequency_count"""
    # Mock: Exactly 10 similar memories
    mock_results = [Mock(score=0.85) for _ in range(10)]
    mock_qdrant.search.return_value = mock_results
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    # Frequency = 10 / 10 = 1.0
    assert result.score == 1.0
    assert result.metadata['frequency_count'] == 10


@pytest.mark.asyncio
async def test_frequency_scorer_metadata_includes_lookback_days(frequency_scorer, mock_qdrant):
    """Test that metadata includes lookback_days"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    assert result.metadata['lookback_days'] == 30


@pytest.mark.asyncio
async def test_frequency_scorer_metadata_includes_threshold(frequency_scorer, mock_qdrant):
    """Test that metadata includes similarity_threshold"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    assert result.metadata['similarity_threshold'] == 0.8


@pytest.mark.asyncio
async def test_frequency_scorer_single_mention(frequency_scorer, mock_qdrant):
    """Test frequency with single similar memory"""
    mock_results = [Mock(score=0.85)]
    mock_qdrant.search.return_value = mock_results
    
    memory = {
        'content': 'I love steaks',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    # Frequency = 1 / 10 = 0.1
    assert result.score == 0.1
    assert result.metadata['frequency_count'] == 1


@pytest.mark.asyncio
async def test_frequency_scorer_real_world_steak_example(frequency_scorer, mock_qdrant):
    """Test real-world example: 'I love steaks' mentioned 3 times"""
    # Mock: User has mentioned steaks 3 times before
    mock_results = [
        Mock(score=0.92),  # "I love steaks"
        Mock(score=0.88),  # "I love wagyu steaks"
        Mock(score=0.85)   # "steaks are my favorite"
    ]
    mock_qdrant.search.return_value = mock_results
    
    memory = {
        'content': 'what about wagyu steaks?',
        'user_id': 'saswata',
        'embedding': [0.1] * 1536
    }
    
    result = await frequency_scorer.score(memory)
    
    # Frequency = 3 / 10 = 0.3
    assert result.score == 0.3
    assert result.metadata['frequency_count'] == 3

