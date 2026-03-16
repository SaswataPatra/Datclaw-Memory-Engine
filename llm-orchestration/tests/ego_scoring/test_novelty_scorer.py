"""
Comprehensive test suite for NoveltyScorer
Tests novelty detection based on semantic similarity to existing memories
"""

import pytest
from unittest.mock import Mock, AsyncMock
from ml.component_scorers import NoveltyScorer, ScorerResult


@pytest.fixture
def config():
    """Configuration for NoveltyScorer"""
    return {
        'qdrant': {
            'collection_name': 'test_memories'
        },
        'ego_scoring': {
            'novelty_similarity_threshold': 0.8,
            'novelty_top_k': 5
        }
    }


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    return AsyncMock()


@pytest.fixture
def novelty_scorer(config, mock_qdrant):
    """Create NoveltyScorer instance"""
    return NoveltyScorer(config, mock_qdrant)


@pytest.mark.asyncio
async def test_novelty_scorer_high_novelty_no_similar_memories(novelty_scorer, mock_qdrant):
    """Test high novelty when no similar memories exist"""
    # Mock: No similar memories found
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await novelty_scorer.score(memory)
    
    assert result.score == 1.0
    assert result.metadata['reason'] == 'no_similar_memories'


@pytest.mark.asyncio
async def test_novelty_scorer_low_novelty_very_similar_memory(novelty_scorer, mock_qdrant):
    """Test low novelty when very similar memory exists"""
    # Mock: Very similar memory found (0.95 similarity)
    mock_result = Mock()
    mock_result.score = 0.95
    mock_result.payload = {'content': 'I love quantum physics'}
    mock_qdrant.search.return_value = [mock_result]
    
    memory = {
        'content': 'I really love quantum physics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await novelty_scorer.score(memory)
    
    # Novelty = 1 - similarity = 1 - 0.95 = 0.05
    assert abs(result.score - 0.05) < 0.001
    assert result.metadata['max_similarity'] == 0.95
    assert result.metadata['similar_memories_count'] == 1


@pytest.mark.asyncio
async def test_novelty_scorer_medium_novelty(novelty_scorer, mock_qdrant):
    """Test medium novelty with moderately similar memory"""
    # Mock: Moderately similar memory (0.85 similarity)
    mock_result = Mock()
    mock_result.score = 0.85
    mock_result.payload = {'content': 'I like physics'}
    mock_qdrant.search.return_value = [mock_result]
    
    memory = {
        'content': 'I love quantum mechanics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await novelty_scorer.score(memory)
    
    # Novelty = 1 - 0.85 = 0.15
    assert abs(result.score - 0.15) < 0.001


@pytest.mark.asyncio
async def test_novelty_scorer_missing_embedding(novelty_scorer):
    """Test handling of missing embedding"""
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user123',
        # No embedding
    }
    
    result = await novelty_scorer.score(memory)
    
    assert result.score == 0.5
    assert result.metadata['reason'] == 'missing_embedding_or_user_id'


@pytest.mark.asyncio
async def test_novelty_scorer_missing_user_id(novelty_scorer):
    """Test handling of missing user_id"""
    memory = {
        'content': 'I love quantum physics',
        'embedding': [0.1] * 1536
        # No user_id
    }
    
    result = await novelty_scorer.score(memory)
    
    assert result.score == 0.5
    assert result.metadata['reason'] == 'missing_embedding_or_user_id'


@pytest.mark.asyncio
async def test_novelty_scorer_multiple_similar_memories(novelty_scorer, mock_qdrant):
    """Test novelty with multiple similar memories (uses max similarity)"""
    # Mock: Multiple similar memories
    mock_results = [
        Mock(score=0.90, payload={'content': 'I love physics'}),
        Mock(score=0.85, payload={'content': 'I like science'}),
        Mock(score=0.82, payload={'content': 'Physics is great'})
    ]
    mock_qdrant.search.return_value = mock_results
    
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await novelty_scorer.score(memory)
    
    # Uses max similarity (0.90)
    assert abs(result.score - 0.10) < 0.001
    assert result.metadata['max_similarity'] == 0.90
    assert result.metadata['similar_memories_count'] == 3


@pytest.mark.asyncio
async def test_novelty_scorer_filters_by_user_id(novelty_scorer, mock_qdrant):
    """Test that novelty scorer filters by user_id"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user456',
        'embedding': [0.1] * 1536
    }
    
    await novelty_scorer.score(memory)
    
    # Verify filter was applied
    call_args = mock_qdrant.search.call_args
    query_filter = call_args.kwargs['query_filter']
    
    # Check that user_id filter is present
    assert query_filter.must[0].key == 'user_id'
    assert query_filter.must[0].match.value == 'user456'


@pytest.mark.asyncio
async def test_novelty_scorer_respects_similarity_threshold(novelty_scorer, mock_qdrant):
    """Test that similarity threshold is passed to Qdrant"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    await novelty_scorer.score(memory)
    
    # Verify threshold was passed
    call_args = mock_qdrant.search.call_args
    assert call_args.kwargs['score_threshold'] == 0.8


@pytest.mark.asyncio
async def test_novelty_scorer_respects_top_k(novelty_scorer, mock_qdrant):
    """Test that top_k limit is respected"""
    mock_qdrant.search.return_value = []
    
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    await novelty_scorer.score(memory)
    
    # Verify limit was passed
    call_args = mock_qdrant.search.call_args
    assert call_args.kwargs['limit'] == 5


@pytest.mark.asyncio
async def test_novelty_scorer_edge_case_zero_similarity(novelty_scorer, mock_qdrant):
    """Test edge case where similarity is 0 (completely novel)"""
    mock_result = Mock()
    mock_result.score = 0.0
    mock_result.payload = {'content': 'Something completely different'}
    mock_qdrant.search.return_value = [mock_result]
    
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await novelty_scorer.score(memory)
    
    # Novelty = 1 - 0 = 1.0 (completely novel)
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_novelty_scorer_edge_case_perfect_similarity(novelty_scorer, mock_qdrant):
    """Test edge case where similarity is 1.0 (exact duplicate)"""
    mock_result = Mock()
    mock_result.score = 1.0
    mock_result.payload = {'content': 'I love quantum physics'}
    mock_qdrant.search.return_value = [mock_result]
    
    memory = {
        'content': 'I love quantum physics',
        'user_id': 'user123',
        'embedding': [0.1] * 1536
    }
    
    result = await novelty_scorer.score(memory)
    
    # Novelty = 1 - 1 = 0.0 (exact duplicate)
    assert result.score == 0.0

