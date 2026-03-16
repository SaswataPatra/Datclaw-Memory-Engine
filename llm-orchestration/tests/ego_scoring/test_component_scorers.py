"""
Test suite for all component scorers in the ego scoring system.
Tests NoveltyScorer, FrequencyScorer, SentimentScorer, ExplicitImportanceScorer, and EngagementScorer.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from ml.component_scorers import (
    NoveltyScorer,
    FrequencyScorer,
    SentimentScorer,
    ExplicitImportanceScorer,
    EngagementScorer,
    ScorerResult
)


@pytest.fixture
def base_config():
    """Base configuration for all scorers"""
    return {
        'qdrant': {
            'collection_name': 'test_memories'
        },
        'ego_scoring': {
            'novelty_similarity_threshold': 0.8,
            'novelty_top_k': 5,
            'frequency_lookback_days': 30,
            'frequency_similarity_threshold': 0.8,
            'max_frequency_count': 10,
            'explicit_importance_map': {
                'identity': 1.0,
                'family': 1.0,
                'high_value': 0.95,
                'preference': 0.9,
                'goal': 0.85,
                'relationship': 0.85,
                'fact': 0.7,
                'work': 0.7,
                'education': 0.7,
                'event': 0.6,
                'opinion': 0.5,
                'unknown': 0.5
            },
            'default_explicit_importance': 0.5,
            'sentiment': {
                'positive_words': [
                    'love', 'like', 'enjoy', 'great', 'awesome', 'amazing',
                    'excellent', 'wonderful', 'fantastic', 'happy'
                ],
                'negative_words': [
                    'hate', 'dislike', 'terrible', 'awful', 'horrible',
                    'bad', 'worst', 'annoying', 'frustrating', 'angry'
                ],
                'intensifiers': ['very', 'extremely', 'really', 'super'],
                'diminishers': ['a little', 'slightly', 'somewhat'],
                'negations': ['not', 'no', 'never', "don't", "doesn't"]
            },
            'engagement': {
                'response_length_weight': 0.4,
                'followup_count_weight': 0.3,
                'elaboration_score_weight': 0.3,
                'max_response_length': 200,
                'max_followup_count': 5
            }
        }
    }


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing"""
    client = AsyncMock()
    return client


# ==================== NoveltyScorer Tests ====================

@pytest.mark.asyncio
async def test_novelty_scorer_high_novelty(base_config, mock_qdrant_client):
    """Test NoveltyScorer when memory is very novel (no similar memories)"""
    scorer = NoveltyScorer(base_config, mock_qdrant_client)
    
    # Mock no similar memories found
    mock_qdrant_client.search.return_value = []
    
    memory = {
        'embedding': [0.1] * 384,  # Mock embedding
        'user_id': 'test_user',
        'content': 'I just discovered quantum computing'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score == 1.0  # Maximum novelty
    assert result.metadata['reason'] == 'no_similar_memories'


@pytest.mark.asyncio
async def test_novelty_scorer_low_novelty(base_config, mock_qdrant_client):
    """Test NoveltyScorer when memory is similar to existing ones"""
    scorer = NoveltyScorer(base_config, mock_qdrant_client)
    
    # Mock similar memories found
    mock_hit = Mock()
    mock_hit.score = 0.95  # High similarity
    mock_hit.payload = {'content': 'I love quantum computing'}
    mock_qdrant_client.search.return_value = [mock_hit]
    
    memory = {
        'embedding': [0.1] * 384,
        'user_id': 'test_user',
        'content': 'I really love quantum computing'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert abs(result.score - 0.05) < 0.001  # 1.0 - 0.95 = 0.05 (low novelty)
    assert result.metadata['max_similarity'] == 0.95


@pytest.mark.asyncio
async def test_novelty_scorer_missing_embedding(base_config, mock_qdrant_client):
    """Test NoveltyScorer with missing embedding"""
    scorer = NoveltyScorer(base_config, mock_qdrant_client)
    
    memory = {
        'user_id': 'test_user',
        'content': 'Test content'
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 0.5  # Default score
    assert result.metadata['reason'] == 'missing_embedding_or_user_id'


# ==================== FrequencyScorer Tests ====================

@pytest.mark.asyncio
async def test_frequency_scorer_high_frequency(base_config, mock_qdrant_client):
    """Test FrequencyScorer with high frequency (repeated memory)"""
    scorer = FrequencyScorer(base_config, mock_qdrant_client)
    
    # Mock 10 similar memories (max frequency)
    mock_hits = [Mock() for _ in range(10)]
    for hit in mock_hits:
        hit.score = 0.85
    mock_qdrant_client.search.return_value = mock_hits
    
    memory = {
        'embedding': [0.1] * 384,
        'user_id': 'test_user',
        'content': 'I love Python'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score == 1.0  # Maximum frequency
    assert result.metadata['frequency_count'] == 10


@pytest.mark.asyncio
async def test_frequency_scorer_low_frequency(base_config, mock_qdrant_client):
    """Test FrequencyScorer with low frequency (rare memory)"""
    scorer = FrequencyScorer(base_config, mock_qdrant_client)
    
    # Mock 2 similar memories
    mock_hits = [Mock(), Mock()]
    for hit in mock_hits:
        hit.score = 0.85
    mock_qdrant_client.search.return_value = mock_hits
    
    memory = {
        'embedding': [0.1] * 384,
        'user_id': 'test_user',
        'content': 'I tried skydiving once'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score == 0.2  # 2 / 10 = 0.2
    assert result.metadata['frequency_count'] == 2


@pytest.mark.asyncio
async def test_frequency_scorer_no_frequency(base_config, mock_qdrant_client):
    """Test FrequencyScorer with no similar memories"""
    scorer = FrequencyScorer(base_config, mock_qdrant_client)
    
    mock_qdrant_client.search.return_value = []
    
    memory = {
        'embedding': [0.1] * 384,
        'user_id': 'test_user',
        'content': 'Unique experience'
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 0.0
    assert result.metadata['frequency_count'] == 0


# ==================== SentimentScorer Tests ====================

@pytest.mark.asyncio
async def test_sentiment_scorer_positive(base_config):
    """Test SentimentScorer with positive sentiment"""
    scorer = SentimentScorer(base_config)
    
    memory = {
        'content': 'I love this amazing wonderful experience! It was fantastic and great!'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score > 0.3  # Positive intensity (adjusted threshold)
    assert result.metadata['raw_sentiment'] > 0


@pytest.mark.asyncio
async def test_sentiment_scorer_negative(base_config):
    """Test SentimentScorer with negative sentiment"""
    scorer = SentimentScorer(base_config)
    
    memory = {
        'content': 'I hate this terrible awful horrible experience! It was the worst!'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score > 0.3  # Negative intensity (adjusted threshold)
    assert result.metadata['raw_sentiment'] < 0


@pytest.mark.asyncio
async def test_sentiment_scorer_neutral(base_config):
    """Test SentimentScorer with neutral sentiment"""
    scorer = SentimentScorer(base_config)
    
    memory = {
        'content': 'The meeting is scheduled for tomorrow at 3pm.'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score < 0.3  # Low intensity (neutral)


@pytest.mark.asyncio
async def test_sentiment_scorer_with_negation(base_config):
    """Test SentimentScorer with negation"""
    scorer = SentimentScorer(base_config)
    
    memory = {
        'content': 'I do not like this at all'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    # "not like" should be treated as negative
    assert result.metadata['raw_sentiment'] < 0


@pytest.mark.asyncio
async def test_sentiment_scorer_with_intensifier(base_config):
    """Test SentimentScorer with intensifier"""
    scorer = SentimentScorer(base_config)
    
    memory = {
        'content': 'I extremely love this'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    # Intensifier should boost the score
    assert result.score > 0.3


@pytest.mark.asyncio
async def test_sentiment_scorer_empty_content(base_config):
    """Test SentimentScorer with empty content"""
    scorer = SentimentScorer(base_config)
    
    memory = {
        'content': ''
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 0.0
    assert result.metadata['reason'] == 'empty_content'


# ==================== ExplicitImportanceScorer Tests ====================

@pytest.mark.asyncio
async def test_explicit_importance_scorer_identity(base_config):
    """Test ExplicitImportanceScorer with identity label"""
    scorer = ExplicitImportanceScorer(base_config)
    
    memory = {
        'label': 'identity:name',
        'content': 'My name is Saswata'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score == 1.0  # Maximum importance
    assert result.metadata['label'] == 'identity:name'


@pytest.mark.asyncio
async def test_explicit_importance_scorer_family(base_config):
    """Test ExplicitImportanceScorer with family label"""
    scorer = ExplicitImportanceScorer(base_config)
    
    memory = {
        'label': 'family',
        'content': 'My mother is Sunita'
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_explicit_importance_scorer_preference(base_config):
    """Test ExplicitImportanceScorer with preference label"""
    scorer = ExplicitImportanceScorer(base_config)
    
    memory = {
        'label': 'preference:food',
        'content': 'I love pizza'
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 0.9


@pytest.mark.asyncio
async def test_explicit_importance_scorer_fact(base_config):
    """Test ExplicitImportanceScorer with fact label"""
    scorer = ExplicitImportanceScorer(base_config)
    
    memory = {
        'label': 'fact',
        'content': 'Python was created in 1991'
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 0.7


@pytest.mark.asyncio
async def test_explicit_importance_scorer_unknown(base_config):
    """Test ExplicitImportanceScorer with unknown label"""
    scorer = ExplicitImportanceScorer(base_config)
    
    memory = {
        'label': 'unknown',
        'content': 'Random content'
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 0.5  # Default


@pytest.mark.asyncio
async def test_explicit_importance_scorer_missing_label(base_config):
    """Test ExplicitImportanceScorer with missing label"""
    scorer = ExplicitImportanceScorer(base_config)
    
    memory = {
        'content': 'Content without label'
    }
    
    result = await scorer.score(memory)
    
    assert result.score == 0.5  # Default


# ==================== EngagementScorer Tests ====================

@pytest.mark.asyncio
async def test_engagement_scorer_high_engagement(base_config):
    """Test EngagementScorer with high engagement"""
    scorer = EngagementScorer(base_config)
    
    memory = {
        'user_response_length': 200,  # Max length
        'followup_count': 5,  # Max followups
        'elaboration_score': 1.0  # Max elaboration
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score == 1.0  # Maximum engagement


@pytest.mark.asyncio
async def test_engagement_scorer_medium_engagement(base_config):
    """Test EngagementScorer with medium engagement"""
    scorer = EngagementScorer(base_config)
    
    memory = {
        'user_response_length': 100,  # Half of max
        'followup_count': 2,
        'elaboration_score': 0.5
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert 0.4 < result.score < 0.6  # Medium engagement


@pytest.mark.asyncio
async def test_engagement_scorer_low_engagement(base_config):
    """Test EngagementScorer with low engagement"""
    scorer = EngagementScorer(base_config)
    
    memory = {
        'user_response_length': 10,
        'followup_count': 0,
        'elaboration_score': 0.0
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score < 0.3  # Low engagement


@pytest.mark.asyncio
async def test_engagement_scorer_missing_fields(base_config):
    """Test EngagementScorer with missing fields (defaults to 0)"""
    scorer = EngagementScorer(base_config)
    
    memory = {
        'content': 'Test content'
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score == 0.0  # All fields default to 0


@pytest.mark.asyncio
async def test_engagement_scorer_exceeds_max(base_config):
    """Test EngagementScorer with values exceeding max (should be capped)"""
    scorer = EngagementScorer(base_config)
    
    memory = {
        'user_response_length': 500,  # Exceeds max of 200
        'followup_count': 10,  # Exceeds max of 5
        'elaboration_score': 1.5  # Exceeds max of 1.0
    }
    
    result = await scorer.score(memory)
    
    assert isinstance(result, ScorerResult)
    assert result.score == 1.0  # Should be capped at 1.0


# ==================== Integration Tests ====================

@pytest.mark.asyncio
async def test_all_scorers_integration(base_config, mock_qdrant_client):
    """Test all scorers working together on the same memory"""
    
    # Initialize all scorers
    novelty_scorer = NoveltyScorer(base_config, mock_qdrant_client)
    frequency_scorer = FrequencyScorer(base_config, mock_qdrant_client)
    sentiment_scorer = SentimentScorer(base_config)
    explicit_scorer = ExplicitImportanceScorer(base_config)
    engagement_scorer = EngagementScorer(base_config)
    
    # Mock Qdrant responses
    mock_qdrant_client.search.return_value = []
    
    # Test memory
    memory = {
        'embedding': [0.1] * 384,
        'user_id': 'test_user',
        'content': 'I love Python programming! It is amazing!',
        'label': 'preference:programming',
        'user_response_length': 100,
        'followup_count': 3,
        'elaboration_score': 0.8
    }
    
    # Score with all scorers
    novelty_result = await novelty_scorer.score(memory)
    frequency_result = await frequency_scorer.score(memory)
    sentiment_result = await sentiment_scorer.score(memory)
    explicit_result = await explicit_scorer.score(memory)
    engagement_result = await engagement_scorer.score(memory)
    
    # All should return valid ScorerResult objects
    assert all(isinstance(r, ScorerResult) for r in [
        novelty_result, frequency_result, sentiment_result,
        explicit_result, engagement_result
    ])
    
    # All scores should be between 0 and 1
    assert all(0 <= r.score <= 1 for r in [
        novelty_result, frequency_result, sentiment_result,
        explicit_result, engagement_result
    ])
    
    # Check specific expectations
    assert novelty_result.score == 1.0  # No similar memories
    assert frequency_result.score == 0.0  # No similar memories
    assert sentiment_result.score > 0.2  # Positive sentiment (adjusted)
    assert explicit_result.score == 0.9  # Preference label
    assert engagement_result.score > 0.5  # Good engagement


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

