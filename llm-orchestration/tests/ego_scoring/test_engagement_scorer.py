"""
Comprehensive test suite for EngagementScorer
Tests engagement detection based on conversation dynamics
"""

import pytest
from ml.component_scorers import EngagementScorer, ScorerResult


@pytest.fixture
def config():
    """Configuration for EngagementScorer"""
    return {
        'ego_scoring': {
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
def engagement_scorer(config):
    """Create EngagementScorer instance"""
    return EngagementScorer(config)


@pytest.mark.asyncio
async def test_engagement_scorer_high_engagement_all_high(engagement_scorer):
    """Test high engagement when all factors are high"""
    memory = {
        'content': 'Tell me about quantum physics',
        'user_response_length': 200,  # Max length
        'followup_count': 5,  # Max followups
        'elaboration_score': 1.0  # Max elaboration
    }
    
    result = await engagement_scorer.score(memory)
    
    # All factors at max: (1.0 * 0.4) + (1.0 * 0.3) + (1.0 * 0.3) = 1.0
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_engagement_scorer_low_engagement_all_low(engagement_scorer):
    """Test low engagement when all factors are low"""
    memory = {
        'content': 'ok',
        'user_response_length': 0,
        'followup_count': 0,
        'elaboration_score': 0.0
    }
    
    result = await engagement_scorer.score(memory)
    
    # All factors at 0: (0 * 0.4) + (0 * 0.3) + (0 * 0.3) = 0.0
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_engagement_scorer_medium_response_length(engagement_scorer):
    """Test medium engagement with medium response length"""
    memory = {
        'content': 'Tell me more',
        'user_response_length': 100,  # Half of max (200)
        'followup_count': 0,
        'elaboration_score': 0.0
    }
    
    result = await engagement_scorer.score(memory)
    
    # Normalized response length: 100/200 = 0.5
    # Score: (0.5 * 0.4) + (0 * 0.3) + (0 * 0.3) = 0.2
    assert result.score == 0.2


@pytest.mark.asyncio
async def test_engagement_scorer_high_response_length_exceeds_max(engagement_scorer):
    """Test that response length is capped at max"""
    memory = {
        'content': 'Very long question',
        'user_response_length': 500,  # Exceeds max (200)
        'followup_count': 0,
        'elaboration_score': 0.0
    }
    
    result = await engagement_scorer.score(memory)
    
    # Normalized response length: min(1.0, 500/200) = 1.0 (capped)
    # Score: (1.0 * 0.4) + (0 * 0.3) + (0 * 0.3) = 0.4
    assert result.score == 0.4


@pytest.mark.asyncio
async def test_engagement_scorer_medium_followup_count(engagement_scorer):
    """Test medium engagement with medium followup count"""
    memory = {
        'content': 'Tell me more',
        'user_response_length': 0,
        'followup_count': 2,  # 2 out of max 5
        'elaboration_score': 0.0
    }
    
    result = await engagement_scorer.score(memory)
    
    # Normalized followup: 2/5 = 0.4
    # Score: (0 * 0.4) + (0.4 * 0.3) + (0 * 0.3) = 0.12
    assert result.score == 0.12


@pytest.mark.asyncio
async def test_engagement_scorer_high_followup_count_exceeds_max(engagement_scorer):
    """Test that followup count is capped at max"""
    memory = {
        'content': 'Tell me more',
        'user_response_length': 0,
        'followup_count': 10,  # Exceeds max (5)
        'elaboration_score': 0.0
    }
    
    result = await engagement_scorer.score(memory)
    
    # Normalized followup: min(1.0, 10/5) = 1.0 (capped)
    # Score: (0 * 0.4) + (1.0 * 0.3) + (0 * 0.3) = 0.3
    assert result.score == 0.3


@pytest.mark.asyncio
async def test_engagement_scorer_medium_elaboration(engagement_scorer):
    """Test medium engagement with medium elaboration score"""
    memory = {
        'content': 'Tell me more',
        'user_response_length': 0,
        'followup_count': 0,
        'elaboration_score': 0.5
    }
    
    result = await engagement_scorer.score(memory)
    
    # Score: (0 * 0.4) + (0 * 0.3) + (0.5 * 0.3) = 0.15
    assert result.score == 0.15


@pytest.mark.asyncio
async def test_engagement_scorer_balanced_engagement(engagement_scorer):
    """Test balanced engagement across all factors"""
    memory = {
        'content': 'Tell me about quantum physics in detail',
        'user_response_length': 100,  # 50% of max
        'followup_count': 2,  # 40% of max
        'elaboration_score': 0.6
    }
    
    result = await engagement_scorer.score(memory)
    
    # Normalized response: 100/200 = 0.5
    # Normalized followup: 2/5 = 0.4
    # Score: (0.5 * 0.4) + (0.4 * 0.3) + (0.6 * 0.3) = 0.2 + 0.12 + 0.18 = 0.5
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_engagement_scorer_missing_fields_use_defaults(engagement_scorer):
    """Test that missing fields default to 0"""
    memory = {
        'content': 'Tell me more'
        # All engagement fields missing
    }
    
    result = await engagement_scorer.score(memory)
    
    # All defaults to 0: (0 * 0.4) + (0 * 0.3) + (0 * 0.3) = 0.0
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_engagement_scorer_metadata_includes_normalized_values(engagement_scorer):
    """Test that metadata includes normalized values"""
    memory = {
        'content': 'Tell me more',
        'user_response_length': 100,
        'followup_count': 2,
        'elaboration_score': 0.6
    }
    
    result = await engagement_scorer.score(memory)
    
    assert 'normalized_response_length' in result.metadata
    assert 'normalized_followup_count' in result.metadata
    assert 'elaboration_score' in result.metadata
    assert result.metadata['normalized_response_length'] == 0.5
    assert result.metadata['normalized_followup_count'] == 0.4
    assert result.metadata['elaboration_score'] == 0.6


@pytest.mark.asyncio
async def test_engagement_scorer_score_clamped_to_range(engagement_scorer):
    """Test that score is always in [0, 1] range"""
    test_cases = [
        {'user_response_length': 1000, 'followup_count': 20, 'elaboration_score': 2.0},
        {'user_response_length': -100, 'followup_count': -5, 'elaboration_score': -1.0},
        {'user_response_length': 0, 'followup_count': 0, 'elaboration_score': 0.0},
    ]
    
    for memory_data in test_cases:
        memory = {'content': 'test', **memory_data}
        result = await engagement_scorer.score(memory)
        
        # Score should always be in [0, 1]
        assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_engagement_scorer_real_world_short_response(engagement_scorer):
    """Test real-world example: short response"""
    memory = {
        'content': 'i love steaks',
        'user_response_length': 50,  # Short response
        'followup_count': 0,
        'elaboration_score': 0.3
    }
    
    result = await engagement_scorer.score(memory)
    
    # Normalized response: 50/200 = 0.25
    # Score: (0.25 * 0.4) + (0 * 0.3) + (0.3 * 0.3) = 0.1 + 0 + 0.09 = 0.19
    assert abs(result.score - 0.19) < 0.01


@pytest.mark.asyncio
async def test_engagement_scorer_real_world_long_response(engagement_scorer):
    """Test real-world example: long detailed response"""
    memory = {
        'content': 'i love steaks, especially when cooked medium rare...',
        'user_response_length': 180,  # Long response
        'followup_count': 3,
        'elaboration_score': 0.8
    }
    
    result = await engagement_scorer.score(memory)
    
    # Normalized response: 180/200 = 0.9
    # Normalized followup: 3/5 = 0.6
    # Score: (0.9 * 0.4) + (0.6 * 0.3) + (0.8 * 0.3) = 0.36 + 0.18 + 0.24 = 0.78
    assert abs(result.score - 0.78) < 0.01


@pytest.mark.asyncio
async def test_engagement_scorer_weights_sum_to_one(engagement_scorer):
    """Test that weights sum to 1.0"""
    weights = [
        engagement_scorer.response_length_weight,
        engagement_scorer.followup_count_weight,
        engagement_scorer.elaboration_score_weight
    ]
    
    assert abs(sum(weights) - 1.0) < 0.001


@pytest.mark.asyncio
async def test_engagement_scorer_single_factor_dominance(engagement_scorer):
    """Test that each factor can dominate the score"""
    # Response length dominance
    memory1 = {
        'content': 'test',
        'user_response_length': 200,
        'followup_count': 0,
        'elaboration_score': 0.0
    }
    result1 = await engagement_scorer.score(memory1)
    assert result1.score == 0.4  # Only response_length_weight
    
    # Followup count dominance
    memory2 = {
        'content': 'test',
        'user_response_length': 0,
        'followup_count': 5,
        'elaboration_score': 0.0
    }
    result2 = await engagement_scorer.score(memory2)
    assert result2.score == 0.3  # Only followup_count_weight
    
    # Elaboration dominance
    memory3 = {
        'content': 'test',
        'user_response_length': 0,
        'followup_count': 0,
        'elaboration_score': 1.0
    }
    result3 = await engagement_scorer.score(memory3)
    assert result3.score == 0.3  # Only elaboration_score_weight

