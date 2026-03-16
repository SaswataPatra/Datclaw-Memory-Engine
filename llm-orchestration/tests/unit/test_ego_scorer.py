"""
Unit tests for Ego Scorer
"""

import pytest
from datetime import datetime, timedelta
from core.scoring.ego_scorer import TemporalEgoScorer, RecencyCalculator


@pytest.fixture
def config():
    return {
        'ego_scoring': {
            'weights': {
                'explicit_importance': 0.20,
                'recency_decay': 0.20,
                'frequency': 0.10,
                'sentiment_intensity': 0.08,
                'engagement': 0.12,
                'reference_count': 0.10,
                'confidence': 0.10,
                'source_weight': 0.05,
                'novelty': 0.05
            },
            'thresholds': {
                'tier1': 0.75,
                'tier2': 0.45,
                'tier3': 0.20
            },
            'recency': {
                'tier1_half_life_days': 180,
                'tier2_half_life_days': 7,
                'tier3_half_life_days': 1,
                'tier4_half_life_minutes': 5
            }
        }
    }


@pytest.fixture
def ego_scorer(config):
    return TemporalEgoScorer(config)


def test_recency_calculator_tier1(config):
    """Test recency calculation for Tier 1 (6-month half-life)"""
    calc = RecencyCalculator(config)
    
    # Just now
    observed_at = datetime.utcnow()
    recency = calc.calculate(observed_at, tier='tier1')
    assert recency == pytest.approx(1.0, 0.01)
    
    # 180 days ago (half-life)
    observed_at = datetime.utcnow() - timedelta(days=180)
    recency = calc.calculate(observed_at, tier='tier1')
    assert recency == pytest.approx(0.5, 0.01)
    
    # 360 days ago (2 half-lives)
    observed_at = datetime.utcnow() - timedelta(days=360)
    recency = calc.calculate(observed_at, tier='tier1')
    assert recency == pytest.approx(0.25, 0.01)


def test_recency_calculator_tier2(config):
    """Test recency calculation for Tier 2 (1-week half-life)"""
    calc = RecencyCalculator(config)
    
    # Just now
    observed_at = datetime.utcnow()
    recency = calc.calculate(observed_at, tier='tier2')
    assert recency == pytest.approx(1.0, 0.01)
    
    # 7 days ago (half-life)
    observed_at = datetime.utcnow() - timedelta(days=7)
    recency = calc.calculate(observed_at, tier='tier2')
    assert recency == pytest.approx(0.5, 0.01)


def test_ego_score_high_importance(ego_scorer):
    """Test ego scoring for high-importance memory"""
    memory = {
        'memory_id': 'test_1',
        'content': 'User is allergic to peanuts',
        'explicit_importance': 1.0,  # Pinned
        'observed_at': datetime.utcnow(),
        'frequency_7d': 0,
        'sentiment_score': -0.5,
        'user_response_length': 50,
        'followup_count': 2,
        'reference_count': 0,
        'llm_confidence': 0.9,
        'source_weight': 1.0,
        'novelty_score': 0.8
    }
    
    result = ego_scorer.calculate(memory)
    
    # Verify score is calculated and in valid range
    assert 0.0 <= result.ego_score <= 1.0
    assert result.ego_score >= 0.70  # High importance should yield good score
    assert result.tier in ['tier1', 'tier2']  # Should be high tier
    assert result.components.explicit_importance == 1.0
    assert result.components.recency_decay > 0.9  # Recent


def test_ego_score_low_importance(ego_scorer):
    """Test ego scoring for low-importance memory"""
    memory = {
        'memory_id': 'test_2',
        'content': 'Random fact',
        'explicit_importance': 0.0,
        'observed_at': datetime.utcnow() - timedelta(days=30),  # Old
        'frequency_7d': 0,
        'sentiment_score': 0.0,
        'user_response_length': 10,
        'followup_count': 0,
        'reference_count': 0,
        'llm_confidence': 0.3,
        'source_weight': 0.4,
        'novelty_score': 0.2
    }
    
    result = ego_scorer.calculate(memory)
    
    assert result.ego_score < 0.45  # Should be Tier 3 or 4
    assert result.tier in ['tier3', 'tier4']


def test_ego_score_components(ego_scorer):
    """Test that all components are calculated"""
    memory = {
        'memory_id': 'test_3',
        'explicit_importance': 0.5,
        'observed_at': datetime.utcnow(),
        'frequency_7d': 3,
        'sentiment_score': 0.8,
        'user_response_length': 100,
        'followup_count': 1,
        'reference_count': 2,
        'llm_confidence': 0.7,
        'source_weight': 1.0,
        'novelty_score': 0.6
    }
    
    result = ego_scorer.calculate(memory)
    
    # Check all components exist
    assert result.components.explicit_importance == 0.5
    assert result.components.recency_decay > 0
    assert result.components.frequency > 0
    assert result.components.sentiment_intensity == 0.8
    assert result.components.engagement > 0
    assert result.components.reference_count > 0
    assert result.components.confidence == 0.7
    assert result.components.source_weight == 1.0
    assert result.components.novelty == 0.6


def test_ego_score_explanation(ego_scorer):
    """Test explanation generation"""
    memory = {
        'memory_id': 'test_4',
        'explicit_importance': 0.8,
        'observed_at': datetime.utcnow(),
        'frequency_7d': 5,
        'sentiment_score': 0.5,
        'user_response_length': 75,
        'followup_count': 1,
        'reference_count': 1,
        'llm_confidence': 0.8,
        'source_weight': 1.0,
        'novelty_score': 0.7
    }
    
    result = ego_scorer.calculate(memory)
    explanation = ego_scorer.explain(result)
    
    assert "Ego Score" in explanation
    assert "tier" in explanation.lower()
    assert "Component Breakdown" in explanation
    assert "explicit_importance" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

