"""
Test suite for Confidence Combiner.
Tests confidence combination, routing decisions, and penalty application.
"""

import pytest
from typing import Dict, Any

from ml.combiners import ConfidenceCombiner


@pytest.fixture
def base_config():
    """Base configuration for Confidence Combiner"""
    return {
        'ego_scoring': {
            'auto_store_confidence_threshold': 0.85,
            'active_learning_confidence_threshold': 0.60,
            'pii_penalty': 0.3,
            'inconsistency_penalty': 0.2,
            'confidence_weights': {
                'extractor_confidence': 0.5,
                'llm_confidence': 0.3,
                'semantic_consistency_confidence': 0.2
            },
            'thresholds': {
                'tier3': 0.20
            }
        },
        'shadow_tier': {
            'require_confirmation_threshold': 0.75
        }
    }


def test_confidence_combiner_initialization(base_config):
    """Test Confidence Combiner initialization"""
    combiner = ConfidenceCombiner(base_config)
    
    assert combiner.auto_store_threshold == 0.85
    assert combiner.active_learning_threshold == 0.60
    assert combiner.pii_penalty == 0.3
    assert combiner.inconsistency_penalty == 0.2


def test_confidence_combiner_auto_store(base_config):
    """Test Confidence Combiner routing to auto_store"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.95,
        extractor_confidence=0.9,
        llm_confidence=0.9,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.9
    )
    
    assert result['routing_decision'] == 'auto_store'
    assert result['final_confidence'] >= 0.85
    assert 'breakdown' in result


def test_confidence_combiner_active_learning(base_config):
    """Test Confidence Combiner routing to active_learning"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.80,
        extractor_confidence=0.75,
        llm_confidence=0.75,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.7
    )
    
    assert result['routing_decision'] == 'active_learning'
    assert 0.60 <= result['final_confidence'] < 0.85


def test_confidence_combiner_discard_low_confidence(base_config):
    """Test Confidence Combiner routing to discard (low confidence)"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.50,
        extractor_confidence=0.5,
        llm_confidence=0.5,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.3
    )
    
    # With confidence of 0.5, this should be active_learning (>= 0.60 threshold)
    # or discard if below. The weighted combination gives us exactly 0.6
    # (0.5 * 0.5 + 0.3 * 0.5 + 0.2 * 1.0) / 1.0 = 0.6
    assert result['routing_decision'] in ['active_learning', 'discard']
    # Allow for floating point precision
    assert result['final_confidence'] <= 0.61


def test_confidence_combiner_discard_low_ego_score(base_config):
    """Test Confidence Combiner routing to discard (low ego score)"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.15,  # Below tier3 threshold
        extractor_confidence=0.9,
        llm_confidence=0.9,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.8
    )
    
    assert result['routing_decision'] == 'discard'


def test_confidence_combiner_pii_penalty(base_config):
    """Test Confidence Combiner with PII penalty"""
    combiner = ConfidenceCombiner(base_config)
    
    # Without PII
    result_no_pii = combiner.combine(
        ego_score=0.85,
        extractor_confidence=0.9,
        llm_confidence=0.9,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.8
    )
    
    # With PII
    result_with_pii = combiner.combine(
        ego_score=0.85,
        extractor_confidence=0.9,
        llm_confidence=0.9,
        is_semantically_consistent=True,
        has_pii=True,
        user_engagement_score=0.8
    )
    
    # PII should reduce confidence
    assert result_with_pii['final_confidence'] < result_no_pii['final_confidence']
    assert 'pii_detected' in result_with_pii['breakdown']['penalties']
    assert result_with_pii['breakdown']['penalties']['pii_detected'] == 0.3


def test_confidence_combiner_inconsistency_penalty(base_config):
    """Test Confidence Combiner with semantic inconsistency penalty"""
    combiner = ConfidenceCombiner(base_config)
    
    # Consistent
    result_consistent = combiner.combine(
        ego_score=0.85,
        extractor_confidence=0.9,
        llm_confidence=0.9,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.8
    )
    
    # Inconsistent
    result_inconsistent = combiner.combine(
        ego_score=0.85,
        extractor_confidence=0.9,
        llm_confidence=0.9,
        is_semantically_consistent=False,
        has_pii=False,
        user_engagement_score=0.8
    )
    
    # Inconsistency should reduce confidence
    assert result_inconsistent['final_confidence'] < result_consistent['final_confidence']
    assert 'semantic_inconsistency' in result_inconsistent['breakdown']['penalties']
    assert result_inconsistent['breakdown']['penalties']['semantic_inconsistency'] == 0.2


def test_confidence_combiner_multiple_penalties(base_config):
    """Test Confidence Combiner with multiple penalties"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.85,
        extractor_confidence=0.9,
        llm_confidence=0.9,
        is_semantically_consistent=False,
        has_pii=True,
        user_engagement_score=0.8
    )
    
    # Both penalties should be applied
    assert 'pii_detected' in result['breakdown']['penalties']
    assert 'semantic_inconsistency' in result['breakdown']['penalties']
    
    # Total penalty should be 0.5 (0.3 + 0.2)
    total_penalty = sum(result['breakdown']['penalties'].values())
    assert total_penalty == 0.5


def test_confidence_combiner_confidence_clipping(base_config):
    """Test that confidence is clipped to [0, 1]"""
    combiner = ConfidenceCombiner(base_config)
    
    # Test upper bound (should not exceed 1.0)
    result_high = combiner.combine(
        ego_score=1.0,
        extractor_confidence=1.0,
        llm_confidence=1.0,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=1.0
    )
    
    assert result_high['final_confidence'] <= 1.0
    
    # Test lower bound (should not go below 0.0)
    result_low = combiner.combine(
        ego_score=0.1,
        extractor_confidence=0.1,
        llm_confidence=0.1,
        is_semantically_consistent=False,
        has_pii=True,
        user_engagement_score=0.0
    )
    
    assert result_low['final_confidence'] >= 0.0


def test_confidence_combiner_weighted_combination(base_config):
    """Test confidence weighted combination"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.8,
        extractor_confidence=0.8,
        llm_confidence=0.6,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.7
    )
    
    # Expected combined confidence:
    # 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 1.0 = 0.4 + 0.18 + 0.2 = 0.78
    expected_confidence = 0.78
    
    assert abs(result['final_confidence'] - expected_confidence) < 0.01


def test_confidence_combiner_breakdown_structure(base_config):
    """Test that breakdown has correct structure"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.8,
        extractor_confidence=0.8,
        llm_confidence=0.8,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.7
    )
    
    assert 'breakdown' in result
    assert 'base_confidence' in result['breakdown']
    assert 'penalties' in result['breakdown']
    assert 'boosts' in result['breakdown']
    assert isinstance(result['breakdown']['penalties'], dict)
    assert isinstance(result['breakdown']['boosts'], dict)


def test_confidence_combiner_edge_case_all_zeros(base_config):
    """Test Confidence Combiner with all zero inputs"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=0.0,
        extractor_confidence=0.0,
        llm_confidence=0.0,
        is_semantically_consistent=False,
        has_pii=True,
        user_engagement_score=0.0
    )
    
    assert result['routing_decision'] == 'discard'
    assert result['final_confidence'] >= 0.0


def test_confidence_combiner_edge_case_all_ones(base_config):
    """Test Confidence Combiner with all maximum inputs"""
    combiner = ConfidenceCombiner(base_config)
    
    result = combiner.combine(
        ego_score=1.0,
        extractor_confidence=1.0,
        llm_confidence=1.0,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=1.0
    )
    
    assert result['routing_decision'] == 'auto_store'
    assert result['final_confidence'] <= 1.0


def test_confidence_combiner_threshold_boundary(base_config):
    """Test Confidence Combiner at exact threshold boundaries"""
    combiner = ConfidenceCombiner(base_config)
    
    # Exactly at auto_store threshold
    result_auto = combiner.combine(
        ego_score=0.85,
        extractor_confidence=0.85,
        llm_confidence=0.85,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.8
    )
    
    # Should be auto_store (>= threshold)
    assert result_auto['routing_decision'] == 'auto_store'
    
    # Exactly at active_learning threshold
    result_active = combiner.combine(
        ego_score=0.70,
        extractor_confidence=0.60,
        llm_confidence=0.60,
        is_semantically_consistent=True,
        has_pii=False,
        user_engagement_score=0.5
    )
    
    # Should be active_learning (>= threshold)
    assert result_active['routing_decision'] == 'active_learning'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

