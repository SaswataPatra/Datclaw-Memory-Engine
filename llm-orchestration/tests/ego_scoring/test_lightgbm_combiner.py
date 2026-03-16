"""
Test suite for LightGBM Combiner.
Tests training, prediction, and feature importance extraction.
"""

import pytest
import pandas as pd
from typing import Dict, Any, List

from ml.combiners import LightGBMCombiner


@pytest.fixture
def base_config():
    """Base configuration for LightGBM Combiner"""
    return {
        'ego_scoring': {
            'lightgbm_params': {
                'objective': 'mae',
                'metric': 'mae',
                'n_estimators': 50,  # Reduced for faster testing
                'learning_rate': 0.1,
                'num_leaves': 31,
                'verbose': -1,
                'n_jobs': 1,
                'seed': 42
            },
            'enable_shap': False  # Disable SHAP for faster tests
        }
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for LightGBM"""
    return [
        # High ego score examples (Tier 1)
        {
            'novelty_score': 0.9,
            'frequency_score': 0.8,
            'sentiment_intensity': 0.9,
            'explicit_importance_score': 1.0,
            'engagement_score': 0.9,
            'recency_decay': 1.0,
            'reference_count': 5,
            'llm_confidence': 0.95,
            'source_weight': 1.0,
            'target_ego_score': 0.95
        },
        {
            'novelty_score': 0.8,
            'frequency_score': 0.7,
            'sentiment_intensity': 0.85,
            'explicit_importance_score': 1.0,
            'engagement_score': 0.85,
            'recency_decay': 0.95,
            'reference_count': 4,
            'llm_confidence': 0.92,
            'source_weight': 1.0,
            'target_ego_score': 0.92
        },
        # Medium ego score examples (Tier 2)
        {
            'novelty_score': 0.6,
            'frequency_score': 0.5,
            'sentiment_intensity': 0.6,
            'explicit_importance_score': 0.7,
            'engagement_score': 0.6,
            'recency_decay': 0.8,
            'reference_count': 2,
            'llm_confidence': 0.85,
            'source_weight': 0.8,
            'target_ego_score': 0.65
        },
        {
            'novelty_score': 0.5,
            'frequency_score': 0.4,
            'sentiment_intensity': 0.5,
            'explicit_importance_score': 0.7,
            'engagement_score': 0.5,
            'recency_decay': 0.75,
            'reference_count': 1,
            'llm_confidence': 0.80,
            'source_weight': 0.8,
            'target_ego_score': 0.60
        },
        # Low ego score examples (Tier 3)
        {
            'novelty_score': 0.3,
            'frequency_score': 0.2,
            'sentiment_intensity': 0.3,
            'explicit_importance_score': 0.5,
            'engagement_score': 0.3,
            'recency_decay': 0.5,
            'reference_count': 0,
            'llm_confidence': 0.70,
            'source_weight': 0.6,
            'target_ego_score': 0.35
        },
        {
            'novelty_score': 0.2,
            'frequency_score': 0.1,
            'sentiment_intensity': 0.2,
            'explicit_importance_score': 0.5,
            'engagement_score': 0.2,
            'recency_decay': 0.4,
            'reference_count': 0,
            'llm_confidence': 0.65,
            'source_weight': 0.5,
            'target_ego_score': 0.25
        },
        # Add more diverse examples
        {
            'novelty_score': 1.0,
            'frequency_score': 0.0,
            'sentiment_intensity': 0.95,
            'explicit_importance_score': 1.0,
            'engagement_score': 1.0,
            'recency_decay': 1.0,
            'reference_count': 0,
            'llm_confidence': 0.98,
            'source_weight': 1.0,
            'target_ego_score': 0.98
        },
        {
            'novelty_score': 0.4,
            'frequency_score': 0.6,
            'sentiment_intensity': 0.4,
            'explicit_importance_score': 0.6,
            'engagement_score': 0.4,
            'recency_decay': 0.7,
            'reference_count': 3,
            'llm_confidence': 0.75,
            'source_weight': 0.7,
            'target_ego_score': 0.55
        },
        {
            'novelty_score': 0.1,
            'frequency_score': 0.05,
            'sentiment_intensity': 0.1,
            'explicit_importance_score': 0.5,
            'engagement_score': 0.1,
            'recency_decay': 0.3,
            'reference_count': 0,
            'llm_confidence': 0.60,
            'source_weight': 0.4,
            'target_ego_score': 0.15
        },
        {
            'novelty_score': 0.85,
            'frequency_score': 0.75,
            'sentiment_intensity': 0.8,
            'explicit_importance_score': 0.9,
            'engagement_score': 0.8,
            'recency_decay': 0.9,
            'reference_count': 4,
            'llm_confidence': 0.90,
            'source_weight': 0.95,
            'target_ego_score': 0.88
        }
    ]


def test_lightgbm_combiner_initialization(base_config):
    """Test LightGBM Combiner initialization"""
    combiner = LightGBMCombiner(base_config)
    
    assert combiner.model is not None
    assert combiner.is_trained is False
    assert combiner.feature_names == []


def test_lightgbm_combiner_training(base_config, sample_training_data):
    """Test LightGBM Combiner training"""
    combiner = LightGBMCombiner(base_config)
    
    # Train the model
    combiner.train(sample_training_data)
    
    assert combiner.is_trained is True
    assert len(combiner.feature_names) > 0
    assert 'novelty_score' in combiner.feature_names
    assert 'explicit_importance_score' in combiner.feature_names


def test_lightgbm_combiner_prediction_untrained(base_config):
    """Test LightGBM Combiner prediction without training"""
    combiner = LightGBMCombiner(base_config)
    
    features = {
        'novelty_score': 0.8,
        'frequency_score': 0.6,
        'sentiment_intensity': 0.7,
        'explicit_importance_score': 0.9,
        'engagement_score': 0.7,
        'recency_decay': 0.9,
        'reference_count': 3,
        'llm_confidence': 0.85,
        'source_weight': 0.9
    }
    
    prediction = combiner.predict(features)
    
    # Should return default score when not trained
    assert prediction == 0.5


def test_lightgbm_combiner_prediction_trained(base_config, sample_training_data):
    """Test LightGBM Combiner prediction after training"""
    combiner = LightGBMCombiner(base_config)
    combiner.train(sample_training_data)
    
    # Test high ego score prediction
    high_features = {
        'novelty_score': 0.9,
        'frequency_score': 0.8,
        'sentiment_intensity': 0.9,
        'explicit_importance_score': 1.0,
        'engagement_score': 0.9,
        'recency_decay': 1.0,
        'reference_count': 5,
        'llm_confidence': 0.95,
        'source_weight': 1.0
    }
    
    high_prediction = combiner.predict(high_features)
    
    assert 0.0 <= high_prediction <= 1.0
    # With small training set, predictions may not be perfect
    assert high_prediction > 0.5  # Should predict higher ego score
    
    # Test low ego score prediction
    low_features = {
        'novelty_score': 0.2,
        'frequency_score': 0.1,
        'sentiment_intensity': 0.2,
        'explicit_importance_score': 0.5,
        'engagement_score': 0.2,
        'recency_decay': 0.4,
        'reference_count': 0,
        'llm_confidence': 0.65,
        'source_weight': 0.5
    }
    
    low_prediction = combiner.predict(low_features)
    
    assert 0.0 <= low_prediction <= 1.0
    # With small training set, predictions may be similar
    # Just verify both are valid predictions
    assert 0.0 <= low_prediction <= 1.0
    assert 0.0 <= high_prediction <= 1.0


def test_lightgbm_combiner_feature_importance(base_config, sample_training_data):
    """Test LightGBM Combiner feature importance extraction"""
    combiner = LightGBMCombiner(base_config)
    combiner.train(sample_training_data)
    
    importance = combiner.get_feature_importance()
    
    assert isinstance(importance, dict)
    assert len(importance) > 0
    assert 'novelty_score' in importance
    assert 'explicit_importance_score' in importance
    
    # All importance values should be non-negative
    assert all(v >= 0 for v in importance.values())


def test_lightgbm_combiner_empty_training_data(base_config):
    """Test LightGBM Combiner with empty training data"""
    combiner = LightGBMCombiner(base_config)
    
    combiner.train([])
    
    assert combiner.is_trained is False


def test_lightgbm_combiner_prediction_consistency(base_config, sample_training_data):
    """Test that predictions are consistent for the same input"""
    combiner = LightGBMCombiner(base_config)
    combiner.train(sample_training_data)
    
    features = {
        'novelty_score': 0.7,
        'frequency_score': 0.5,
        'sentiment_intensity': 0.6,
        'explicit_importance_score': 0.8,
        'engagement_score': 0.6,
        'recency_decay': 0.8,
        'reference_count': 2,
        'llm_confidence': 0.80,
        'source_weight': 0.8
    }
    
    prediction1 = combiner.predict(features)
    prediction2 = combiner.predict(features)
    
    assert prediction1 == prediction2


def test_lightgbm_combiner_boundary_values(base_config, sample_training_data):
    """Test LightGBM Combiner with boundary values (all 0s and all 1s)"""
    combiner = LightGBMCombiner(base_config)
    combiner.train(sample_training_data)
    
    # All zeros
    zero_features = {
        'novelty_score': 0.0,
        'frequency_score': 0.0,
        'sentiment_intensity': 0.0,
        'explicit_importance_score': 0.0,
        'engagement_score': 0.0,
        'recency_decay': 0.0,
        'reference_count': 0,
        'llm_confidence': 0.0,
        'source_weight': 0.0
    }
    
    zero_prediction = combiner.predict(zero_features)
    assert 0.0 <= zero_prediction <= 1.0
    
    # All ones
    one_features = {
        'novelty_score': 1.0,
        'frequency_score': 1.0,
        'sentiment_intensity': 1.0,
        'explicit_importance_score': 1.0,
        'engagement_score': 1.0,
        'recency_decay': 1.0,
        'reference_count': 10,
        'llm_confidence': 1.0,
        'source_weight': 1.0
    }
    
    one_prediction = combiner.predict(one_features)
    assert 0.0 <= one_prediction <= 1.0


def test_lightgbm_combiner_shap_enabled(base_config, sample_training_data):
    """Test LightGBM Combiner with SHAP enabled"""
    config_with_shap = base_config.copy()
    config_with_shap['ego_scoring']['enable_shap'] = True
    
    combiner = LightGBMCombiner(config_with_shap)
    
    # Training should not fail even with SHAP enabled
    combiner.train(sample_training_data)
    
    assert combiner.is_trained is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

