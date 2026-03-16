"""
Test cases for LightGBM Training Components

Tests the bootstrap data generator, training utilities, and LightGBM combiner.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from ml.training.lightgbm_data_generator import LightGBMDataGenerator, compute_component_scores
from ml.combiners.lightgbm_combiner import LightGBMCombiner


class TestLightGBMDataGenerator:
    """Test LightGBM bootstrap data generator"""
    
    @pytest.fixture
    def generator(self):
        """Create data generator with mock LLM client"""
        mock_client = AsyncMock()
        return LightGBMDataGenerator(openai_api_key="test_key")
    
    def test_initialization(self, generator):
        """Test that generator initializes with seed examples"""
        assert generator is not None
        assert len(generator.seed_examples) > 0
        
        # Check seed examples have correct format
        for text, label, ego_score, reasoning in generator.seed_examples:
            assert isinstance(text, str)
            assert isinstance(label, str)
            assert 0.0 <= ego_score <= 1.0
            assert isinstance(reasoning, str)
    
    def test_get_seed_examples(self, generator):
        """Test getting seed examples"""
        examples = generator.get_seed_examples()
        
        assert len(examples) > 0
        assert all(len(ex) == 4 for ex in examples)  # (text, label, score, reasoning)
    
    def test_seed_examples_cover_all_tiers(self, generator):
        """Test that seed examples cover all memory tiers"""
        examples = generator.get_seed_examples()
        
        tier1_count = sum(1 for _, _, score, _ in examples if score >= 0.75)
        tier2_count = sum(1 for _, _, score, _ in examples if 0.50 <= score < 0.75)
        tier3_count = sum(1 for _, _, score, _ in examples if 0.20 <= score < 0.50)
        tier4_count = sum(1 for _, _, score, _ in examples if score < 0.20)
        
        # Should have examples in all tiers
        assert tier1_count > 0, "No Tier 1 examples"
        assert tier2_count > 0, "No Tier 2 examples"
        assert tier3_count > 0, "No Tier 3 examples"
        assert tier4_count > 0, "No Tier 4 examples"
    
    @pytest.mark.asyncio
    async def test_generate_variations(self, generator):
        """Test generating variations of a seed example"""
        seed_text = "My name is Sarah"
        label = "identity"
        target_score = 0.95
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
        - My name is John
        - I'm called Alex
        - People know me as Mike
        """
        
        generator.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        variations = await generator.generate_variations(seed_text, label, target_score, num_variations=3)
        
        assert len(variations) > 0
        assert all(isinstance(var, tuple) and len(var) == 3 for var in variations)
        
        # Check ego scores are close to target
        for text, var_label, ego_score in variations:
            assert var_label == label
            assert abs(ego_score - target_score) <= 0.05  # Within ±0.05
    
    @pytest.mark.asyncio
    async def test_generate_full_dataset(self, generator):
        """Test generating full dataset"""
        # Mock LLM to avoid actual API calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "- Variation 1\n- Variation 2"
        
        generator.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Generate with minimal variations for speed
        dataset = await generator.generate_full_dataset(variations_per_seed=1)
        
        # Should have seed examples + variations
        assert len(dataset) > len(generator.seed_examples)
        
        # Check format
        for text, label, ego_score in dataset:
            assert isinstance(text, str)
            assert isinstance(label, str)
            assert 0.0 <= ego_score <= 1.0
    
    def test_get_ego_score_distribution(self, generator):
        """Test getting ego score distribution"""
        dataset = [
            ("text1", "identity", 0.95),
            ("text2", "preference", 0.65),
            ("text3", "event", 0.35),
            ("text4", "unknown", 0.10)
        ]
        
        distribution = generator.get_ego_score_distribution(dataset)
        
        assert "Tier 1 (>= 0.75)" in distribution
        assert "Tier 2 (0.50-0.75)" in distribution
        assert "Tier 3 (0.20-0.50)" in distribution
        assert "Tier 4 (< 0.20)" in distribution
        
        assert distribution["Tier 1 (>= 0.75)"] == 1
        assert distribution["Tier 2 (0.50-0.75)"] == 1
        assert distribution["Tier 3 (0.20-0.50)"] == 1
        assert distribution["Tier 4 (< 0.20)"] == 1
    
    def test_get_label_distribution(self, generator):
        """Test getting label distribution"""
        dataset = [
            ("text1", "identity", 0.95),
            ("text2", "identity", 0.90),
            ("text3", "preference", 0.65),
            ("text4", "unknown", 0.10)
        ]
        
        distribution = generator.get_label_distribution(dataset)
        
        assert distribution["identity"] == 2
        assert distribution["preference"] == 1
        assert distribution["unknown"] == 1
    
    def test_save_and_load_dataset(self, generator, tmp_path):
        """Test saving and loading dataset"""
        dataset = [
            ("My name is John", "identity", 0.95),
            ("I love pizza", "preference", 0.70)
        ]
        
        filepath = tmp_path / "test_dataset.jsonl"
        
        # Save
        generator.save_dataset(dataset, str(filepath))
        assert filepath.exists()
        
        # Load
        loaded_dataset = generator.load_dataset(str(filepath))
        
        assert len(loaded_dataset) == len(dataset)
        assert loaded_dataset == dataset


class TestComputeComponentScores:
    """Test component score computation"""
    
    @pytest.fixture
    def mock_scorers(self):
        """Create mock component scorers"""
        from ml.component_scorers.base import ScorerResult
        
        scorers = {}
        for name in ['novelty', 'frequency', 'sentiment', 'explicit_importance', 'engagement']:
            mock_scorer = Mock()
            mock_scorer.score = AsyncMock(return_value=ScorerResult(score=0.5))
            scorers[name] = mock_scorer
        
        return scorers
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service"""
        service = Mock()
        service.generate = AsyncMock(return_value=[0.1] * 1536)  # Mock embedding
        return service
    
    @pytest.mark.asyncio
    async def test_compute_component_scores(self, mock_scorers, mock_embedding_service):
        """Test computing component scores for a text"""
        text = "My name is Sarah"
        label = "identity"
        user_id = "test_user"
        
        scores = await compute_component_scores(
            text, label, user_id, mock_scorers, mock_embedding_service
        )
        
        # Should have all required scores
        assert 'novelty_score' in scores
        assert 'frequency_score' in scores
        assert 'sentiment_intensity' in scores
        assert 'explicit_importance_score' in scores
        assert 'engagement_score' in scores
        assert 'recency_decay' in scores
        assert 'reference_count' in scores
        assert 'llm_confidence' in scores
        assert 'source_weight' in scores
        
        # All scores should be between 0 and 1
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range"
    
    @pytest.mark.asyncio
    async def test_compute_scores_without_embedding(self, mock_scorers):
        """Test computing scores when embedding generation fails"""
        text = "My name is Sarah"
        label = "identity"
        user_id = "test_user"
        
        # Mock embedding service that returns None
        mock_embedding_service = Mock()
        mock_embedding_service.generate = AsyncMock(return_value=None)
        
        scores = await compute_component_scores(
            text, label, user_id, mock_scorers, mock_embedding_service
        )
        
        # Should still return scores (with defaults)
        assert 'novelty_score' in scores
        assert scores['novelty_score'] == 0.5  # Default


class TestLightGBMCombiner:
    """Test LightGBM combiner"""
    
    @pytest.fixture
    def combiner(self):
        """Create LightGBM combiner"""
        config = {
            'ego_scoring': {
                'lightgbm_params': {
                    'objective': 'mae',
                    'n_estimators': 10,  # Small for testing
                    'learning_rate': 0.1,
                    'verbose': -1
                }
            }
        }
        return LightGBMCombiner(config)
    
    def test_initialization(self, combiner):
        """Test combiner initialization"""
        assert combiner is not None
        assert combiner.model is not None
        assert not combiner.is_trained
    
    def test_train_with_data(self, combiner):
        """Test training with sample data"""
        # Create small training dataset
        training_data = []
        for i in range(20):
            training_data.append({
                'novelty_score': 0.5 + (i % 5) * 0.1,
                'frequency_score': 0.3,
                'sentiment_intensity': 0.4,
                'explicit_importance_score': 0.7 + (i % 3) * 0.1,
                'engagement_score': 0.5,
                'recency_decay': 1.0,
                'reference_count': 0,
                'llm_confidence': 0.8,
                'source_weight': 1.0,
                'target_ego_score': 0.6 + (i % 4) * 0.1
            })
        
        # Train
        combiner.train(training_data)
        
        assert combiner.is_trained
        assert len(combiner.feature_names) > 0
    
    def test_predict_untrained(self, combiner):
        """Test prediction with untrained model"""
        features = {
            'novelty_score': 0.8,
            'frequency_score': 0.2,
            'sentiment_intensity': 0.6,
            'explicit_importance_score': 0.9,
            'engagement_score': 0.7,
            'recency_decay': 1.0,
            'reference_count': 0,
            'llm_confidence': 0.8,
            'source_weight': 1.0
        }
        
        # Should return default score
        prediction = combiner.predict(features)
        assert prediction == 0.5
    
    def test_predict_trained(self, combiner):
        """Test prediction with trained model"""
        # Train first
        training_data = []
        for i in range(20):
            training_data.append({
                'novelty_score': 0.5,
                'frequency_score': 0.3,
                'sentiment_intensity': 0.4,
                'explicit_importance_score': 0.7,
                'engagement_score': 0.5,
                'recency_decay': 1.0,
                'reference_count': 0,
                'llm_confidence': 0.8,
                'source_weight': 1.0,
                'target_ego_score': 0.6
            })
        
        combiner.train(training_data)
        
        # Predict
        features = {
            'novelty_score': 0.8,
            'frequency_score': 0.2,
            'sentiment_intensity': 0.6,
            'explicit_importance_score': 0.9,
            'engagement_score': 0.7,
            'recency_decay': 1.0,
            'reference_count': 0,
            'llm_confidence': 0.8,
            'source_weight': 1.0
        }
        
        prediction = combiner.predict(features)
        
        # Should return a score between 0 and 1
        assert 0.0 <= prediction <= 1.0
    
    def test_get_feature_importance(self, combiner):
        """Test getting feature importance"""
        # Train first
        training_data = []
        for i in range(20):
            training_data.append({
                'novelty_score': 0.5,
                'frequency_score': 0.3,
                'sentiment_intensity': 0.4,
                'explicit_importance_score': 0.7,
                'engagement_score': 0.5,
                'recency_decay': 1.0,
                'reference_count': 0,
                'llm_confidence': 0.8,
                'source_weight': 1.0,
                'target_ego_score': 0.6
            })
        
        combiner.train(training_data)
        
        importance = combiner.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Feature importance values can be numpy types
        import numpy as np
        assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in importance.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

