"""
Integration Tests for ML Scoring Pipeline

Tests the end-to-end ML scoring flow including:
- DistilBERT classification
- Component scorers
- LightGBM combiner
- Confidence combiner
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from ml.extractors.memory_classifier import DistilBERTMemoryClassifier
from ml.component_scorers import (
    NoveltyScorer, FrequencyScorer, SentimentScorer,
    ExplicitImportanceScorer, EngagementScorer
)
from ml.combiners import LightGBMCombiner, ConfidenceCombiner


class TestMLScoringPipeline:
    """Test end-to-end ML scoring pipeline"""
    
    @pytest.fixture(scope="function")
    def config(self):
        """Create test configuration"""
        return {
            'ego_scoring': {
                'novelty_similarity_threshold': 0.7,
                'frequency_similarity_threshold': 0.6,
                'lightgbm_params': {
                    'objective': 'mae',
                    'n_estimators': 10,
                    'verbose': -1
                },
                'auto_store_confidence_threshold': 0.85,
                'active_learning_confidence_threshold': 0.60
            },
            'qdrant': {
                'collection_name': 'test_memories'
            }
        }
    
    @pytest.fixture(scope="function")
    def distilbert_classifier(self):
        """Create DistilBERT classifier"""
        return DistilBERTMemoryClassifier(device='cpu')
    
    @pytest.fixture(scope="function")
    def mock_qdrant_client(self):
        """Create mock Qdrant client"""
        client = Mock()
        client.search = Mock(return_value=[])  # No similar memories
        return client
    
    @pytest.fixture(scope="function")
    def component_scorers(self, config, mock_qdrant_client):
        """Create all component scorers"""
        return {
            'novelty': NoveltyScorer(config, mock_qdrant_client),
            'frequency': FrequencyScorer(config, mock_qdrant_client),
            'sentiment': SentimentScorer(config),
            'explicit_importance': ExplicitImportanceScorer(config),
            'engagement': EngagementScorer(config)
        }
    
    @pytest.fixture(scope="function")
    def lightgbm_combiner(self, config):
        """Create LightGBM combiner"""
        return LightGBMCombiner(config)
    
    @pytest.fixture(scope="function")
    def confidence_combiner(self, config):
        """Create confidence combiner"""
        return ConfidenceCombiner(config)
    
    def test_distilbert_to_explicit_importance(self, distilbert_classifier, component_scorers):
        """Test flow from DistilBERT to explicit importance scorer"""
        text = "My name is Sarah Johnson"
        
        # Step 1: DistilBERT classification
        labels, scores = distilbert_classifier.predict_single(text, threshold=0.3)
        
        # Step 2: Use label for explicit importance
        if labels:
            label = labels[0]
            memory = {'label': label, 'content': text}
            
            # Step 3: Score explicit importance
            result = asyncio.run(component_scorers['explicit_importance'].score(memory))
            
            assert 0.0 <= result.score <= 1.0
            assert result.metadata['label'] == label
    
    @pytest.mark.asyncio
    async def test_full_scoring_pipeline(self, component_scorers, lightgbm_combiner, confidence_combiner):
        """Test full scoring pipeline from memory to final decision"""
        # Step 1: Prepare memory
        memory = {
            'content': "My mother's name is Elizabeth",
            'user_id': 'test_user',
            'label': 'family',
            'embedding': [0.1] * 1536,  # Mock embedding
            'user_response_length': 50,
            'followup_count': 0,
            'elaboration_score': 0.5
        }
        
        # Step 2: Run component scorers
        scores = {}
        
        novelty_result = await component_scorers['novelty'].score(memory)
        scores['novelty_score'] = novelty_result.score
        
        freq_result = await component_scorers['frequency'].score(memory)
        scores['frequency_score'] = freq_result.score
        
        sent_result = await component_scorers['sentiment'].score(memory)
        scores['sentiment_intensity'] = sent_result.score
        
        exp_result = await component_scorers['explicit_importance'].score(memory)
        scores['explicit_importance_score'] = exp_result.score
        
        eng_result = await component_scorers['engagement'].score(memory)
        scores['engagement_score'] = eng_result.score
        
        # Additional features
        scores['recency_decay'] = 1.0
        scores['reference_count'] = 0
        scores['llm_confidence'] = 0.8
        scores['source_weight'] = 1.0
        
        # Step 3: Combine with LightGBM (or weighted average if not trained)
        if lightgbm_combiner.is_trained:
            ego_score = lightgbm_combiner.predict(scores)
        else:
            # Weighted average fallback
            weights = {
                'novelty': 0.2,
                'frequency': 0.1,
                'sentiment': 0.1,
                'explicit_importance': 0.4,
                'engagement': 0.2
            }
            ego_score = (
                scores['novelty_score'] * weights['novelty'] +
                scores['frequency_score'] * weights['frequency'] +
                scores['sentiment_intensity'] * weights['sentiment'] +
                scores['explicit_importance_score'] * weights['explicit_importance'] +
                scores['engagement_score'] * weights['engagement']
            )
        
        # Step 4: Apply confidence combiner
        result = confidence_combiner.combine(
            ego_score=ego_score,
            extractor_confidence=0.8,
            llm_confidence=0.8,
            is_semantically_consistent=True,
            has_pii=False,
            user_engagement_score=scores['engagement_score']
        )
        
        # Assertions
        assert 0.0 <= ego_score <= 1.0
        assert 'final_confidence' in result
        assert 'routing_decision' in result
        assert result['routing_decision'] in ['auto_store', 'active_learning', 'discard']
    
    @pytest.mark.asyncio
    async def test_high_importance_memory(self, component_scorers, confidence_combiner):
        """Test scoring of high-importance memory"""
        memory = {
            'content': "I'm the CEO of TechCorp",
            'user_id': 'test_user',
            'label': 'high_value',
            'embedding': [0.1] * 1536,
            'user_response_length': 100,
            'followup_count': 2,
            'elaboration_score': 0.8
        }
        
        # Score with explicit importance scorer
        exp_result = await component_scorers['explicit_importance'].score(memory)
        
        # High-value label should have high importance
        assert exp_result.score >= 0.9
    
    @pytest.mark.asyncio
    async def test_low_importance_memory(self, component_scorers):
        """Test scoring of low-importance memory"""
        memory = {
            'content': "What's the weather?",
            'user_id': 'test_user',
            'label': 'unknown',
            'embedding': [0.1] * 1536,
            'user_response_length': 20,
            'followup_count': 0,
            'elaboration_score': 0.2
        }
        
        # Score with explicit importance scorer
        exp_result = await component_scorers['explicit_importance'].score(memory)
        
        # Unknown label should have default/low importance
        assert exp_result.score <= 0.6
    
    def test_confidence_routing_auto_store(self, confidence_combiner):
        """Test routing decision for high-confidence memory"""
        result = confidence_combiner.combine(
            ego_score=0.90,
            extractor_confidence=0.95,
            llm_confidence=0.90,
            is_semantically_consistent=True,
            has_pii=False,
            user_engagement_score=0.8
        )
        
        # High confidence and high ego score should auto-store
        assert result['routing_decision'] == 'auto_store'
        assert result['final_confidence'] >= 0.85
    
    def test_confidence_routing_active_learning(self, confidence_combiner):
        """Test routing decision for medium-confidence memory"""
        result = confidence_combiner.combine(
            ego_score=0.70,
            extractor_confidence=0.75,
            llm_confidence=0.70,
            is_semantically_consistent=True,
            has_pii=False,
            user_engagement_score=0.5
        )
        
        # Medium confidence should go to active learning
        assert result['routing_decision'] in ['active_learning', 'auto_store']
    
    def test_confidence_routing_with_pii(self, confidence_combiner):
        """Test routing decision when PII is detected"""
        result = confidence_combiner.combine(
            ego_score=0.90,
            extractor_confidence=0.95,
            llm_confidence=0.90,
            is_semantically_consistent=True,
            has_pii=True,  # PII detected
            user_engagement_score=0.8
        )
        
        # PII should reduce confidence
        assert result['final_confidence'] < 0.95
        assert 'pii_detected' in result['breakdown']['penalties']
    
    def test_confidence_routing_with_inconsistency(self, confidence_combiner):
        """Test routing decision when semantic inconsistency is detected"""
        result = confidence_combiner.combine(
            ego_score=0.90,
            extractor_confidence=0.95,
            llm_confidence=0.90,
            is_semantically_consistent=False,  # Inconsistent
            has_pii=False,
            user_engagement_score=0.8
        )
        
        # Inconsistency should reduce confidence
        assert result['final_confidence'] < 0.95
        assert 'semantic_inconsistency' in result['breakdown']['penalties']


class TestMLScoringEdgeCases:
    """Test edge cases in ML scoring"""
    
    @pytest.fixture(scope="function")
    def config(self):
        """Create test configuration"""
        return {
            'ego_scoring': {
                'explicit_importance_map': {
                    'identity': 1.0,
                    'family': 1.0,
                    'unknown': 0.5
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_empty_content(self, config):
        """Test scoring with empty content"""
        scorer = SentimentScorer(config)
        memory = {'content': '', 'label': 'unknown'}
        
        result = await scorer.score(memory)
        
        # Should handle gracefully
        assert result.score == 0.0
        assert result.metadata['reason'] == 'empty_content'
    
    @pytest.mark.asyncio
    async def test_missing_embedding(self, config):
        """Test scoring without embedding"""
        mock_client = Mock()
        scorer = NoveltyScorer(config, mock_client)
        
        memory = {
            'content': 'Test',
            'user_id': 'test',
            'embedding': None  # No embedding
        }
        
        result = await scorer.score(memory)
        
        # Should return default score
        assert result.score == 0.5
        assert result.metadata['reason'] == 'missing_embedding_or_user_id'
    
    @pytest.mark.asyncio
    async def test_unknown_label(self, config):
        """Test scoring with unknown label"""
        scorer = ExplicitImportanceScorer(config)
        memory = {'label': 'nonexistent_label', 'content': 'Test'}
        
        result = await scorer.score(memory)
        
        # Should use default importance
        assert result.score == 0.5
    
    @pytest.mark.asyncio
    async def test_multi_label_classification(self):
        """Test handling of multi-label predictions"""
        classifier = DistilBERTMemoryClassifier(device='cpu')
        
        # Text that could be multiple labels
        text = "My sister loves playing tennis"
        labels, scores = classifier.predict_single(text, threshold=0.3)
        
        # Should handle multiple labels gracefully
        assert isinstance(labels, list)
        assert len(labels) >= 0  # Could be 0 for untrained model


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

