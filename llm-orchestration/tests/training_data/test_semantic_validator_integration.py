"""
Integration Tests for SemanticValidator with TrainingDataCollector

Tests:
1. Semantic correction logging
2. Confidence anomaly logging
3. Integration with classifier manager
4. End-to-end correction flow
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock

from services.training_data_collector import TrainingDataCollector
from services.classification.semantic_validator import SemanticValidator


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def training_collector(temp_db):
    """Create TrainingDataCollector instance."""
    return TrainingDataCollector(db_path=temp_db)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for semantic validation."""
    service = Mock()
    service.chat = AsyncMock()
    return service


@pytest.fixture
def semantic_validator(mock_llm_service, training_collector):
    """Create SemanticValidator with training collector."""
    return SemanticValidator(
        llm_service=mock_llm_service,
        training_collector=training_collector
    )


class TestSemanticCorrectionLogging:
    """Test that semantic corrections are logged to training collector."""
    
    @pytest.mark.asyncio
    async def test_logs_correction_on_invalid_labels(
        self, 
        semantic_validator, 
        mock_llm_service,
        training_collector
    ):
        """Test that corrections are logged when LLM identifies invalid labels."""
        # Mock LLM response indicating invalid labels
        mock_response = Mock()
        mock_response.content = """{
            "is_coherent": false,
            "invalid_labels": ["preference"],
            "valid_labels": ["opinion"],
            "reasoning": "preference implies personal taste, but this is asking for expert opinion"
        }"""
        mock_llm_service.chat.return_value = mock_response
        
        # Run semantic coherence check
        is_coherent, invalid_labels = await semantic_validator.check_semantic_coherence(
            text="can you tell me the best player?",
            predicted_labels=["opinion", "preference"],
            scores={"opinion": 0.94, "preference": 0.89}
        )
        
        # Verify correction was logged
        stats = training_collector.get_stats()
        assert stats['total_corrections'] == 1
        assert stats['by_source']['semantic_validator'] == 1
        
        # Verify correction details
        dataset = training_collector.get_training_dataset()
        assert len(dataset) == 1
        assert dataset[0]['text'] == "can you tell me the best player?"
        assert "preference" in dataset[0]['reasoning']
    
    @pytest.mark.asyncio
    async def test_no_logging_when_coherent(
        self,
        semantic_validator,
        mock_llm_service,
        training_collector
    ):
        """Test that no correction is logged when labels are valid."""
        # Mock LLM response indicating all labels are valid
        mock_response = Mock()
        mock_response.content = """{
            "is_coherent": true,
            "invalid_labels": [],
            "valid_labels": ["preference", "programming_affinity"],
            "reasoning": "All labels are appropriate"
        }"""
        mock_llm_service.chat.return_value = mock_response
        
        # Run semantic coherence check
        is_coherent, invalid_labels = await semantic_validator.check_semantic_coherence(
            text="I love Python programming",
            predicted_labels=["preference", "programming_affinity"],
            scores={"preference": 0.95, "programming_affinity": 0.88}
        )
        
        # Verify no correction was logged
        stats = training_collector.get_stats()
        assert stats['total_corrections'] == 0


class TestConfidenceAnomalyLogging:
    """Test that confidence anomalies are logged."""
    
    def test_logs_too_many_labels(self, semantic_validator, training_collector):
        """Test logging when too many labels are predicted."""
        # Analyze distribution with too many labels
        is_suspicious, reason = semantic_validator.analyze_confidence_distribution(
            scores={
                "label1": 0.8,
                "label2": 0.75,
                "label3": 0.7,
                "label4": 0.65,
                "label5": 0.6,
                "label6": 0.55
            },
            threshold=0.5,
            text="test message with too many labels"
        )
        
        # Verify anomaly was logged
        assert is_suspicious
        stats = training_collector.get_stats()
        assert stats['total_anomalies'] == 1
    
    def test_logs_suspicious_distribution(self, semantic_validator, training_collector):
        """Test logging when distribution is suspicious."""
        # Analyze distribution with multiple mid-range similar scores
        is_suspicious, reason = semantic_validator.analyze_confidence_distribution(
            scores={
                "label1": 0.65,
                "label2": 0.63,
                "label3": 0.61
            },
            threshold=0.5,
            text="test message with suspicious distribution"
        )
        
        # Verify anomaly was logged
        assert is_suspicious
        stats = training_collector.get_stats()
        assert stats['total_anomalies'] == 1
    
    def test_no_logging_for_normal_distribution(self, semantic_validator, training_collector):
        """Test no logging for normal confidence distribution."""
        # Analyze normal distribution
        is_suspicious, reason = semantic_validator.analyze_confidence_distribution(
            scores={
                "label1": 0.95,
                "label2": 0.88
            },
            threshold=0.5,
            text="test message with normal distribution"
        )
        
        # Verify no anomaly was logged
        assert not is_suspicious
        stats = training_collector.get_stats()
        assert stats['total_anomalies'] == 0


class TestValidationWithLogging:
    """Test complete validation flow with logging."""
    
    @pytest.mark.asyncio
    async def test_validate_classification_with_logging(
        self,
        semantic_validator,
        mock_llm_service,
        training_collector
    ):
        """Test complete validation flow logs corrections."""
        # Mock LLM to identify false positive
        mock_response = Mock()
        mock_response.content = """{
            "is_coherent": false,
            "invalid_labels": ["medical_condition"],
            "valid_labels": ["preference", "technology_interest"],
            "reasoning": "medical_condition is wrong - this is about software memory, not health"
        }"""
        mock_llm_service.chat.return_value = mock_response
        
        # Run validation
        filtered_labels, filtered_scores, needs_discovery = await semantic_validator.validate_classification(
            text="I'm working on a memory optimization project",
            predicted_labels=["preference", "technology_interest", "medical_condition"],
            scores={"preference": 0.7, "technology_interest": 0.68, "medical_condition": 0.65},
            threshold=0.5
        )
        
        # Verify correction was logged
        stats = training_collector.get_stats()
        assert stats['total_corrections'] >= 1
        
        # Verify anomaly was also logged (suspicious distribution)
        assert stats['total_anomalies'] >= 1
        
        # Verify filtered labels
        assert "medical_condition" not in filtered_labels
        assert "preference" in filtered_labels


class TestCaching:
    """Test that caching doesn't interfere with logging."""
    
    @pytest.mark.asyncio
    async def test_cache_doesnt_prevent_logging(
        self,
        semantic_validator,
        mock_llm_service,
        training_collector
    ):
        """Test that cached results still log corrections."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "is_coherent": false,
            "invalid_labels": ["preference"],
            "valid_labels": ["opinion"],
            "reasoning": "test"
        }"""
        mock_llm_service.chat.return_value = mock_response
        
        # First call - should hit LLM and log
        await semantic_validator.check_semantic_coherence(
            text="test message",
            predicted_labels=["opinion", "preference"],
            scores={"opinion": 0.9, "preference": 0.8}
        )
        
        stats1 = training_collector.get_stats()
        assert stats1['total_corrections'] == 1
        
        # Second call - should hit cache
        await semantic_validator.check_semantic_coherence(
            text="test message",
            predicted_labels=["opinion", "preference"],
            scores={"opinion": 0.9, "preference": 0.8}
        )
        
        # Should not log again (cached)
        stats2 = training_collector.get_stats()
        assert stats2['total_corrections'] == 1  # Still 1, not 2


class TestErrorHandling:
    """Test error handling in logging."""
    
    @pytest.mark.asyncio
    async def test_continues_on_logging_error(
        self,
        semantic_validator,
        mock_llm_service
    ):
        """Test that validation continues even if logging fails."""
        # Set training_collector to None to simulate logging failure
        semantic_validator.training_collector = None
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "is_coherent": false,
            "invalid_labels": ["preference"],
            "valid_labels": ["opinion"],
            "reasoning": "test"
        }"""
        mock_llm_service.chat.return_value = mock_response
        
        # Should not raise exception
        is_coherent, invalid_labels = await semantic_validator.check_semantic_coherence(
            text="test message",
            predicted_labels=["opinion", "preference"],
            scores={"opinion": 0.9, "preference": 0.8}
        )
        
        # Validation should still work
        assert not is_coherent
        assert "preference" in invalid_labels


class TestProvenanceTracking:
    """Test that full provenance is tracked."""
    
    @pytest.mark.asyncio
    async def test_tracks_confidence_scores(
        self,
        semantic_validator,
        mock_llm_service,
        training_collector
    ):
        """Test that confidence scores are tracked."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "is_coherent": false,
            "invalid_labels": ["preference"],
            "valid_labels": ["opinion"],
            "reasoning": "test"
        }"""
        mock_llm_service.chat.return_value = mock_response
        
        # Run validation
        await semantic_validator.check_semantic_coherence(
            text="test",
            predicted_labels=["opinion", "preference"],
            scores={"opinion": 0.94, "preference": 0.89}
        )
        
        # Verify confidence is tracked
        dataset = training_collector.get_training_dataset()
        assert len(dataset) == 1
        assert dataset[0]['confidence'] == 0.94  # Max confidence
        assert dataset[0]['scores']['opinion'] == 0.94
        assert dataset[0]['scores']['preference'] == 0.89
    
    @pytest.mark.asyncio
    async def test_tracks_reasoning(
        self,
        semantic_validator,
        mock_llm_service,
        training_collector
    ):
        """Test that LLM reasoning is tracked."""
        # Mock LLM response with specific reasoning
        mock_response = Mock()
        mock_response.content = """{
            "is_coherent": false,
            "invalid_labels": ["medical_condition"],
            "valid_labels": ["technology_interest"],
            "reasoning": "This is about software memory, not medical conditions"
        }"""
        mock_llm_service.chat.return_value = mock_response
        
        # Run validation
        await semantic_validator.check_semantic_coherence(
            text="working on memory project",
            predicted_labels=["technology_interest", "medical_condition"],
            scores={"technology_interest": 0.8, "medical_condition": 0.7}
        )
        
        # Verify reasoning is tracked
        dataset = training_collector.get_training_dataset()
        assert len(dataset) == 1
        assert "software memory" in dataset[0]['reasoning']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

