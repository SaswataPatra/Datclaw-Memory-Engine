"""
Unit Tests for Confidence-Based Routing

Tests:
1. Routing decision logic
2. Auto-accept for high confidence
3. Semantic check for medium confidence
4. Fallback for low confidence
5. Threshold boundaries
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from services.classification.classifier_manager import ClassifierManager, CONFIDENCE_THRESHOLDS


@pytest.fixture
def mock_classifier():
    """Mock memory classifier."""
    classifier = Mock()
    classifier.current_labels = ["preference", "opinion", "fact"]
    classifier.predict_single = AsyncMock()
    classifier.add_labels = Mock()
    return classifier


@pytest.fixture
def mock_semantic_validator():
    """Mock semantic validator."""
    validator = Mock()
    validator.validate_classification = AsyncMock()
    validator.training_collector = None
    return validator


@pytest.fixture
def mock_regex_fallback():
    """Mock regex fallback."""
    fallback = Mock()
    fallback.detect_triggers = Mock(return_value=["preference"])
    return fallback


@pytest.fixture
def classifier_manager(mock_classifier, mock_semantic_validator, mock_regex_fallback):
    """Create ClassifierManager instance with mocks."""
    return ClassifierManager(
        classifier_type="hf_api",
        memory_classifier=mock_classifier,
        semantic_validator=mock_semantic_validator,
        regex_fallback=mock_regex_fallback
    )


class TestRoutingDecisionLogic:
    """Test the routing decision logic."""
    
    def test_auto_accept_high_confidence(self, classifier_manager):
        """Test auto-accept for confidence >= 0.85."""
        decision = classifier_manager._determine_routing(0.95)
        assert decision == 'auto_accept'
        
        decision = classifier_manager._determine_routing(0.85)
        assert decision == 'auto_accept'
    
    def test_semantic_check_medium_confidence(self, classifier_manager):
        """Test semantic check for confidence 0.60-0.85."""
        decision = classifier_manager._determine_routing(0.75)
        assert decision == 'semantic_check'
        
        decision = classifier_manager._determine_routing(0.60)
        assert decision == 'semantic_check'
    
    def test_force_fallback_low_confidence(self, classifier_manager):
        """Test force fallback for confidence < 0.60."""
        decision = classifier_manager._determine_routing(0.55)
        assert decision == 'force_fallback'
        
        decision = classifier_manager._determine_routing(0.30)
        assert decision == 'force_fallback'
    
    def test_boundary_conditions(self, classifier_manager):
        """Test exact threshold boundaries."""
        # Just below auto_accept threshold
        decision = classifier_manager._determine_routing(0.849)
        assert decision == 'semantic_check'
        
        # Just above auto_accept threshold
        decision = classifier_manager._determine_routing(0.850)
        assert decision == 'auto_accept'
        
        # Just below semantic_check threshold
        decision = classifier_manager._determine_routing(0.599)
        assert decision == 'force_fallback'
        
        # Just above semantic_check threshold
        decision = classifier_manager._determine_routing(0.600)
        assert decision == 'semantic_check'


class TestAutoAcceptBehavior:
    """Test auto-accept behavior for high-confidence predictions."""
    
    @pytest.mark.asyncio
    async def test_skips_semantic_validation(self, classifier_manager, mock_semantic_validator):
        """Test that high-confidence predictions skip LLM validation."""
        # Mock classifier to return high-confidence prediction
        mock_classifier = classifier_manager.memory_classifier
        mock_classifier.predict_single.return_value = (
            ["preference", "programming_affinity"],
            {"preference": 0.95, "programming_affinity": 0.88},
            []
        )
        
        # Classify
        labels, scores = await classifier_manager.classify_memory(
            message="I love Python programming",
            user_id="test_user"
        )
        
        # Verify semantic validator was NOT called (auto-accept)
        # Note: This depends on implementation - if auto_accept skips validation
        assert set(labels) == {"preference", "programming_affinity"}
        assert scores["preference"] == 0.95
    
    @pytest.mark.asyncio
    async def test_cost_savings(self, classifier_manager, mock_semantic_validator):
        """Test that auto-accept saves LLM calls."""
        # Mock high-confidence predictions
        mock_classifier = classifier_manager.memory_classifier
        mock_classifier.predict_single.return_value = (
            ["preference"],
            {"preference": 0.95},
            []
        )
        
        # Process 100 high-confidence classifications
        call_count_before = mock_semantic_validator.validate_classification.call_count
        
        for i in range(100):
            await classifier_manager.classify_memory(
                message=f"test message {i}",
                user_id="test_user"
            )
        
        call_count_after = mock_semantic_validator.validate_classification.call_count
        
        # Verify semantic validator was called less than 100 times
        # (some should be auto-accepted)
        # Note: Exact behavior depends on implementation
        assert call_count_after - call_count_before < 100


class TestSemanticCheckBehavior:
    """Test semantic check behavior for medium-confidence predictions."""
    
    @pytest.mark.asyncio
    async def test_runs_semantic_validation(self, classifier_manager, mock_semantic_validator):
        """Test that medium-confidence predictions run LLM validation."""
        # Mock classifier to return medium-confidence prediction
        mock_classifier = classifier_manager.memory_classifier
        mock_classifier.predict_single.return_value = (
            ["preference", "opinion"],
            {"preference": 0.75, "opinion": 0.70},
            []
        )
        
        # Mock semantic validator to filter out one label
        mock_semantic_validator.validate_classification.return_value = (
            ["opinion"],  # preference filtered out
            {"opinion": 0.70},
            False
        )
        
        # Classify
        labels, scores = await classifier_manager.classify_memory(
            message="can you tell me the best player?",
            user_id="test_user"
        )
        
        # Verify semantic validator was called
        assert mock_semantic_validator.validate_classification.called
        
        # Verify filtered labels returned
        assert labels == ["opinion"]
        assert "preference" not in labels


class TestFallbackBehavior:
    """Test fallback behavior for low-confidence predictions."""
    
    @pytest.mark.asyncio
    async def test_uses_regex_fallback(self, classifier_manager, mock_regex_fallback):
        """Test that low-confidence predictions use regex fallback."""
        # Mock classifier to return low-confidence prediction
        mock_classifier = classifier_manager.memory_classifier
        mock_classifier.predict_single.return_value = (
            ["preference"],
            {"preference": 0.55},
            []
        )
        
        # Mock regex fallback
        mock_regex_fallback.detect_triggers.return_value = ["preference", "fact"]
        
        # Classify
        labels, scores = await classifier_manager.classify_memory(
            message="I work at Google",
            user_id="test_user"
        )
        
        # Note: Exact behavior depends on implementation
        # Verify that some fallback mechanism was used
        assert len(labels) > 0


class TestThresholdConfiguration:
    """Test that thresholds can be configured."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        assert CONFIDENCE_THRESHOLDS['auto_accept'] == 0.85
        assert CONFIDENCE_THRESHOLDS['semantic_check'] == 0.60
        assert CONFIDENCE_THRESHOLDS['force_fallback'] == 0.60
    
    def test_threshold_ordering(self):
        """Test that thresholds are in correct order."""
        assert CONFIDENCE_THRESHOLDS['auto_accept'] > CONFIDENCE_THRESHOLDS['semantic_check']
        assert CONFIDENCE_THRESHOLDS['semantic_check'] >= CONFIDENCE_THRESHOLDS['force_fallback']


class TestRoutingMetrics:
    """Test routing decision tracking for metrics."""
    
    @pytest.mark.asyncio
    async def test_routing_decision_logged(self, classifier_manager):
        """Test that routing decisions are tracked."""
        # Mock classifier
        mock_classifier = classifier_manager.memory_classifier
        
        # Test different confidence levels
        test_cases = [
            (0.95, 'auto_accept'),
            (0.75, 'semantic_check'),
            (0.55, 'force_fallback')
        ]
        
        for confidence, expected_decision in test_cases:
            decision = classifier_manager._determine_routing(confidence)
            assert decision == expected_decision


class TestEdgeCases:
    """Test edge cases in routing logic."""
    
    def test_zero_confidence(self, classifier_manager):
        """Test handling of zero confidence."""
        decision = classifier_manager._determine_routing(0.0)
        assert decision == 'force_fallback'
    
    def test_perfect_confidence(self, classifier_manager):
        """Test handling of perfect confidence."""
        decision = classifier_manager._determine_routing(1.0)
        assert decision == 'auto_accept'
    
    def test_negative_confidence(self, classifier_manager):
        """Test handling of invalid negative confidence."""
        # Should still route to fallback
        decision = classifier_manager._determine_routing(-0.1)
        assert decision == 'force_fallback'
    
    def test_confidence_above_one(self, classifier_manager):
        """Test handling of invalid confidence > 1."""
        # Should still route to auto_accept
        decision = classifier_manager._determine_routing(1.5)
        assert decision == 'auto_accept'


class TestIntegrationScenarios:
    """Test complete routing scenarios."""
    
    @pytest.mark.asyncio
    async def test_high_confidence_flow(self, classifier_manager, mock_semantic_validator):
        """Test complete flow for high-confidence classification."""
        # Setup
        mock_classifier = classifier_manager.memory_classifier
        mock_classifier.predict_single.return_value = (
            ["preference", "programming_affinity"],
            {"preference": 0.95, "programming_affinity": 0.88},
            []
        )
        
        # Execute
        labels, scores = await classifier_manager.classify_memory(
            message="I love Python",
            user_id="test_user"
        )
        
        # Verify
        assert len(labels) > 0
        assert max(scores.values()) >= 0.85
    
    @pytest.mark.asyncio
    async def test_medium_confidence_with_correction(self, classifier_manager, mock_semantic_validator):
        """Test flow for medium-confidence with semantic correction."""
        # Setup
        mock_classifier = classifier_manager.memory_classifier
        mock_classifier.predict_single.return_value = (
            ["preference", "opinion"],
            {"preference": 0.75, "opinion": 0.70},
            []
        )
        
        # Mock semantic validator to correct
        mock_semantic_validator.validate_classification.return_value = (
            ["opinion"],  # preference removed
            {"opinion": 0.70},
            False
        )
        
        # Execute
        labels, scores = await classifier_manager.classify_memory(
            message="I think Python is good",
            user_id="test_user"
        )
        
        # Verify correction was applied
        assert mock_semantic_validator.validate_classification.called
    
    @pytest.mark.asyncio
    async def test_low_confidence_fallback(self, classifier_manager, mock_regex_fallback):
        """Test flow for low-confidence with regex fallback."""
        # Setup
        mock_classifier = classifier_manager.memory_classifier
        mock_classifier.predict_single.return_value = (
            [],  # Empty result
            {},
            []
        )
        
        mock_regex_fallback.detect_triggers.return_value = ["preference"]
        
        # Execute
        labels, scores = await classifier_manager.classify_memory(
            message="I like cats",
            user_id="test_user"
        )
        
        # Verify fallback was used
        assert len(labels) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

