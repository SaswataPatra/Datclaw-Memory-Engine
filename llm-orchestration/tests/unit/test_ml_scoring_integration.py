"""
Test ML-based Ego Scoring Integration
Tests the integration of ML components into ChatbotService
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict

from services.chatbot_service import ChatbotService
from ml.component_scorers import ScorerResult


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider"""
    provider = AsyncMock()
    provider.name = "mock-llm"
    return provider


@pytest.fixture
def mock_context_manager():
    """Mock context manager"""
    return Mock()


@pytest.fixture
def mock_ego_scorer():
    """Mock legacy ego scorer"""
    scorer = Mock()
    result = Mock()
    result.ego_score = 0.6
    result.tier = 2
    scorer.calculate = Mock(return_value=result)
    return scorer


@pytest.fixture
def mock_event_bus():
    """Mock event bus"""
    return AsyncMock()


@pytest.fixture
def mock_config():
    """Mock configuration"""
    return {
        'ego_scoring': {
            'novelty_similarity_threshold': 0.8,
            'frequency_lookback_days': 30,
            'explicit_importance_map': {
                'identity': 1.0,
                'family': 1.0,
                'preference': 0.9,
                'fact': 0.7
            },
            'sentiment': {
                'positive_words': ['love', 'like', 'enjoy'],
                'negative_words': ['hate', 'dislike']
            },
            'engagement': {
                'response_length_weight': 0.4,
                'followup_count_weight': 0.3,
                'elaboration_score_weight': 0.3
            }
        }
    }


@pytest.mark.asyncio
async def test_legacy_scoring_still_works(
    mock_llm_provider,
    mock_context_manager,
    mock_ego_scorer,
    mock_event_bus,
    mock_config
):
    """Test that legacy regex-based scoring still works"""
    chatbot = ChatbotService(
        llm_provider=mock_llm_provider,
        context_manager=mock_context_manager,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        config=mock_config,
        use_ml_scoring=False  # Use legacy
    )
    
    # Verify legacy mode
    assert chatbot.use_ml_scoring == False
    assert chatbot.ml_scorers is None
    
    # Test memory extraction with legacy scoring
    await chatbot._extract_and_score_memories(
        user_id="test_user",
        session_id="test_session",
        user_message="My name is Alice",
        assistant_response="Nice to meet you, Alice!",
        message_id="msg_1"
    )
    
    # Verify event was published
    assert mock_event_bus.publish.called


@pytest.mark.asyncio
async def test_ml_scoring_initialization(
    mock_llm_provider,
    mock_context_manager,
    mock_ego_scorer,
    mock_event_bus,
    mock_config
):
    """Test that ML components are initialized when enabled"""
    chatbot = ChatbotService(
        llm_provider=mock_llm_provider,
        context_manager=mock_context_manager,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        config=mock_config,
        qdrant_client=None,  # No Qdrant for this test
        use_ml_scoring=True  # Enable ML
    )
    
    # Verify ML mode
    assert chatbot.use_ml_scoring == True
    assert chatbot.ml_scorers is not None
    assert 'sentiment' in chatbot.ml_scorers
    assert 'explicit_importance' in chatbot.ml_scorers
    assert 'engagement' in chatbot.ml_scorers


@pytest.mark.asyncio
async def test_ml_based_scoring_method(
    mock_llm_provider,
    mock_context_manager,
    mock_ego_scorer,
    mock_event_bus,
    mock_config
):
    """Test ML-based scoring method"""
    chatbot = ChatbotService(
        llm_provider=mock_llm_provider,
        context_manager=mock_context_manager,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        config=mock_config,
        use_ml_scoring=True
    )
    
    # Test ML scoring
    ego_score, confidence, triggers = await chatbot._ml_based_scoring(
        user_id="test_user",
        user_message="I love Python programming",
        assistant_response="That's great!"
    )
    
    # Verify results
    assert 0.0 <= ego_score <= 1.0
    assert 0.0 <= confidence <= 1.0
    assert isinstance(triggers, list)
    assert len(triggers) > 0  # Should detect 'preference' trigger


@pytest.mark.asyncio
async def test_ml_scoring_with_identity_trigger(
    mock_llm_provider,
    mock_context_manager,
    mock_ego_scorer,
    mock_event_bus,
    mock_config
):
    """Test ML scoring with identity trigger (should score high)"""
    chatbot = ChatbotService(
        llm_provider=mock_llm_provider,
        context_manager=mock_context_manager,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        config=mock_config,
        use_ml_scoring=True
    )
    
    ego_score, confidence, triggers = await chatbot._ml_based_scoring(
        user_id="test_user",
        user_message="My name is Saswata Patra",
        assistant_response="Nice to meet you, Saswata!"
    )
    
    # Identity should score reasonably high (weighted average without trained model)
    assert ego_score >= 0.5  # Above average importance
    assert 'identity' in triggers
    
    # Note: With trained LightGBM model, this would score higher (>0.7)


@pytest.mark.asyncio
async def test_ml_scoring_fallback_on_error(
    mock_llm_provider,
    mock_context_manager,
    mock_ego_scorer,
    mock_event_bus,
    mock_config
):
    """Test that ML scoring falls back to legacy on error"""
    chatbot = ChatbotService(
        llm_provider=mock_llm_provider,
        context_manager=mock_context_manager,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        config=mock_config,
        use_ml_scoring=True
    )
    
    # Break one of the ML scorers to trigger error
    chatbot.ml_scorers['sentiment'].score = AsyncMock(side_effect=Exception("Test error"))
    
    # Should still work via fallback
    ego_score, confidence, triggers = await chatbot._ml_based_scoring(
        user_id="test_user",
        user_message="I love Python",
        assistant_response="Great!"
    )
    
    # Should return fallback values
    assert 0.0 <= ego_score <= 1.0
    assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_determine_tier():
    """Test tier determination from ego score"""
    chatbot = ChatbotService(
        llm_provider=AsyncMock(),
        context_manager=Mock(),
        ego_scorer=Mock(),
        event_bus=AsyncMock(),
        use_ml_scoring=False
    )
    
    assert chatbot._determine_tier(0.9) == 1  # Core
    assert chatbot._determine_tier(0.6) == 2  # Long-term
    assert chatbot._determine_tier(0.3) == 3  # Short-term
    assert chatbot._determine_tier(0.1) == 4  # Hot buffer


@pytest.mark.asyncio
async def test_extract_and_score_with_ml(
    mock_llm_provider,
    mock_context_manager,
    mock_ego_scorer,
    mock_event_bus,
    mock_config
):
    """Test full extract_and_score_memories with ML enabled"""
    chatbot = ChatbotService(
        llm_provider=mock_llm_provider,
        context_manager=mock_context_manager,
        ego_scorer=mock_ego_scorer,
        event_bus=mock_event_bus,
        config=mock_config,
        use_ml_scoring=True
    )
    
    # Test extraction with ML
    await chatbot._extract_and_score_memories(
        user_id="test_user",
        session_id="test_session",
        user_message="I work at Nexqloud and love Python",
        assistant_response="That's awesome!",
        message_id="msg_1"
    )
    
    # Verify processing state was tracked
    state = chatbot.get_processing_state("test_session")
    assert state.get("msg_1") == "completed"
    
    # Verify event was published
    assert mock_event_bus.publish.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

