"""
Shared fixtures for training data tests.

Provides common test fixtures like temp databases, mock services, etc.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from services.training_data_collector import TrainingDataCollector


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def training_collector(temp_db):
    """Create a TrainingDataCollector instance with temp database."""
    return TrainingDataCollector(db_path=temp_db)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for semantic validation."""
    service = Mock()
    service.chat = AsyncMock()
    return service


@pytest.fixture
def mock_classifier():
    """Mock memory classifier."""
    classifier = Mock()
    classifier.current_labels = ["preference", "opinion", "fact", "identity"]
    classifier.predict_single = AsyncMock()
    classifier.add_labels = Mock()
    return classifier

