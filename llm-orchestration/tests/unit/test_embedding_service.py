"""
Test suite for EmbeddingService
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from services.embedding_service import EmbeddingService


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI embedding response"""
    mock_embedding = [0.1] * 1536
    mock_data = Mock()
    mock_data.embedding = mock_embedding
    mock_response = Mock()
    mock_response.data = [mock_data]
    return mock_response


@pytest.mark.asyncio
async def test_embedding_service_initialization():
    """Test EmbeddingService initialization"""
    service = EmbeddingService(api_key="sk-test-123")
    
    assert service.api_key == "sk-test-123"
    assert service.model == "text-embedding-3-small"
    assert service.embedding_dim == 1536


@pytest.mark.asyncio
async def test_embedding_service_custom_model():
    """Test EmbeddingService with custom model"""
    service = EmbeddingService(api_key="sk-test-123", model="text-embedding-3-large")
    
    assert service.model == "text-embedding-3-large"
    assert service.embedding_dim == 3072


@pytest.mark.asyncio
async def test_generate_embedding_success(mock_openai_response):
    """Test successful embedding generation"""
    service = EmbeddingService(api_key="sk-test-123")
    
    # Mock the client
    service.client.embeddings.create = AsyncMock(return_value=mock_openai_response)
    
    result = await service.generate("I love steaks")
    
    assert result is not None
    assert len(result) == 1536
    assert result == mock_openai_response.data[0].embedding
    
    # Verify API call
    service.client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="I love steaks"
    )


@pytest.mark.asyncio
async def test_generate_embedding_api_error():
    """Test embedding generation handles API errors"""
    service = EmbeddingService(api_key="sk-test-123")
    
    # Mock API error
    service.client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))
    
    result = await service.generate("test")
    
    assert result is None


@pytest.mark.asyncio
async def test_generate_batch_success(mock_openai_response):
    """Test successful batch embedding generation"""
    service = EmbeddingService(api_key="sk-test-123")
    
    # Mock batch response (3 embeddings)
    mock_batch_response = Mock()
    mock_batch_response.data = [
        Mock(embedding=[0.1] * 1536),
        Mock(embedding=[0.2] * 1536),
        Mock(embedding=[0.3] * 1536)
    ]
    
    service.client.embeddings.create = AsyncMock(return_value=mock_batch_response)
    
    texts = ["I love steaks", "I love wagyu", "What about ribeye?"]
    results = await service.generate_batch(texts)
    
    assert len(results) == 3
    assert all(r is not None for r in results)
    assert all(len(r) == 1536 for r in results)
    
    # Verify API call
    service.client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=texts
    )


@pytest.mark.asyncio
async def test_generate_batch_api_error():
    """Test batch generation handles API errors"""
    service = EmbeddingService(api_key="sk-test-123")
    
    # Mock API error
    service.client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))
    
    texts = ["test1", "test2", "test3"]
    results = await service.generate_batch(texts)
    
    # Should return None for all texts
    assert len(results) == 3
    assert all(r is None for r in results)


@pytest.mark.asyncio
async def test_get_embedding_dimension_small():
    """Test get_embedding_dimension for small model"""
    service = EmbeddingService(api_key="sk-test-123", model="text-embedding-3-small")
    
    assert service.get_embedding_dimension() == 1536


@pytest.mark.asyncio
async def test_get_embedding_dimension_large():
    """Test get_embedding_dimension for large model"""
    service = EmbeddingService(api_key="sk-test-123", model="text-embedding-3-large")
    
    assert service.get_embedding_dimension() == 3072


@pytest.mark.asyncio
async def test_generate_empty_text(mock_openai_response):
    """Test embedding generation with empty text"""
    service = EmbeddingService(api_key="sk-test-123")
    service.client.embeddings.create = AsyncMock(return_value=mock_openai_response)
    
    result = await service.generate("")
    
    assert result is not None  # OpenAI handles empty strings


@pytest.mark.asyncio
async def test_generate_long_text(mock_openai_response):
    """Test embedding generation with very long text"""
    service = EmbeddingService(api_key="sk-test-123")
    service.client.embeddings.create = AsyncMock(return_value=mock_openai_response)
    
    long_text = "I love steaks " * 1000
    result = await service.generate(long_text)
    
    assert result is not None

