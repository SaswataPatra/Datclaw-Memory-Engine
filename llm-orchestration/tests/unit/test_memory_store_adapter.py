"""
Unit tests for Memory Store Adapter Layer.

Tests the abstraction layer (Protocol, Factory, REST client) to ensure
transport-agnostic design works correctly.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

from adapters.memory_store_protocol import (
    MemoryStoreProtocol,
    MemoryStoreError,
    MemoryNotFoundError,
    MemoryStoreConnectionError,
    MemoryStoreTimeoutError
)
from adapters.memory_store_factory import MemoryStoreFactory, get_memory_store
from adapters.rest.memory_client import RestMemoryClient


class TestMemoryStoreProtocol:
    """Test Protocol definition"""
    
    def test_protocol_has_required_methods(self):
        """Protocol should define all required methods"""
        required_methods = [
            'get_memory',
            'get_user_memories',
            'get_related_memories',
            'record_access',
            'search_memories',
            'get_expiring_memories',
            'health_check',
            'close'
        ]
        
        for method in required_methods:
            assert hasattr(MemoryStoreProtocol, method)
    
    def test_mock_implements_protocol(self):
        """AsyncMock with spec should satisfy protocol"""
        mock_store = AsyncMock(spec=MemoryStoreProtocol)
        
        # Should have all protocol methods
        assert hasattr(mock_store, 'get_memory')
        assert hasattr(mock_store, 'get_user_memories')
        assert hasattr(mock_store, 'health_check')


class TestMemoryStoreFactory:
    """Test Factory pattern"""
    
    def test_create_rest_client(self):
        """Factory should create REST client"""
        client = MemoryStoreFactory.create(transport="rest")
        
        assert isinstance(client, RestMemoryClient)
        assert client.base_url == "http://localhost:8080"
    
    def test_create_rest_with_custom_url(self):
        """Factory should respect custom URL"""
        client = MemoryStoreFactory.create(
            transport="rest",
            base_url="http://prod-server:8080"
        )
        
        assert client.base_url == "http://prod-server:8080"
    
    def test_create_grpc_client_not_implemented(self):
        """Factory should raise ImportError for gRPC in Phase 1"""
        with pytest.raises(ImportError, match="gRPC client not available"):
            MemoryStoreFactory.create(transport="grpc")
    
    def test_create_unknown_transport(self):
        """Factory should reject unknown transport"""
        with pytest.raises(ValueError, match="Unknown transport"):
            MemoryStoreFactory.create(transport="websocket")
    
    def test_create_from_config_rest(self):
        """Factory should create from config dict"""
        config = {
            "transport": "rest",
            "rest": {
                "base_url": "http://test:8080",
                "timeout": 5.0
            }
        }
        
        client = MemoryStoreFactory.create_from_config(config)
        
        assert isinstance(client, RestMemoryClient)
        assert client.base_url == "http://test:8080"
        assert client.timeout == 5.0
    
    def test_get_memory_store_default(self):
        """Convenience function should use REST by default"""
        client = get_memory_store()
        
        assert isinstance(client, RestMemoryClient)
    
    def test_get_memory_store_with_config(self):
        """Convenience function should accept config"""
        config = {
            "transport": "rest",
            "rest": {"base_url": "http://custom:8080"}
        }
        
        client = get_memory_store(config=config)
        
        assert isinstance(client, RestMemoryClient)
        assert client.base_url == "http://custom:8080"


class TestRestMemoryClient:
    """Test REST client implementation"""
    
    @pytest.fixture
    def client(self):
        """Create REST client for testing"""
        return RestMemoryClient(base_url="http://test:8080", timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_get_memory_success(self, client):
        """Should retrieve memory by ID"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node_id": "mem123",
            "content": "test memory",
            "ego_score": 0.8
        }
        mock_response.raise_for_status = Mock()
        
        with patch.object(client.client, 'get', return_value=mock_response):
            result = await client.get_memory("mem123")
        
        assert result["node_id"] == "mem123"
        assert result["ego_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, client):
        """Should return None for 404"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(client.client, 'get', return_value=mock_response):
            result = await client.get_memory("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_user_memories(self, client):
        """Should retrieve user memories with filters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"node_id": "mem1", "tier": 1},
            {"node_id": "mem2", "tier": 1}
        ]
        mock_response.raise_for_status = Mock()
        
        with patch.object(client.client, 'get', return_value=mock_response):
            results = await client.get_user_memories("user123", tier=1, limit=10)
        
        assert len(results) == 2
        assert results[0]["node_id"] == "mem1"
    
    @pytest.mark.asyncio
    async def test_get_user_memories_object_response(self, client):
        """Should handle response with 'memories' key"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "memories": [{"node_id": "mem1"}],
            "total_count": 1
        }
        mock_response.raise_for_status = Mock()
        
        with patch.object(client.client, 'get', return_value=mock_response):
            results = await client.get_user_memories("user123")
        
        assert len(results) == 1
        assert results[0]["node_id"] == "mem1"
    
    @pytest.mark.asyncio
    async def test_record_access_success(self, client):
        """Should record access successfully"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        with patch.object(client.client, 'post', return_value=mock_response):
            result = await client.record_access("mem123")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_record_access_failure(self, client):
        """Should return False on failure"""
        with patch.object(client.client, 'post', side_effect=Exception("Network error")):
            result = await client.record_access("mem123")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Should return True for healthy service"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(client.client, 'get', return_value=mock_response):
            result = await client.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        """Should return False for unhealthy service"""
        with patch.object(client.client, 'get', side_effect=Exception("Connection refused")):
            result = await client.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_close(self, client):
        """Should close HTTP client"""
        with patch.object(client.client, 'aclose', new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()


class TestTransportAgnosticUsage:
    """Test that business logic works with any transport"""
    
    @pytest.mark.asyncio
    async def test_mock_protocol_usage(self):
        """Business logic should work with mocked protocol"""
        # Create mock that satisfies protocol
        mock_store = AsyncMock(spec=MemoryStoreProtocol)
        mock_store.get_user_memories.return_value = [
            {"node_id": "mem1", "content": "test", "ego_score": 0.9}
        ]
        
        # Simulated business logic (like TwoStageRetriever)
        memories = await mock_store.get_user_memories("user123", tier=1)
        
        assert len(memories) == 1
        assert memories[0]["ego_score"] == 0.9
        mock_store.get_user_memories.assert_called_once_with("user123", tier=1)
    
    def test_client_satisfies_protocol(self):
        """REST client should satisfy protocol contract"""
        client = RestMemoryClient()
        
        # Check all protocol methods exist
        assert callable(getattr(client, 'get_memory'))
        assert callable(getattr(client, 'get_user_memories'))
        assert callable(getattr(client, 'get_related_memories'))
        assert callable(getattr(client, 'record_access'))
        assert callable(getattr(client, 'search_memories'))
        assert callable(getattr(client, 'get_expiring_memories'))
        assert callable(getattr(client, 'health_check'))
        assert callable(getattr(client, 'close'))
    
    @pytest.mark.asyncio
    async def test_swap_implementations(self):
        """Should be able to swap REST → gRPC without code changes"""
        # Create REST client
        rest_client = MemoryStoreFactory.create(transport="rest")
        
        # Use it (simulated business logic)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"node_id": "mem1"}]
        mock_response.raise_for_status = Mock()
        
        with patch.object(rest_client.client, 'get', return_value=mock_response):
            memories_from_rest = await rest_client.get_user_memories("user123")
        
        assert len(memories_from_rest) == 1
        
        # In Phase 1.5, this would work:
        # grpc_client = MemoryStoreFactory.create(transport="grpc")
        # memories_from_grpc = await grpc_client.get_user_memories("user123")
        # 
        # Business logic using MemoryStoreProtocol would work with BOTH!


class TestErrorHandling:
    """Test error handling across transport"""
    
    def test_exception_hierarchy(self):
        """Error classes should inherit correctly"""
        assert issubclass(MemoryNotFoundError, MemoryStoreError)
        assert issubclass(MemoryStoreConnectionError, MemoryStoreError)
        assert issubclass(MemoryStoreTimeoutError, MemoryStoreError)
    
    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Should raise MemoryStoreTimeoutError on timeout"""
        client = RestMemoryClient(timeout=0.001)
        
        # Mock a timeout
        import httpx
        with patch.object(client.client, 'get', side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(MemoryStoreTimeoutError):
                await client.get_memory("mem123")
    
    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Should raise MemoryStoreConnectionError on connection failure"""
        client = RestMemoryClient()
        
        with patch.object(client.client, 'get', side_effect=Exception("Connection refused")):
            with pytest.raises(MemoryStoreConnectionError):
                await client.get_memory("mem123")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

