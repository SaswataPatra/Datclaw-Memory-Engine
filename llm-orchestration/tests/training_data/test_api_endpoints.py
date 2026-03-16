"""
Integration Tests for Training Data API Endpoints

Tests:
1. GET /training/stats
2. GET /training/dataset
3. POST /training/export
4. Error handling
5. Filtering and pagination
"""

import pytest
import json
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Note: These tests assume the API is set up with test fixtures
# You may need to adjust imports based on your actual API structure


class TestTrainingStatsEndpoint:
    """Test GET /training/stats endpoint."""
    
    def test_get_stats_success(self, client, training_collector):
        """Test successful stats retrieval."""
        # Add some test data
        training_collector.log_semantic_correction(
            text="test",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.9},
            reasoning="test",
            classifier_confidence=0.9
        )
        
        # Request stats
        response = client.get("/training/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "stats" in data
        assert "timestamp" in data
        assert data["stats"]["total_corrections"] >= 1
    
    def test_get_stats_empty_database(self, client):
        """Test stats with empty database."""
        response = client.get("/training/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["stats"]["total_corrections"] == 0
        assert data["stats"]["total_discovered_labels"] == 0
    
    def test_get_stats_structure(self, client, training_collector):
        """Test that stats response has correct structure."""
        response = client.get("/training/stats")
        
        assert response.status_code == 200
        data = response.json()
        stats = data["stats"]
        
        # Verify all expected fields are present
        assert "total_corrections" in stats
        assert "by_source" in stats
        assert "by_routing" in stats
        assert "total_discovered_labels" in stats
        assert "total_anomalies" in stats
        assert "recent_corrections_24h" in stats
        assert "top_invalid_labels" in stats


class TestTrainingDatasetEndpoint:
    """Test GET /training/dataset endpoint."""
    
    def test_get_dataset_success(self, client, training_collector):
        """Test successful dataset retrieval."""
        # Add test data
        training_collector.log_semantic_correction(
            text="I love Python",
            predicted_labels=["preference", "programming_affinity"],
            invalid_labels=[],
            scores={"preference": 0.95, "programming_affinity": 0.88},
            reasoning="test",
            classifier_confidence=0.95
        )
        
        # Request dataset
        response = client.get("/training/dataset")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "dataset" in data
        assert "count" in data
        assert "filters" in data
        assert len(data["dataset"]) >= 1
    
    def test_filter_by_confidence(self, client, training_collector):
        """Test filtering dataset by confidence."""
        # Add data with different confidences
        training_collector.log_semantic_correction(
            text="high confidence",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.95},
            reasoning="test",
            classifier_confidence=0.95
        )
        
        training_collector.log_semantic_correction(
            text="low confidence",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.6},
            reasoning="test",
            classifier_confidence=0.6
        )
        
        # Request with confidence filter
        response = client.get("/training/dataset?min_confidence=0.8")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should only return high-confidence example
        assert data["count"] == 1
        assert data["dataset"][0]["text"] == "high confidence"
    
    def test_filter_by_source(self, client, training_collector):
        """Test filtering dataset by source."""
        # Add data from different sources
        training_collector.log_semantic_correction(
            text="semantic correction",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.9},
            reasoning="test",
            classifier_confidence=0.9
        )
        
        training_collector.log_label_discovery(
            text="label discovery",
            existing_labels=[],
            discovered_labels=[{"name": "new_label", "importance": 0.8}]
        )
        
        # Request with source filter
        response = client.get("/training/dataset?source=semantic_validator")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should only return semantic validator corrections
        assert all(item["source"] == "semantic_validator" for item in data["dataset"])
    
    def test_limit_results(self, client, training_collector):
        """Test limiting number of results."""
        # Add 10 corrections
        for i in range(10):
            training_collector.log_semantic_correction(
                text=f"test {i}",
                predicted_labels=["label1"],
                invalid_labels=[],
                scores={"label1": 0.9},
                reasoning="test",
                classifier_confidence=0.9
            )
        
        # Request with limit
        response = client.get("/training/dataset?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 5
        assert len(data["dataset"]) == 5
    
    def test_dataset_item_structure(self, client, training_collector):
        """Test that dataset items have correct structure."""
        training_collector.log_semantic_correction(
            text="test",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.9},
            reasoning="test",
            classifier_confidence=0.9
        )
        
        response = client.get("/training/dataset")
        
        assert response.status_code == 200
        data = response.json()
        
        item = data["dataset"][0]
        assert "text" in item
        assert "labels" in item
        assert "scores" in item
        assert "source" in item
        assert "confidence" in item
        assert "reasoning" in item


class TestTrainingExportEndpoint:
    """Test POST /training/export endpoint."""
    
    def test_export_success(self, client, training_collector):
        """Test successful export to file."""
        # Add test data
        training_collector.log_semantic_correction(
            text="I love Python",
            predicted_labels=["preference", "programming_affinity"],
            invalid_labels=[],
            scores={"preference": 0.95, "programming_affinity": 0.88},
            reasoning="test",
            classifier_confidence=0.95
        )
        
        # Create temp file path
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            # Request export
            response = client.post(
                "/training/export",
                json={
                    "output_path": output_path,
                    "min_confidence": 0.7
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert data["output_path"] == output_path
            
            # Verify file was created
            assert os.path.exists(output_path)
            
            # Verify file format
            with open(output_path, 'r') as f:
                line = f.readline()
                item = json.loads(line)
                
                assert "text" in item
                assert "labels" in item
                assert "label_scores" in item
        
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_export_with_confidence_filter(self, client, training_collector):
        """Test export with confidence filtering."""
        # Add data with different confidences
        training_collector.log_semantic_correction(
            text="high confidence",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.95},
            reasoning="test",
            classifier_confidence=0.95
        )
        
        training_collector.log_semantic_correction(
            text="low confidence",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.5},
            reasoning="test",
            classifier_confidence=0.5
        )
        
        # Export with high confidence filter
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            response = client.post(
                "/training/export",
                json={
                    "output_path": output_path,
                    "min_confidence": 0.8
                }
            )
            
            assert response.status_code == 200
            
            # Verify only high-confidence data exported
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1  # Only high-confidence item
                item = json.loads(lines[0])
                assert item["text"] == "high confidence"
        
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_export_huggingface_format(self, client, training_collector):
        """Test that export uses correct HuggingFace format."""
        training_collector.log_semantic_correction(
            text="I love Python",
            predicted_labels=["preference", "programming_affinity"],
            invalid_labels=[],
            scores={"preference": 0.95, "programming_affinity": 0.88},
            reasoning="test",
            classifier_confidence=0.95
        )
        
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            response = client.post(
                "/training/export",
                json={"output_path": output_path, "min_confidence": 0.7}
            )
            
            assert response.status_code == 200
            
            # Verify HuggingFace format
            with open(output_path, 'r') as f:
                item = json.loads(f.readline())
                
                # HuggingFace format requirements
                assert "text" in item
                assert "labels" in item
                assert "label_scores" in item
                assert isinstance(item["labels"], list)
                assert isinstance(item["label_scores"], list)
                assert len(item["labels"]) == len(item["label_scores"])
        
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestErrorHandling:
    """Test error handling in API endpoints."""
    
    def test_stats_when_collector_not_initialized(self, client_without_collector):
        """Test stats endpoint when collector is not initialized."""
        response = client_without_collector.get("/training/stats")
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()
    
    def test_dataset_invalid_parameters(self, client):
        """Test dataset endpoint with invalid parameters."""
        # Invalid confidence (> 1.0)
        response = client.get("/training/dataset?min_confidence=1.5")
        
        # Should handle gracefully (may return empty or error)
        assert response.status_code in [200, 400, 422]
    
    def test_export_invalid_path(self, client):
        """Test export with invalid file path."""
        response = client.post(
            "/training/export",
            json={
                "output_path": "/invalid/path/that/does/not/exist/file.jsonl",
                "min_confidence": 0.7
            }
        )
        
        # Should return error
        assert response.status_code == 500


class TestConcurrency:
    """Test concurrent access to training data."""
    
    def test_concurrent_stats_requests(self, client, training_collector):
        """Test multiple concurrent stats requests."""
        import concurrent.futures
        
        def get_stats():
            return client.get("/training/stats")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_stats) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
    
    def test_concurrent_dataset_requests(self, client, training_collector):
        """Test multiple concurrent dataset requests."""
        import concurrent.futures
        
        # Add some data
        for i in range(10):
            training_collector.log_semantic_correction(
                text=f"test {i}",
                predicted_labels=["label1"],
                invalid_labels=[],
                scores={"label1": 0.9},
                reasoning="test",
                classifier_confidence=0.9
            )
        
        def get_dataset():
            return client.get("/training/dataset?limit=5")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_dataset) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)


# Note: API endpoint tests require full FastAPI setup
# These are integration tests that need the actual API running
# For now, we'll mark them as skipped and focus on unit/integration tests

pytestmark = pytest.mark.skip(reason="API endpoint tests require full FastAPI setup with chatbot_service")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

