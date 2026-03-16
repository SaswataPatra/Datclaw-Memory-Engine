"""
Unit Tests for TrainingDataCollector

Tests:
1. Database initialization
2. Semantic correction logging
3. Label discovery logging
4. Confidence anomaly logging
5. Dataset export
6. Statistics retrieval
7. HuggingFace format export
"""

import pytest
import os
import tempfile
import json
from pathlib import Path

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
def collector(temp_db):
    """Create a TrainingDataCollector instance with temp database."""
    return TrainingDataCollector(db_path=temp_db)


class TestDatabaseInitialization:
    """Test database creation and schema."""
    
    def test_database_created(self, temp_db, collector):
        """Test that database file is created."""
        assert os.path.exists(temp_db)
    
    def test_tables_created(self, collector):
        """Test that all required tables are created."""
        import sqlite3
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        assert 'classification_corrections' in tables
        assert 'discovered_labels' in tables
        assert 'confidence_anomalies' in tables
        
        conn.close()
    
    def test_indexes_created(self, collector):
        """Test that indexes are created for performance."""
        import sqlite3
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}
        
        assert 'idx_corrections_timestamp' in indexes
        assert 'idx_corrections_source' in indexes
        assert 'idx_labels_name' in indexes
        assert 'idx_anomalies_timestamp' in indexes
        
        conn.close()


class TestSemanticCorrectionLogging:
    """Test logging of semantic validation corrections."""
    
    def test_log_semantic_correction(self, collector):
        """Test logging a semantic correction."""
        correction_id = collector.log_semantic_correction(
            text="can you tell me the best player in this space?",
            predicted_labels=["opinion", "preference", "technology_opinion"],
            invalid_labels=["preference"],
            scores={"opinion": 0.94, "preference": 0.89, "technology_opinion": 0.58},
            reasoning="preference implies personal taste, but this is asking for expert opinion",
            user_id="test_user",
            session_id="test_session",
            classifier_confidence=0.94,
            routing_decision="semantic_check"
        )
        
        assert correction_id is not None
        assert len(correction_id) == 36  # UUID length
    
    def test_semantic_correction_stored(self, collector):
        """Test that semantic correction is properly stored in database."""
        collector.log_semantic_correction(
            text="test message",
            predicted_labels=["label1", "label2"],
            invalid_labels=["label2"],
            scores={"label1": 0.9, "label2": 0.7},
            reasoning="test reasoning",
            classifier_confidence=0.9
        )
        
        # Retrieve from database
        dataset = collector.get_training_dataset()
        
        assert len(dataset) == 1
        assert dataset[0]['text'] == "test message"
        assert dataset[0]['labels'] == ["label1"]  # label2 filtered out
        assert dataset[0]['source'] == 'semantic_validator'
        assert dataset[0]['reasoning'] == "test reasoning"
        assert dataset[0]['confidence'] == 0.9
    
    def test_multiple_corrections(self, collector):
        """Test logging multiple corrections."""
        for i in range(5):
            collector.log_semantic_correction(
                text=f"test message {i}",
                predicted_labels=["label1", "label2"],
                invalid_labels=["label2"],
                scores={"label1": 0.9, "label2": 0.7},
                reasoning=f"test reasoning {i}",
                classifier_confidence=0.9
            )
        
        dataset = collector.get_training_dataset()
        assert len(dataset) == 5


class TestLabelDiscoveryLogging:
    """Test logging of label discovery events."""
    
    def test_log_label_discovery(self, collector):
        """Test logging a label discovery event."""
        discovery_id, label_ids = collector.log_label_discovery(
            text="I have 1 cat",
            existing_labels=["preference", "opinion"],
            discovered_labels=[
                {"name": "pet_info", "importance": 0.80},
                {"name": "animal_affinity", "importance": 0.75}
            ],
            user_id="test_user",
            session_id="test_session"
        )
        
        assert discovery_id is not None
        assert len(label_ids) == 2
    
    def test_discovered_labels_stored(self, collector):
        """Test that discovered labels are stored in separate table."""
        collector.log_label_discovery(
            text="I have 1 cat",
            existing_labels=["preference"],
            discovered_labels=[
                {"name": "pet_info", "importance": 0.80},
                {"name": "animal_affinity", "importance": 0.75}
            ]
        )
        
        # Check correction log
        dataset = collector.get_training_dataset(source_filter=['label_discovery'])
        assert len(dataset) == 1
        assert dataset[0]['labels'] == ["pet_info", "animal_affinity"]
        
        # Check discovered labels table
        import sqlite3
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT label_name, importance FROM discovered_labels")
        labels = cursor.fetchall()
        conn.close()
        
        assert len(labels) == 2
        assert ("pet_info", 0.80) in labels
        assert ("animal_affinity", 0.75) in labels


class TestConfidenceAnomalyLogging:
    """Test logging of confidence distribution anomalies."""
    
    def test_log_confidence_anomaly(self, collector):
        """Test logging a confidence anomaly."""
        anomaly_id = collector.log_confidence_anomaly(
            text="or is it just me?",
            predicted_labels=["opinion", "preference", "event", "identity"],
            scores={"opinion": 0.7, "preference": 0.68, "event": 0.65, "identity": 0.62},
            issue_type="too_many_labels",
            issue_description="Too many labels (4) above threshold",
            user_id="test_user"
        )
        
        assert anomaly_id is not None
        assert len(anomaly_id) == 36
    
    def test_anomaly_stored(self, collector):
        """Test that anomaly is properly stored."""
        collector.log_confidence_anomaly(
            text="test message",
            predicted_labels=["label1", "label2", "label3"],
            scores={"label1": 0.6, "label2": 0.58, "label3": 0.56},
            issue_type="suspicious_distribution",
            issue_description="Multiple mid-range labels"
        )
        
        # Check stats
        stats = collector.get_stats()
        assert stats['total_anomalies'] == 1


class TestDatasetExport:
    """Test dataset export functionality."""
    
    def test_get_training_dataset(self, collector):
        """Test retrieving training dataset."""
        # Add some corrections
        collector.log_semantic_correction(
            text="test 1",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.9},
            reasoning="test",
            classifier_confidence=0.9
        )
        
        dataset = collector.get_training_dataset()
        assert len(dataset) == 1
        assert 'text' in dataset[0]
        assert 'labels' in dataset[0]
        assert 'scores' in dataset[0]
    
    def test_filter_by_confidence(self, collector):
        """Test filtering dataset by confidence threshold."""
        # Add corrections with different confidences
        collector.log_semantic_correction(
            text="high confidence",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.95},
            reasoning="test",
            classifier_confidence=0.95
        )
        
        collector.log_semantic_correction(
            text="low confidence",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.6},
            reasoning="test",
            classifier_confidence=0.6
        )
        
        # Filter by confidence
        high_conf_dataset = collector.get_training_dataset(min_confidence=0.8)
        assert len(high_conf_dataset) == 1
        assert high_conf_dataset[0]['text'] == "high confidence"
    
    def test_filter_by_source(self, collector):
        """Test filtering dataset by source."""
        # Add different sources
        collector.log_semantic_correction(
            text="semantic correction",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.9},
            reasoning="test",
            classifier_confidence=0.9
        )
        
        collector.log_label_discovery(
            text="label discovery",
            existing_labels=[],
            discovered_labels=[{"name": "new_label", "importance": 0.8}]
        )
        
        # Filter by source
        semantic_dataset = collector.get_training_dataset(source_filter=['semantic_validator'])
        assert len(semantic_dataset) == 1
        assert semantic_dataset[0]['text'] == "semantic correction"
        
        discovery_dataset = collector.get_training_dataset(source_filter=['label_discovery'])
        assert len(discovery_dataset) == 1
        assert discovery_dataset[0]['text'] == "label discovery"
    
    def test_limit_results(self, collector):
        """Test limiting number of results."""
        # Add 10 corrections
        for i in range(10):
            collector.log_semantic_correction(
                text=f"test {i}",
                predicted_labels=["label1"],
                invalid_labels=[],
                scores={"label1": 0.9},
                reasoning="test",
                classifier_confidence=0.9
            )
        
        # Limit to 5
        dataset = collector.get_training_dataset(limit=5)
        assert len(dataset) == 5


class TestHuggingFaceExport:
    """Test export to HuggingFace format."""
    
    def test_export_to_huggingface_format(self, collector, temp_db):
        """Test exporting to HuggingFace JSONL format."""
        # Add some corrections
        collector.log_semantic_correction(
            text="I love Python",
            predicted_labels=["preference", "programming_affinity"],
            invalid_labels=[],
            scores={"preference": 0.95, "programming_affinity": 0.88},
            reasoning="test",
            classifier_confidence=0.95
        )
        
        # Export
        output_path = temp_db.replace('.db', '.jsonl')
        collector.export_to_huggingface_format(output_path, min_confidence=0.7)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify format
        with open(output_path, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            
            assert 'text' in data
            assert 'labels' in data
            assert 'label_scores' in data
            assert data['text'] == "I love Python"
            assert data['labels'] == ["preference", "programming_affinity"]
            assert len(data['label_scores']) == 2
        
        # Cleanup
        os.remove(output_path)


class TestStatistics:
    """Test statistics retrieval."""
    
    def test_get_stats_empty(self, collector):
        """Test getting stats from empty database."""
        stats = collector.get_stats()
        
        assert stats['total_corrections'] == 0
        assert stats['total_discovered_labels'] == 0
        assert stats['total_anomalies'] == 0
        assert stats['recent_corrections_24h'] == 0
    
    def test_get_stats_with_data(self, collector):
        """Test getting stats with data."""
        # Add various types of data
        collector.log_semantic_correction(
            text="test 1",
            predicted_labels=["label1", "label2"],
            invalid_labels=["label2"],
            scores={"label1": 0.9, "label2": 0.7},
            reasoning="test",
            classifier_confidence=0.9
        )
        
        collector.log_label_discovery(
            text="test 2",
            existing_labels=[],
            discovered_labels=[{"name": "new_label", "importance": 0.8}]
        )
        
        collector.log_confidence_anomaly(
            text="test 3",
            predicted_labels=["label1", "label2"],
            scores={"label1": 0.6, "label2": 0.58},
            issue_type="suspicious_distribution",
            issue_description="test"
        )
        
        stats = collector.get_stats()
        
        assert stats['total_corrections'] == 2
        assert stats['total_discovered_labels'] == 1
        assert stats['total_anomalies'] == 1
        assert stats['by_source']['semantic_validator'] == 1
        assert stats['by_source']['label_discovery'] == 1
    
    def test_top_invalid_labels(self, collector):
        """Test tracking of most common invalid labels."""
        # Add corrections with same invalid label
        for i in range(3):
            collector.log_semantic_correction(
                text=f"test {i}",
                predicted_labels=["preference", "opinion"],
                invalid_labels=["preference"],
                scores={"preference": 0.7, "opinion": 0.9},
                reasoning="test",
                classifier_confidence=0.9
            )
        
        # Add correction with different invalid label
        collector.log_semantic_correction(
            text="test 4",
            predicted_labels=["event", "opinion"],
            invalid_labels=["event"],
            scores={"event": 0.6, "opinion": 0.9},
            reasoning="test",
            classifier_confidence=0.9
        )
        
        stats = collector.get_stats()
        
        assert 'top_invalid_labels' in stats
        assert stats['top_invalid_labels']['preference'] == 3
        assert stats['top_invalid_labels']['event'] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_labels(self, collector):
        """Test handling of empty label lists."""
        correction_id = collector.log_semantic_correction(
            text="test",
            predicted_labels=[],
            invalid_labels=[],
            scores={},
            reasoning="test",
            classifier_confidence=None
        )
        
        assert correction_id is not None
    
    def test_none_values(self, collector):
        """Test handling of None values."""
        correction_id = collector.log_semantic_correction(
            text="test",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.9},
            reasoning="test",
            user_id=None,
            session_id=None,
            classifier_confidence=None,
            routing_decision=None
        )
        
        assert correction_id is not None
    
    def test_special_characters(self, collector):
        """Test handling of special characters in text."""
        correction_id = collector.log_semantic_correction(
            text="Test with special chars: 你好, émojis 🎉, and \"quotes\"",
            predicted_labels=["label1"],
            invalid_labels=[],
            scores={"label1": 0.9},
            reasoning="test with 'quotes' and \"double quotes\"",
            classifier_confidence=0.9
        )
        
        assert correction_id is not None
        
        dataset = collector.get_training_dataset()
        assert len(dataset) == 1
        assert "你好" in dataset[0]['text']
        assert "🎉" in dataset[0]['text']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

