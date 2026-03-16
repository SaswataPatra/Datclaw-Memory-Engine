"""
Unit tests for RelationClassifier and RelationTrainingCollector.

Tests:
1. Relation classification with heuristics
2. Relation category mapping
3. Training data collection
4. Discovered relation types
"""

import pytest
import tempfile
import os
from core.graph.relation_classifier import (
    RelationClassifier,
    RelationResult,
    RELATION_CATEGORIES,
    get_all_relation_types,
    get_relation_category,
)
from core.graph.relation_training_collector import RelationTrainingCollector


class TestRelationClassifier:
    """Test RelationClassifier functionality."""
    
    @pytest.fixture
    def classifier(self):
        """Create a RelationClassifier instance."""
        return RelationClassifier(config={})
    
    def test_init(self, classifier):
        """Test classifier initialization."""
        assert classifier is not None
        assert len(classifier.relation_types) > 0
        assert classifier.confidence_threshold == 0.5
    
    def test_all_relation_types(self, classifier):
        """Test getting all relation types."""
        types = classifier.all_relation_types
        assert "sister_of" in types
        assert "works_at" in types
        assert "likes" in types
        assert "located_at" in types
    
    def test_add_discovered_type(self, classifier):
        """Test adding a discovered relation type."""
        classifier.add_discovered_type("mentors", "professional")
        assert "mentors" in classifier.discovered_types
        assert "mentors" in classifier.all_relation_types
    
    def test_heuristic_family_sister(self, classifier):
        """Test heuristic classification for sister relation."""
        result = classifier.classify_with_heuristics(
            subject="Sarah",
            object_text="me",
            context="Sarah is my sister and she lives in New York."
        )
        
        assert result.relation == "sister_of"
        assert result.category == "family"
        assert result.source == "heuristic"
        assert 0 < result.confidence <= 1
    
    def test_heuristic_family_brother(self, classifier):
        """Test heuristic classification for brother relation."""
        result = classifier.classify_with_heuristics(
            subject="John",
            object_text="me",
            context="John is my brother."
        )
        
        assert result.relation == "brother_of"
        assert result.category == "family"
    
    def test_heuristic_works_at(self, classifier):
        """Test heuristic classification for works_at relation."""
        result = classifier.classify_with_heuristics(
            subject="Sarah",
            object_text="Google",
            context="Sarah works at Google."
        )
        
        assert result.relation == "works_at"
        assert result.category == "professional"
    
    def test_heuristic_colleague(self, classifier):
        """Test heuristic classification for colleague relation."""
        result = classifier.classify_with_heuristics(
            subject="John",
            object_text="Sarah",
            context="John is my colleague at work."
        )
        
        assert result.relation == "colleague_of"
        assert result.category == "professional"
    
    def test_heuristic_friend(self, classifier):
        """Test heuristic classification for friend relation."""
        result = classifier.classify_with_heuristics(
            subject="Mike",
            object_text="me",
            context="Mike is my best friend."
        )
        
        assert result.relation == "friend_of"
        assert result.category == "personal"
    
    def test_heuristic_likes(self, classifier):
        """Test heuristic classification for likes relation."""
        result = classifier.classify_with_heuristics(
            subject="I",
            object_text="pizza",
            context="I really like pizza."
        )
        
        assert result.relation == "likes"
        assert result.category == "personal"
    
    def test_heuristic_dislikes(self, classifier):
        """Test heuristic classification for dislikes relation."""
        result = classifier.classify_with_heuristics(
            subject="I",
            object_text="spiders",
            context="I hate spiders."
        )
        
        assert result.relation == "dislikes"
        assert result.category == "personal"
    
    def test_heuristic_lives_in(self, classifier):
        """Test heuristic classification for lives_in relation."""
        result = classifier.classify_with_heuristics(
            subject="Sarah",
            object_text="New York",
            context="Sarah lives in New York."
        )
        
        assert result.relation == "lives_in"
        assert result.category == "factual"
    
    def test_heuristic_default(self, classifier):
        """Test heuristic classification falls back to 'knows'."""
        result = classifier.classify_with_heuristics(
            subject="Sarah",
            object_text="John",
            context="Sarah and John were at the meeting."
        )
        
        assert result.relation == "knows"
        assert result.category == "personal"
        assert result.confidence == 0.3  # Low confidence for default


class TestRelationResult:
    """Test RelationResult dataclass."""
    
    def test_create_result(self):
        """Test creating a RelationResult."""
        result = RelationResult(
            subject="Sarah",
            object="Google",
            relation="works_at",
            category="professional",
            confidence=0.85
        )
        
        assert result.subject == "Sarah"
        assert result.object == "Google"
        assert result.relation == "works_at"
        assert result.category == "professional"
        assert result.confidence == 0.85
        assert result.is_discovered is False
        assert result.source == "deberta"
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = RelationResult(
            subject="Sarah",
            object="Google",
            relation="works_at",
            category="professional",
            confidence=0.85,
            source="llm",
            metadata={"reasoning": "test"}
        )
        
        d = result.to_dict()
        
        assert d["subject"] == "Sarah"
        assert d["object"] == "Google"
        assert d["relation"] == "works_at"
        assert d["category"] == "professional"
        assert d["confidence"] == 0.85
        assert d["source"] == "llm"
        assert d["metadata"]["reasoning"] == "test"


class TestRelationCategories:
    """Test relation category functions."""
    
    def test_get_all_relation_types(self):
        """Test getting all relation types."""
        types = get_all_relation_types()
        
        assert len(types) > 20
        assert "sister_of" in types
        assert "works_at" in types
        assert "likes" in types
    
    def test_get_relation_category(self):
        """Test getting category for relation types."""
        assert get_relation_category("sister_of") == "family"
        assert get_relation_category("works_at") == "professional"
        assert get_relation_category("likes") == "personal"
        assert get_relation_category("contradicts") == "temporal"
        assert get_relation_category("located_at") == "factual"
        assert get_relation_category("unknown_relation") == "other"
    
    def test_relation_categories_structure(self):
        """Test RELATION_CATEGORIES structure."""
        assert "family" in RELATION_CATEGORIES
        assert "professional" in RELATION_CATEGORIES
        assert "personal" in RELATION_CATEGORIES
        assert "temporal" in RELATION_CATEGORIES
        assert "factual" in RELATION_CATEGORIES
        assert "other" in RELATION_CATEGORIES


class TestRelationTrainingCollector:
    """Test RelationTrainingCollector functionality."""
    
    @pytest.fixture
    def collector(self):
        """Create a RelationTrainingCollector with temp database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        collector = RelationTrainingCollector(db_path=db_path)
        yield collector
        
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass
    
    def test_init(self, collector):
        """Test collector initialization."""
        assert collector is not None
        assert collector.db_path.endswith('.db')
    
    def test_log_extraction(self, collector):
        """Test logging a relation extraction."""
        extraction_id = collector.log_extraction(
            subject="Sarah",
            object_text="Google",
            context="Sarah works at Google.",
            predicted_relation="works_at",
            confidence=0.85,
            category="professional",
            source="deberta",
            user_id="user123"
        )
        
        assert extraction_id is not None
        assert len(extraction_id) == 36  # UUID format
    
    def test_log_discovered_relation(self, collector):
        """Test logging a discovered relation type."""
        discovery_id = collector.log_discovered_relation(
            relation_name="mentors",
            category="professional",
            example_context="John mentors Sarah at work.",
            example_subject="John",
            example_object="Sarah"
        )
        
        assert discovery_id is not None
        
        # Check it was stored
        discovered = collector.get_discovered_relations()
        assert len(discovered) == 1
        assert discovered[0]["relation_name"] == "mentors"
    
    def test_log_correction(self, collector):
        """Test logging a relation correction."""
        # First log an extraction
        extraction_id = collector.log_extraction(
            subject="Sarah",
            object_text="John",
            context="Sarah and John are colleagues.",
            predicted_relation="knows",
            confidence=0.5,
            category="personal",
            source="heuristic"
        )
        
        # Then log a correction
        correction_id = collector.log_correction(
            extraction_id=extraction_id,
            original_relation="knows",
            corrected_relation="colleague_of",
            correction_source="user",
            reasoning="They work together"
        )
        
        assert correction_id is not None
    
    def test_get_training_dataset(self, collector):
        """Test getting training dataset."""
        # Log some extractions
        collector.log_extraction(
            subject="Sarah",
            object_text="Google",
            context="Sarah works at Google.",
            predicted_relation="works_at",
            confidence=0.85,
            category="professional",
            source="deberta"
        )
        
        collector.log_extraction(
            subject="John",
            object_text="me",
            context="John is my brother.",
            predicted_relation="brother_of",
            confidence=0.75,
            category="family",
            source="llm"
        )
        
        # Get dataset
        dataset = collector.get_training_dataset(min_confidence=0.7)
        
        assert len(dataset) == 2
        assert all("subject" in d for d in dataset)
        assert all("object" in d for d in dataset)
        assert all("relation" in d for d in dataset)
    
    def test_get_stats(self, collector):
        """Test getting collection statistics."""
        # Log some data
        collector.log_extraction(
            subject="Sarah",
            object_text="Google",
            context="Sarah works at Google.",
            predicted_relation="works_at",
            confidence=0.85,
            category="professional",
            source="deberta"
        )
        
        collector.log_discovered_relation(
            relation_name="mentors",
            category="professional"
        )
        
        stats = collector.get_stats()
        
        assert stats["total_extractions"] == 1
        assert "deberta" in stats["by_source"]
        assert stats["discovered_relations"] == 1
    
    def test_export_to_huggingface_format(self, collector):
        """Test exporting to HuggingFace format."""
        # Log an extraction
        collector.log_extraction(
            subject="Sarah",
            object_text="Google",
            context="Sarah works at Google.",
            predicted_relation="works_at",
            confidence=0.85,
            category="professional",
            source="deberta"
        )
        
        # Export
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            collector.export_to_huggingface_format(output_path, min_confidence=0.7)
            
            # Verify file was created
            assert os.path.exists(output_path)
            
            # Verify content
            import json
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                data = json.loads(lines[0])
                assert data["subject"] == "Sarah"
                assert data["relation"] == "works_at"
        finally:
            os.unlink(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

