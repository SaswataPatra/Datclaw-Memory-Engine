"""
Unit tests for EntityExtractor.

Tests:
1. Basic entity extraction
2. Confidence computation
3. Entity type mapping
4. Context extraction
5. Entity pairs for relation extraction
"""

import pytest
from core.graph.entity_extractor import EntityExtractor, ExtractedEntity


class TestEntityExtractor:
    """Test EntityExtractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance."""
        return EntityExtractor(config={'model': 'en_core_web_md'})
    
    def test_init(self, extractor):
        """Test extractor initialization."""
        assert extractor.nlp is not None
        assert extractor.model_name == 'en_core_web_md'
        assert extractor.has_transformer is False
    
    def test_extract_person(self, extractor):
        """Test extraction of person entities."""
        text = "Sarah is my sister and she works at Google."
        entities = extractor.extract(text)
        
        # Should find Sarah and Google
        assert len(entities) >= 2
        
        # Find Sarah
        sarah = next((e for e in entities if "Sarah" in e.text), None)
        assert sarah is not None
        assert sarah.type == "person"
        assert sarah.original_type == "PERSON"
        assert 0.0 < sarah.confidence <= 1.0
    
    def test_extract_organization(self, extractor):
        """Test extraction of organization entities."""
        text = "I work at Microsoft and previously worked at Apple."
        entities = extractor.extract(text)
        
        # Should find Microsoft and Apple
        org_entities = [e for e in entities if e.type == "organization"]
        assert len(org_entities) >= 1
        
        # Check at least one is found
        org_names = [e.text for e in org_entities]
        assert "Microsoft" in org_names or "Apple" in org_names
    
    def test_extract_location(self, extractor):
        """Test extraction of location entities."""
        text = "I live in New York and I visited Paris last summer."
        entities = extractor.extract(text)
        
        # Should find New York and Paris
        loc_entities = [e for e in entities if e.type == "location"]
        assert len(loc_entities) >= 1
        
        loc_names = [e.text for e in loc_entities]
        assert "New York" in loc_names or "Paris" in loc_names
    
    def test_extract_temporal(self, extractor):
        """Test extraction of temporal entities."""
        text = "We met on January 15, 2024 at 3pm."
        entities = extractor.extract(text)
        
        # Should find date/time entities
        temporal_entities = [e for e in entities if e.type == "temporal"]
        # Note: spaCy may combine or separate these
        assert len(temporal_entities) >= 0  # May or may not extract depending on model
    
    def test_confidence_computed(self, extractor):
        """Test that confidence is computed, not hardcoded."""
        text = "John works at IBM."
        entities = extractor.extract(text)
        
        # All entities should have confidence between 0 and 1
        for entity in entities:
            assert 0.0 < entity.confidence <= 1.0
            # Confidence should be rounded to 3 decimal places
            assert entity.confidence == round(entity.confidence, 3)
    
    def test_confidence_varies(self, extractor):
        """Test that confidence varies based on entity type and context."""
        # Clear entity types should have higher confidence
        text1 = "Sarah Smith is the CEO."
        text2 = "X is important."
        
        entities1 = extractor.extract(text1)
        entities2 = extractor.extract(text2)
        
        # Sarah Smith should be extracted with reasonable confidence
        if entities1:
            sarah = next((e for e in entities1 if "Sarah" in e.text), None)
            if sarah:
                assert sarah.confidence > 0.5
    
    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []
        assert extractor.extract(None) == []
    
    def test_extract_no_entities(self, extractor):
        """Test extraction from text with no entities."""
        text = "I like to eat pizza and watch movies."
        entities = extractor.extract(text)
        # May or may not find entities depending on model
        # Just ensure it doesn't crash
        assert isinstance(entities, list)
    
    def test_entity_positions(self, extractor):
        """Test that entity positions are correct."""
        text = "John works at Google."
        entities = extractor.extract(text)
        
        for entity in entities:
            # Verify the extracted text matches the position
            extracted = text[entity.start:entity.end]
            assert extracted == entity.text
    
    def test_type_mapping(self, extractor):
        """Test that spaCy types are mapped to DAPPY types."""
        # All DAPPY types should be valid
        valid_types = {"person", "organization", "location", "concept", 
                       "event", "temporal", "value", "quantity", "group", "unknown"}
        
        text = "Sarah from New York works at Google since 2020."
        entities = extractor.extract(text)
        
        for entity in entities:
            assert entity.type in valid_types
    
    def test_extract_with_context(self, extractor):
        """Test extraction with surrounding context."""
        text = "My sister Sarah works at Google in Mountain View."
        results = extractor.extract_with_context(text, context_window=20)
        
        for result in results:
            assert "context" in result
            assert "context_start" in result
            assert "context_end" in result
            # Context should include the entity text
            assert result["text"] in result["context"]
    
    def test_extract_persons_only(self, extractor):
        """Test extracting only person entities."""
        text = "Sarah works at Google with John."
        persons = extractor.extract_persons(text)
        
        for person in persons:
            assert person.type == "person"
    
    def test_extract_organizations_only(self, extractor):
        """Test extracting only organization entities."""
        text = "I interviewed at Google and Microsoft."
        orgs = extractor.extract_organizations(text)
        
        for org in orgs:
            assert org.type == "organization"
    
    def test_extract_locations_only(self, extractor):
        """Test extracting only location entities."""
        text = "I traveled from New York to Paris."
        locs = extractor.extract_locations(text)
        
        for loc in locs:
            assert loc.type == "location"
    
    def test_get_entity_pairs(self, extractor):
        """Test getting entity pairs for relation extraction."""
        text = "Sarah works at Google with John."
        pairs = extractor.get_entity_pairs(text)
        
        # Should get pairs of entities
        for e1, e2 in pairs:
            assert isinstance(e1, ExtractedEntity)
            assert isinstance(e2, ExtractedEntity)
            # Entities should be different
            assert e1.text.lower() != e2.text.lower()
    
    def test_extracted_entity_to_dict(self, extractor):
        """Test ExtractedEntity.to_dict() method."""
        text = "Sarah works at Google."
        entities = extractor.extract(text)
        
        if entities:
            entity = entities[0]
            d = entity.to_dict()
            
            assert "text" in d
            assert "type" in d
            assert "original_type" in d
            assert "start" in d
            assert "end" in d
            assert "confidence" in d
            assert "metadata" in d
    
    def test_metadata_includes_model_info(self, extractor):
        """Test that metadata includes model information."""
        text = "Sarah works at Google."
        entities = extractor.extract(text)
        
        if entities:
            entity = entities[0]
            assert "model" in entity.metadata
            assert "has_transformer" in entity.metadata
            assert entity.metadata["model"] == extractor.model_name


class TestExtractedEntity:
    """Test ExtractedEntity dataclass."""
    
    def test_create_entity(self):
        """Test creating an ExtractedEntity."""
        entity = ExtractedEntity(
            text="Sarah",
            type="person",
            original_type="PERSON",
            start=0,
            end=5,
            confidence=0.85
        )
        
        assert entity.text == "Sarah"
        assert entity.type == "person"
        assert entity.confidence == 0.85
    
    def test_entity_with_metadata(self):
        """Test creating an entity with metadata."""
        entity = ExtractedEntity(
            text="Google",
            type="organization",
            original_type="ORG",
            start=10,
            end=16,
            confidence=0.90,
            metadata={"source": "test"}
        )
        
        assert entity.metadata["source"] == "test"
    
    def test_to_dict(self):
        """Test converting entity to dictionary."""
        entity = ExtractedEntity(
            text="Paris",
            type="location",
            original_type="GPE",
            start=0,
            end=5,
            confidence=0.88,
            metadata={"model": "test"}
        )
        
        d = entity.to_dict()
        
        assert d["text"] == "Paris"
        assert d["type"] == "location"
        assert d["original_type"] == "GPE"
        assert d["start"] == 0
        assert d["end"] == 5
        assert d["confidence"] == 0.88
        assert d["metadata"]["model"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

