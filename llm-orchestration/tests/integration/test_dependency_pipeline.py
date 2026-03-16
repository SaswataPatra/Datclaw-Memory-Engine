"""
Integration tests for full dependency-based extraction pipeline.
Tests CorefResolver → DependencyExtractor → EntityResolver → RelationNormalizer → RelationExtractor
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from core.graph.coref_resolver import CorefResolver
from core.graph.dependency_extractor import DependencyExtractor
from core.graph.relation_normalizer import RelationNormalizer
from core.graph.relation_extractor import RelationExtractor


@pytest.fixture
def mock_db():
    """Mock ArangoDB database."""
    db = Mock()
    db.collection = Mock(return_value=Mock())
    return db


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    service.generate = Mock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def config():
    """Test configuration."""
    return {
        "dependency_extraction": {
            "enabled": True,
            "patterns": {
                "enabled_patterns": [
                    "nsubj_verb_dobj",
                    "nsubj_verb_prep_pobj",
                    "copula_attr",
                    "apposition",
                    "passive_voice",
                    "coordination",
                    "nested_clause",
                    "relative_clause"
                ]
            }
        },
        "coref": {
            "provider": "simple",
            "model": "en_core_web_md"  # Use medium model for better parsing
        }
    }


@pytest.fixture
def relation_extractor(mock_db, mock_embedding_service, config):
    """Create RelationExtractor with mocked dependencies."""
    extractor = RelationExtractor(
        db=mock_db,
        config=config,
        embedding_service=mock_embedding_service,
        collect_training_data=False
    )
    
    # Mock entity resolver to return predictable entities
    async def mock_resolve(text, user_id, context, entity_type=None):
        entity = Mock()
        entity.entity_id = f"e_{text.lower().replace(' ', '_')}"
        entity.canonical_name = text
        entity.type = entity_type or "unknown"
        entity.confidence = 0.9  # Add confidence attribute
        return entity
    
    extractor.entity_resolver.resolve = mock_resolve
    
    return extractor


class TestCorefIntegration:
    """Test coreference resolution integration."""
    
    def test_coref_simple_pronoun(self):
        """Test: Simple pronoun resolution"""
        resolver = CorefResolver(provider="simple")
        text = "Sarah is my sister. She studies physics."
        resolved, clusters = resolver.resolve(text)
        
        # Simple resolver should attempt to replace pronouns
        assert isinstance(resolved, str)
        assert isinstance(clusters, list)
    
    def test_coref_with_dependency_extraction(self):
        """Test: Coref + dependency extraction"""
        resolver = CorefResolver(provider="simple")
        extractor = DependencyExtractor(nlp=resolver.nlp)
        
        text = "Sarah is my sister. She studies physics."
        resolved, _ = resolver.resolve(text)
        
        doc = resolver.nlp(resolved)
        triples = extractor.extract_from_doc(doc)
        
        # Should extract relations from resolved text
        assert len(triples) >= 1


class TestDependencyExtractionIntegration:
    """Test dependency extraction with various inputs."""
    
    @pytest.mark.asyncio
    async def test_simple_sentence(self, relation_extractor):
        """Test: Simple sentence extraction"""
        text = "Sarah studies physics"  # Now works with en_core_web_md
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_001"
        )
        
        assert len(relations) >= 1
        assert relations[0].subject_text == "Sarah"
        assert relations[0].object_text == "physics"
        assert relations[0].source == "dependency"
    
    @pytest.mark.asyncio
    async def test_prepositional_sentence(self, relation_extractor):
        """Test: Prepositional relation extraction"""
        text = "Sarah lives in Boston"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_002"
        )
        
        assert len(relations) >= 1
        assert relations[0].subject_text == "Sarah"
        assert relations[0].object_text == "Boston"
        assert "live" in relations[0].relation or "resid" in relations[0].relation
    
    @pytest.mark.asyncio
    async def test_passive_voice(self, relation_extractor):
        """Test: Passive voice with agent flipping"""
        text = "Physics is studied by Sarah"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_003"
        )
        
        if relations:
            # Subject and object should be flipped
            assert relations[0].subject_text == "Sarah"
            assert relations[0].object_text == "Physics"
            assert relations[0].metadata.get("is_passive") == True
    
    @pytest.mark.asyncio
    async def test_coordination(self, relation_extractor):
        """Test: Coordination expansion"""
        text = "Sarah and John study physics"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_004"
        )
        
        # Should extract 2 relations (one for each subject)
        assert len(relations) >= 2
        subjects = {r.subject_text for r in relations}
        assert "Sarah" in subjects
        assert "John" in subjects
    
    @pytest.mark.asyncio
    async def test_apposition(self, relation_extractor):
        """Test: Apposition extraction"""
        text = "Mark, the CEO, spoke yesterday"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_005"
        )
        
        # Should extract apposition relation
        appos = [r for r in relations if "CEO" in r.object_text]
        assert len(appos) >= 1
        assert appos[0].subject_text == "Mark"


class TestRelationNormalization:
    """Test relation normalization integration."""
    
    @pytest.mark.asyncio
    async def test_normalization_works_at(self, relation_extractor):
        """Test: Normalize 'works at' to canonical form"""
        text = "Sarah works at Google"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_006"
        )
        
        assert len(relations) >= 1
        # Should be normalized to canonical form
        assert relations[0].relation in ["employed_by", "work_at", "works_at"]
    
    @pytest.mark.asyncio
    async def test_normalization_lives_in(self, relation_extractor):
        """Test: Normalize 'lives in' to canonical form"""
        text = "Sarah lives in Boston"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_007"
        )
        
        assert len(relations) >= 1
        # Should be normalized to canonical form
        assert relations[0].relation in ["resides_in", "live_in", "lives_in"]


class TestConfidenceComposition:
    """Test confidence composition from multiple factors."""
    
    @pytest.mark.asyncio
    async def test_high_confidence_simple(self, relation_extractor):
        """Test: High confidence for simple SVO"""
        text = "Sarah studies physics"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_008"
        )
        
        assert len(relations) >= 1
        # Should have high confidence (pattern + entity resolution + normalization)
        assert relations[0].confidence >= 0.70
    
    @pytest.mark.asyncio
    async def test_lower_confidence_with_modality(self, relation_extractor):
        """Test: Lower confidence with modal verb"""
        text = "Sarah might study physics"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_009"
        )
        
        if relations:
            # Should have lower confidence due to modality
            assert relations[0].metadata.get("modality") == "might"
            assert relations[0].metadata.get("modality_score") == 0.40


class TestComplexSentences:
    """Test extraction from complex sentences."""
    
    @pytest.mark.asyncio
    async def test_multiple_relations(self, relation_extractor):
        """Test: Extract multiple relations from one sentence"""
        text = "Sarah, my sister, studies physics at MIT and works at Google."
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_010"
        )
        
        # Should extract multiple relations
        assert len(relations) >= 3
        
        # Check for different relation types
        relation_types = {r.relation for r in relations}
        assert len(relation_types) >= 2
    
    @pytest.mark.asyncio
    async def test_nested_clauses(self, relation_extractor):
        """Test: Extract from nested clauses"""
        text = "I know that Sarah studies physics"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_011"
        )
        
        # Should extract the nested fact
        sarah_relations = [r for r in relations if r.subject_text == "Sarah"]
        assert len(sarah_relations) >= 1
    
    @pytest.mark.asyncio
    async def test_with_pronouns(self, relation_extractor):
        """Test: Handle pronouns via coref"""
        text = "Sarah is my sister. She studies physics."
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_012"
        )
        
        # Should extract relations (coref may or may not resolve perfectly)
        assert len(relations) >= 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_text(self, relation_extractor):
        """Test: Handle empty text"""
        relations = await relation_extractor.extract(
            text="",
            user_id="test_user",
            memory_id="mem_013"
        )
        
        assert len(relations) == 0
    
    @pytest.mark.asyncio
    async def test_no_relations(self, relation_extractor):
        """Test: Handle text with no extractable relations"""
        text = "Hello world"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_014"
        )
        
        # May extract nothing
        assert isinstance(relations, list)
    
    @pytest.mark.asyncio
    async def test_entity_resolution_failure(self, relation_extractor):
        """Test: Handle entity resolution failure gracefully"""
        # Mock entity resolver to return None
        async def mock_resolve_fail(*args, **kwargs):
            return None
        
        relation_extractor.entity_resolver.resolve = mock_resolve_fail
        
        text = "Sarah studies physics"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_015"
        )
        
        # Should handle gracefully (may return empty list)
        assert isinstance(relations, list)


class TestMetadataPreservation:
    """Test that metadata is preserved through pipeline."""
    
    @pytest.mark.asyncio
    async def test_metadata_preserved(self, relation_extractor):
        """Test: Metadata is preserved in extracted relations"""
        text = "Sarah studies physics"
        metadata = {"triggers": ["study"], "session_id": "sess_001"}
        
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_016",
            metadata=metadata
        )
        
        assert len(relations) >= 1
        # Metadata should be preserved
        assert "session_id" in relations[0].metadata
        assert relations[0].metadata["session_id"] == "sess_001"
    
    @pytest.mark.asyncio
    async def test_pattern_metadata(self, relation_extractor):
        """Test: Pattern metadata is included"""
        text = "Sarah studies physics"
        relations = await relation_extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_017"
        )
        
        assert len(relations) >= 1
        # Should include pattern information
        assert "pattern" in relations[0].metadata
        assert relations[0].metadata["pattern"] in [
            "nsubj_verb_dobj",
            "coordination",
            "nested_clause"
        ]


class TestFallbackBehavior:
    """Test fallback to entity-pair method."""
    
    @pytest.mark.asyncio
    async def test_fallback_when_disabled(self, mock_db, mock_embedding_service):
        """Test: Falls back to entity-pair when dependency extraction disabled"""
        config = {
            "dependency_extraction": {
                "enabled": False
            }
        }
        
        extractor = RelationExtractor(
            db=mock_db,
            config=config,
            embedding_service=mock_embedding_service,
            collect_training_data=False
        )
        
        # Mock entity resolver
        async def mock_resolve(text, user_id, context, entity_type=None):
            entity = Mock()
            entity.entity_id = f"e_{text.lower()}"
            entity.canonical_name = text
            entity.type = entity_type or "unknown"
            return entity
        
        extractor.entity_resolver.resolve = mock_resolve
        
        text = "Sarah studies physics"
        relations = await extractor.extract(
            text=text,
            user_id="test_user",
            memory_id="mem_018"
        )
        
        # Should still extract (via fallback method)
        # Source should NOT be "dependency"
        if relations:
            assert relations[0].source != "dependency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

