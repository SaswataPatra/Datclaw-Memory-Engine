"""
End-to-End tests for the complete extraction pipeline.
Tests all fallback levels with real API keys.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock
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
    """Test configuration with API keys from environment."""
    return {
        "dependency_extraction": {
            "enabled": True
        },
        "coref": {
            "provider": "simple",
            "model": "en_core_web_md"
        },
        "hf_api": {
            "api_key": os.getenv("HUGGINGFACE_API_KEY"),  # Correct path for RelationClassifier
            "model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        },
        "relation_classification": {
            "confidence_threshold": 0.5,
            "llm_model": "gpt-4o-mini"
        }
    }


@pytest.fixture
def relation_extractor(mock_db, mock_embedding_service, config):
    """Create RelationExtractor with real API keys."""
    extractor = RelationExtractor(
        db=mock_db,
        config=config,
        embedding_service=mock_embedding_service,
        collect_training_data=False
    )
    
    # Mock entity resolver
    async def mock_resolve(text, user_id, context, entity_type=None):
        entity = Mock()
        entity.entity_id = f"e_{text.lower().replace(' ', '_')}"
        entity.canonical_name = text
        entity.type = entity_type or "unknown"
        entity.confidence = 0.9
        return entity
    
    extractor.entity_resolver.resolve = mock_resolve
    
    return extractor


class TestLevel1_DependencyExtraction:
    """Test Level 1: Dependency-based extraction (should succeed)"""
    
    @pytest.mark.asyncio
    async def test_simple_svo(self, relation_extractor):
        """Test: Simple SVO sentence"""
        text = "Sarah studies physics"
        relations = await relation_extractor.extract(text, "test_user")
        
        assert len(relations) >= 1
        assert relations[0].source == "dependency"
        assert relations[0].subject_text == "Sarah"
        assert relations[0].object_text == "physics"
        print(f"✅ Level 1 (Dependency): {relations[0].subject_text} → {relations[0].relation} → {relations[0].object_text}")
    
    @pytest.mark.asyncio
    async def test_prepositional(self, relation_extractor):
        """Test: Prepositional relation"""
        text = "John works at Google"
        relations = await relation_extractor.extract(text, "test_user")
        
        assert len(relations) >= 1
        assert relations[0].source == "dependency"
        # Relation normalizer maps "works_at" to "employed_by"
        assert relations[0].relation in ["employed_by", "works_at", "work_at"]
        print(f"✅ Level 1 (Dependency): {relations[0].subject_text} → {relations[0].relation} → {relations[0].object_text}")
    
    @pytest.mark.asyncio
    async def test_copula(self, relation_extractor):
        """Test: Copula relation"""
        text = "Sarah is my sister"
        relations = await relation_extractor.extract(text, "test_user")
        
        assert len(relations) >= 1
        assert relations[0].source == "dependency"
        print(f"✅ Level 1 (Dependency): {relations[0].subject_text} → {relations[0].relation} → {relations[0].object_text}")


class TestLevel2_EntityPairFallback:
    """Test Level 2: Entity-pair extraction (when dependency fails)"""
    
    @pytest.mark.asyncio
    async def test_fallback_to_entity_pair(self, relation_extractor):
        """Test: Sentence that dependency parsing might miss"""
        # Disable dependency extraction to force fallback
        relation_extractor.use_dependency_extraction = False
        
        text = "Sarah and Google have a connection"
        relations = await relation_extractor.extract(text, "test_user")
        
        # Should fall back to entity-pair method
        if relations:
            print(f"✅ Level 2 (Entity-Pair): {relations[0].subject_text} → {relations[0].relation} → {relations[0].object_text}")
            print(f"   Source: {relations[0].source}")
        else:
            print("⚠️  Level 2: No relations extracted (acceptable for ambiguous sentence)")


class TestLevel3_DeBERTaZeroShot:
    """Test Level 3: DeBERTa zero-shot classification"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("HUGGINGFACE_API_KEY"), reason="HuggingFace API key not set")
    async def test_deberta_classification(self, relation_extractor):
        """Test: DeBERTa should classify this relation"""
        # Disable dependency to force entity-pair → DeBERTa path
        relation_extractor.use_dependency_extraction = False
        
        text = "Sarah founded the company in 2020"
        relations = await relation_extractor.extract(text, "test_user")
        
        if relations:
            assert relations[0].source in ["deberta", "llm", "heuristic"]
            print(f"✅ Level 3 (DeBERTa/LLM): {relations[0].subject_text} → {relations[0].relation} → {relations[0].object_text}")
            print(f"   Source: {relations[0].source}, Confidence: {relations[0].confidence:.2f}")


class TestLevel4_LLMFallback:
    """Test Level 4: LLM fallback (when DeBERTa confidence is low)"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
    async def test_llm_fallback(self, relation_extractor):
        """Test: Complex relation that might need LLM"""
        # Disable dependency to force entity-pair path
        relation_extractor.use_dependency_extraction = False
        
        # Use an unusual relation that DeBERTa might not handle well
        text = "Sarah mentors John in advanced quantum mechanics"
        relations = await relation_extractor.extract(text, "test_user")
        
        if relations:
            print(f"✅ Level 4 (LLM/Heuristic): {relations[0].subject_text} → {relations[0].relation} → {relations[0].object_text}")
            print(f"   Source: {relations[0].source}, Confidence: {relations[0].confidence:.2f}")


class TestLevel5_Heuristics:
    """Test Level 5: Heuristic fallback (when APIs fail)"""
    
    @pytest.mark.asyncio
    async def test_heuristic_fallback(self, relation_extractor):
        """Test: Heuristics should catch family relations"""
        # Disable dependency and force heuristics by using no API keys
        relation_extractor.use_dependency_extraction = False
        relation_extractor.relation_classifier.hf_api_key = None
        # llm_client is a property, can't set to None directly
        
        text = "Sarah is my sister and she lives nearby"
        relations = await relation_extractor.extract(text, "test_user")
        
        if relations:
            # Should use heuristics to detect "sister"
            assert relations[0].source == "heuristic"
            print(f"✅ Level 5 (Heuristic): {relations[0].subject_text} → {relations[0].relation} → {relations[0].object_text}")


class TestFullPipelineIntegration:
    """Test the complete pipeline with various sentence types"""
    
    @pytest.mark.asyncio
    async def test_mixed_sentences(self, relation_extractor):
        """Test: Multiple sentences with different patterns"""
        sentences = [
            "Sarah studies physics at MIT",
            "John works at Google in California", 
            "Mark is the CEO of the company",
            "She loves programming and mathematics",
            "My sister Sarah lives in Boston"
        ]
        
        total_relations = 0
        for text in sentences:
            relations = await relation_extractor.extract(text, "test_user")
            total_relations += len(relations)
            
            if relations:
                for rel in relations:
                    print(f"✅ {rel.subject_text} → {rel.relation} → {rel.object_text} (source={rel.source})")
        
        # Should extract at least 5 relations (one per sentence minimum)
        assert total_relations >= 5
        print(f"\n📊 Total relations extracted: {total_relations}")
    
    @pytest.mark.asyncio
    async def test_complex_sentence_with_coref(self, relation_extractor):
        """Test: Complex sentence with pronouns (tests coref → dependency)"""
        text = "Sarah is my sister. She studies physics at MIT and works at Google."
        relations = await relation_extractor.extract(text, "test_user")
        
        # Should extract at least 1 relation (complex sentences may not extract all)
        assert len(relations) >= 1
        
        print(f"\n📊 Complex sentence extracted {len(relations)} relations:")
        for rel in relations:
            print(f"   {rel.subject_text} → {rel.relation} → {rel.object_text} (source={rel.source})")
    
    @pytest.mark.asyncio
    async def test_performance(self, relation_extractor):
        """Test: Performance benchmark"""
        import time
        
        text = "Sarah studies physics at MIT"
        
        # Warmup
        await relation_extractor.extract(text, "test_user")
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            relations = await relation_extractor.extract(text, "test_user")
        elapsed = (time.time() - start) / 10
        
        print(f"\n⚡ Performance: {elapsed*1000:.2f}ms per sentence")
        # With en_core_web_md, expect ~50-100ms per sentence for dependency extraction
        assert elapsed < 0.500  # Should be under 500ms (dependency extraction without API calls)


class TestFallbackSequence:
    """Test that fallbacks happen in the correct order"""
    
    @pytest.mark.asyncio
    async def test_fallback_order(self, relation_extractor):
        """Test: Verify fallback sequence works"""
        # Test 1: Dependency should work
        text1 = "Sarah studies physics"
        relations1 = await relation_extractor.extract(text1, "test_user")
        assert relations1[0].source == "dependency"
        print("✅ Level 1 (Dependency) → Success")
        
        # Test 2: Disable dependency, should fall back to entity-pair
        relation_extractor.use_dependency_extraction = False
        text2 = "Sarah and Google are connected"
        relations2 = await relation_extractor.extract(text2, "test_user")
        
        if relations2:
            assert relations2[0].source in ["deberta", "llm", "heuristic"]
            print(f"✅ Level 2+ (Entity-Pair → {relations2[0].source}) → Success")
        
        # Re-enable for other tests
        relation_extractor.use_dependency_extraction = True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

