"""
Tests for Graph-of-Thoughts schemas.
Phase 1 implementation verification.
"""

import pytest
from datetime import datetime, timedelta

from core.graph.schemas import (
    ThoughtEdge,
    CandidateEdge,
    Entity,
    SupportingMention,
    EdgeType,
    RelationCategory,
    EDGE_TYPE_TAXONOMY,
    HIGH_IMPACT_RELATIONS,
    get_relation_category,
)


class TestEdgeTypes:
    """Test edge type taxonomy and categorization."""
    
    def test_edge_type_taxonomy_has_all_categories(self):
        """All relation categories should be in taxonomy."""
        for category in RelationCategory:
            assert category in EDGE_TYPE_TAXONOMY
    
    def test_get_relation_category_family(self):
        """Family relations should map to family category."""
        assert get_relation_category("sister_of") == RelationCategory.FAMILY
        assert get_relation_category("brother_of") == RelationCategory.FAMILY
        assert get_relation_category("parent_of") == RelationCategory.FAMILY
    
    def test_get_relation_category_professional(self):
        """Professional relations should map to professional category."""
        assert get_relation_category("works_at") == RelationCategory.PROFESSIONAL
        assert get_relation_category("colleague_of") == RelationCategory.PROFESSIONAL
    
    def test_get_relation_category_temporal(self):
        """Temporal relations should map to temporal category."""
        assert get_relation_category("evolves_to") == RelationCategory.TEMPORAL
        assert get_relation_category("supersedes") == RelationCategory.TEMPORAL
        assert get_relation_category("contradicts") == RelationCategory.TEMPORAL
    
    def test_get_relation_category_unknown(self):
        """Unknown relations should return None."""
        assert get_relation_category("unknown_relation") is None
    
    def test_high_impact_relations(self):
        """High impact relations should be defined."""
        assert EdgeType.SISTER_OF in HIGH_IMPACT_RELATIONS
        assert EdgeType.WORKS_AT in HIGH_IMPACT_RELATIONS
        assert EdgeType.SPOUSE_OF in HIGH_IMPACT_RELATIONS


class TestThoughtEdge:
    """Test ThoughtEdge schema."""
    
    def test_create_thought_edge(self):
        """Create a basic ThoughtEdge."""
        edge = ThoughtEdge(
            _from="memories/m_123",
            _to="memories/m_456",
            relation="contradicts",
            relation_category="temporal",
            strength=0.75
        )
        
        assert edge.id.startswith("e_")
        assert edge._from == "memories/m_123"
        assert edge._to == "memories/m_456"
        assert edge.relation == "contradicts"
        assert edge.relation_category == "temporal"
        assert edge.strength == 0.75
        assert edge.is_active == True
        assert edge.is_bidirectional == False
    
    def test_thought_edge_bidirectional(self):
        """Test bidirectional edge creation (Loophole #4 fix)."""
        edge = ThoughtEdge(
            _from="memories/m_123",
            _to="memories/m_456",
            relation="contradicts",
            is_bidirectional=True,
            pair_edge_id="e_other"
        )
        
        assert edge.is_bidirectional == True
        assert edge.pair_edge_id == "e_other"
    
    def test_thought_edge_temporal_validity(self):
        """Test temporal validity fields (Loophole #6 fix)."""
        now = datetime.utcnow()
        future = now + timedelta(days=30)
        
        edge = ThoughtEdge(
            _from="memories/m_123",
            _to="memories/m_456",
            relation="works_at",
            effective_from=now,
            effective_to=future
        )
        
        assert edge.effective_from == now
        assert edge.effective_to == future
    
    def test_thought_edge_to_arango_doc(self):
        """Test conversion to ArangoDB document."""
        edge = ThoughtEdge(
            _from="memories/m_123",
            _to="memories/m_456",
            relation="supersedes",
            relation_category="temporal",
            strength=0.8,
            tier=1
        )
        
        doc = edge.to_arango_doc()
        
        assert doc["_key"] == edge.id
        assert doc["_from"] == "memories/m_123"
        assert doc["_to"] == "memories/m_456"
        assert doc["relation"] == "supersedes"
        assert doc["relation_category"] == "temporal"
        assert doc["strength"] == 0.8
        assert doc["tier"] == 1
    
    def test_thought_edge_from_arango_doc(self):
        """Test creation from ArangoDB document."""
        doc = {
            "_key": "e_test123",
            "_from": "memories/m_100",
            "_to": "memories/m_200",
            "relation": "refines",
            "relation_category": "temporal",
            "strength": 0.65,
            "is_active": True,
            "is_bidirectional": False,
            "tier": 2,
            "created_at": "2025-11-27T10:00:00"
        }
        
        edge = ThoughtEdge.from_arango_doc(doc)
        
        assert edge.id == "e_test123"
        assert edge._from == "memories/m_100"
        assert edge.relation == "refines"
        assert edge.strength == 0.65


class TestCandidateEdge:
    """Test CandidateEdge schema."""
    
    def test_create_candidate_edge(self):
        """Create a basic CandidateEdge."""
        mention = SupportingMention(
            mem_id="m_123",
            srl_conf=0.85,
            coref_conf=0.90,
            ego=0.62
        )
        
        candidate = CandidateEdge(
            user_id="u1",
            subject_entity_id="e_456",
            predicate="ASKED_ABOUT",
            object_entity_id="c_789",
            supporting_mentions=[mention]
        )
        
        assert candidate.candidate_id.startswith("ce_")
        assert candidate.user_id == "u1"
        assert candidate.subject_entity_id == "e_456"
        assert candidate.predicate == "ASKED_ABOUT"
        assert len(candidate.supporting_mentions) == 1
        assert candidate.status == "candidate"
    
    def test_candidate_edge_aggregation_key(self):
        """Test aggregation key generation (Loophole #2 fix)."""
        # With entity IDs
        candidate1 = CandidateEdge(
            subject_entity_id="e_456",
            predicate="WORKS_AT",
            object_entity_id="e_789"
        )
        
        key1 = candidate1.get_aggregation_key()
        assert key1 == ("e_456", "WORKS_AT", "e_789")
        
        # With span fallback
        candidate2 = CandidateEdge(
            subject_span={"text": "Sarah"},
            predicate="WORKS_AT",
            object_span={"text": "Acme Corp"}
        )
        
        key2 = candidate2.get_aggregation_key()
        assert key2 == ("Sarah", "WORKS_AT", "Acme Corp")
    
    def test_candidate_edge_to_arango_doc(self):
        """Test conversion to ArangoDB document."""
        mention = SupportingMention(
            mem_id="m_123",
            srl_conf=0.85,
            ego=0.62
        )
        
        candidate = CandidateEdge(
            user_id="u1",
            subject_entity_id="e_456",
            predicate="LIKES",
            object_span={"text": "cats"},
            supporting_mentions=[mention],
            edge_evidence_count=1
        )
        
        doc = candidate.to_arango_doc()
        
        assert doc["_key"] == candidate.candidate_id
        assert doc["user_id"] == "u1"
        assert doc["subject_entity_id"] == "e_456"
        assert doc["predicate"] == "LIKES"
        assert len(doc["supporting_mentions"]) == 1


class TestEntity:
    """Test Entity schema."""
    
    def test_create_entity(self):
        """Create a basic Entity."""
        entity = Entity(
            canonical_name="Sarah",
            type="person",
            aliases=["sarah", "Sarah Johnson"],
            user_id="u1"
        )
        
        assert entity.entity_id.startswith("e_")
        assert entity.canonical_name == "Sarah"
        assert entity.type == "person"
        assert "sarah" in entity.aliases
        assert entity.user_id == "u1"
    
    def test_entity_add_alias(self):
        """Test adding aliases."""
        entity = Entity(canonical_name="Sarah")
        
        entity.add_alias("sarah@acme.com")
        assert "sarah@acme.com" in entity.aliases
        
        # Duplicate should not be added
        entity.add_alias("SARAH@ACME.COM")  # Case-insensitive
        assert len([a for a in entity.aliases if "acme" in a.lower()]) == 1
    
    def test_entity_link_memory(self):
        """Test linking memories."""
        entity = Entity(canonical_name="Sarah")
        
        entity.link_memory("m_123")
        entity.link_memory("m_456")
        
        assert "m_123" in entity.linked_memories
        assert "m_456" in entity.linked_memories
        assert entity.stats["mention_frequency"] == 2
        
        # Duplicate should not be added
        entity.link_memory("m_123")
        assert len(entity.linked_memories) == 2
    
    def test_entity_to_arango_doc(self):
        """Test conversion to ArangoDB document."""
        entity = Entity(
            canonical_name="Acme Corp",
            type="organization",
            aliases=["acme", "Acme Corporation"],
            user_id="u1"
        )
        
        doc = entity.to_arango_doc()
        
        assert doc["_key"] == entity.entity_id
        assert doc["canonical_name"] == "Acme Corp"
        assert doc["type"] == "organization"
        assert "acme" in doc["aliases"]


class TestSupportingMention:
    """Test SupportingMention schema."""
    
    def test_create_supporting_mention(self):
        """Create a SupportingMention."""
        mention = SupportingMention(
            mem_id="m_123",
            srl_conf=0.88,
            coref_conf=0.90,
            ego=0.62
        )
        
        assert mention.mem_id == "m_123"
        assert mention.srl_conf == 0.88
        assert mention.coref_conf == 0.90
        assert mention.ego == 0.62
        assert mention.observed_at is not None

