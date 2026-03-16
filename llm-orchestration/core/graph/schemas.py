"""
DAPPY Graph-of-Thoughts Schemas
Version: 3.0 (Implicit Clustering via Edge Types)

Defines:
- ThoughtEdge: Canonical KG edge with relation_category
- CandidateEdge: Tier 4 candidate edges for activation scoring
- Entity: Simplified entity (no cluster membership)
- Edge Type Taxonomy: Config-based implicit clustering
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid


class RelationCategory(str, Enum):
    """
    Edge type categories for implicit clustering.
    Context is derived from edge types, not explicit cluster nodes.
    """
    FAMILY = "family"
    PROFESSIONAL = "professional"
    PERSONAL = "personal"
    TEMPORAL = "temporal"
    FACTUAL = "factual"


class EdgeType(str, Enum):
    """
    GoT Edge Types with semantic meaning.
    See Section 3.3 of DAPPY_UNIFIED_ARCHITECTURE.md
    """
    # Temporal (thought evolution)
    EVOLVES_TO = "evolves_to"      # Natural succession
    SUPERSEDES = "supersedes"       # B replaces A
    REFINES = "refines"             # B adds detail to A
    CLARIFIES = "clarifies"         # B explains A
    CONTRADICTS = "contradicts"     # Conflict (bidirectional)
    SUPPORTS = "supports"           # A is evidence for B
    
    # Family
    SISTER_OF = "sister_of"
    BROTHER_OF = "brother_of"
    DAUGHTER_OF = "daughter_of"
    SON_OF = "son_of"
    PARENT_OF = "parent_of"
    SPOUSE_OF = "spouse_of"
    SIBLING_OF = "sibling_of"
    
    # Professional
    WORKS_AT = "works_at"
    COLLEAGUE_OF = "colleague_of"
    REPORTS_TO = "reports_to"
    MANAGES = "manages"
    COLLABORATES_WITH = "collaborates_with"
    EMPLOYED_BY = "employed_by"
    
    # Personal
    LIKES = "likes"
    DISLIKES = "dislikes"
    PREFERS = "prefers"
    INTERESTED_IN = "interested_in"
    ENJOYS = "enjoys"
    AVOIDS = "avoids"
    
    # Factual
    LOCATED_AT = "located_at"
    OWNS = "owns"
    HAS_PROPERTY = "has_property"
    MEMBER_OF = "member_of"
    PART_OF = "part_of"


# Edge Type Taxonomy (Config-based implicit clustering)
# Maps relation types to categories for context-aware retrieval
EDGE_TYPE_TAXONOMY: Dict[RelationCategory, Dict[str, Any]] = {
    RelationCategory.FAMILY: {
        "relations": [
            EdgeType.SISTER_OF, EdgeType.BROTHER_OF, EdgeType.DAUGHTER_OF,
            EdgeType.SON_OF, EdgeType.PARENT_OF, EdgeType.SPOUSE_OF,
            EdgeType.SIBLING_OF
        ],
        "description": "Family relationships and kinship"
    },
    RelationCategory.PROFESSIONAL: {
        "relations": [
            EdgeType.WORKS_AT, EdgeType.COLLEAGUE_OF, EdgeType.REPORTS_TO,
            EdgeType.MANAGES, EdgeType.COLLABORATES_WITH, EdgeType.EMPLOYED_BY
        ],
        "description": "Work and professional relationships"
    },
    RelationCategory.PERSONAL: {
        "relations": [
            EdgeType.LIKES, EdgeType.DISLIKES, EdgeType.PREFERS,
            EdgeType.INTERESTED_IN, EdgeType.ENJOYS, EdgeType.AVOIDS
        ],
        "description": "Personal preferences and interests"
    },
    RelationCategory.TEMPORAL: {
        "relations": [
            EdgeType.EVOLVES_TO, EdgeType.SUPERSEDES, EdgeType.REFINES,
            EdgeType.CLARIFIES, EdgeType.CONTRADICTS, EdgeType.SUPPORTS
        ],
        "description": "Thought evolution and reasoning"
    },
    RelationCategory.FACTUAL: {
        "relations": [
            EdgeType.LOCATED_AT, EdgeType.OWNS, EdgeType.HAS_PROPERTY,
            EdgeType.MEMBER_OF, EdgeType.PART_OF
        ],
        "description": "Factual relationships and attributes"
    }
}


def get_relation_category(relation: str) -> Optional[RelationCategory]:
    """
    Get the category for a relation type.
    Used for context-aware retrieval (implicit clustering).
    """
    for category, config in EDGE_TYPE_TAXONOMY.items():
        relation_values = [r.value if isinstance(r, EdgeType) else r for r in config["relations"]]
        if relation in relation_values or relation == category.value:
            return category
    return None


# High-impact relations (for Shadow Tier routing)
HIGH_IMPACT_RELATIONS = [
    EdgeType.SISTER_OF, EdgeType.BROTHER_OF, EdgeType.PARENT_OF,
    EdgeType.SPOUSE_OF, EdgeType.WORKS_AT, EdgeType.EMPLOYED_BY,
    EdgeType.OWNS, EdgeType.LOCATED_AT
]


@dataclass
class ThoughtEdge:
    """
    Canonical KG Edge (promoted from candidate edges).
    See Section 4.2 of DAPPY_UNIFIED_ARCHITECTURE.md
    
    Key fields:
    - relation_category: For implicit clustering (family, professional, etc.)
    - effective_from/to: Temporal validity (Loophole #6 fix)
    - is_bidirectional: For contradiction edges (Loophole #4 fix)
    - strength: Unified decay formula (Loophole #3 fix)
    """
    id: str = field(default_factory=lambda: f"e_{uuid.uuid4().hex[:12]}")
    _from: str = ""  # memories/m_123
    _to: str = ""    # memories/m_456
    
    # Relation info
    relation: str = ""  # EdgeType value
    relation_category: str = ""  # RelationCategory value (for implicit clustering)
    
    # Strength & confidence
    strength: float = 0.5
    confidence: float = 0.5
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_by: str = "auto"  # "auto" | "user" | "pending"
    resolution_text: Optional[str] = None
    
    # Temporal gap (for contradiction detection)
    temporal_gap_days: int = 0
    
    # Provenance
    supporting_mentions: List[str] = field(default_factory=list)  # Memory IDs
    
    # Status
    is_active: bool = True
    
    # Temporal validity (Loophole #6 fix)
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    
    # Tier (inherited from highest-tier supporting memory)
    tier: int = 2
    
    # Provenance for activation scoring
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    # Versioning
    version: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Bidirectional edge support (Loophole #4 fix)
    is_bidirectional: bool = False
    pair_edge_id: Optional[str] = None
    
    # Decay tracking (Loophole #3 fix)
    last_fired_at: datetime = field(default_factory=datetime.utcnow)
    decay_lambda: float = 0.000012  # ~7 day half-life
    
    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB document format."""
        return {
            "_key": self.id,
            "_from": self._from,
            "_to": self._to,
            "relation": self.relation,
            "relation_category": self.relation_category,
            "strength": self.strength,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_by": self.resolved_by,
            "resolution_text": self.resolution_text,
            "temporal_gap_days": self.temporal_gap_days,
            "supporting_mentions": self.supporting_mentions,
            "is_active": self.is_active,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "tier": self.tier,
            "provenance": self.provenance,
            "version": self.version,
            "history": self.history,
            "is_bidirectional": self.is_bidirectional,
            "pair_edge_id": self.pair_edge_id,
            "last_fired_at": self.last_fired_at.isoformat() if self.last_fired_at else None,
            "decay_lambda": self.decay_lambda,
        }
    
    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> 'ThoughtEdge':
        """Create from ArangoDB document."""
        return cls(
            id=doc.get("_key", ""),
            _from=doc.get("_from", ""),
            _to=doc.get("_to", ""),
            relation=doc.get("relation", ""),
            relation_category=doc.get("relation_category", ""),
            strength=doc.get("strength", 0.5),
            confidence=doc.get("confidence", 0.5),
            created_at=datetime.fromisoformat(doc["created_at"]) if doc.get("created_at") else datetime.utcnow(),
            resolved_by=doc.get("resolved_by", "auto"),
            resolution_text=doc.get("resolution_text"),
            temporal_gap_days=doc.get("temporal_gap_days", 0),
            supporting_mentions=doc.get("supporting_mentions", []),
            is_active=doc.get("is_active", True),
            effective_from=datetime.fromisoformat(doc["effective_from"]) if doc.get("effective_from") else None,
            effective_to=datetime.fromisoformat(doc["effective_to"]) if doc.get("effective_to") else None,
            tier=doc.get("tier", 2),
            provenance=doc.get("provenance", {}),
            version=doc.get("version", 1),
            history=doc.get("history", []),
            is_bidirectional=doc.get("is_bidirectional", False),
            pair_edge_id=doc.get("pair_edge_id"),
            last_fired_at=datetime.fromisoformat(doc["last_fired_at"]) if doc.get("last_fired_at") else datetime.utcnow(),
            decay_lambda=doc.get("decay_lambda", 0.000012),
        )


@dataclass
class SupportingMention:
    """A single mention that supports a candidate edge."""
    mem_id: str
    srl_conf: float = 0.0  # SRL confidence
    coref_conf: float = 0.0  # Coreference confidence
    ego: float = 0.0  # Ego score of the memory
    observed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CandidateEdge:
    """
    Tier 4 Candidate Edge (before activation scoring).
    See Section 4.3 of DAPPY_UNIFIED_ARCHITECTURE.md
    
    Key changes from v2.0:
    - subject_entity_id/object_entity_id: Entity resolution BEFORE creation (Loophole #2 fix)
    - Keeps spans as fallback for aggregation
    """
    candidate_id: str = field(default_factory=lambda: f"ce_{uuid.uuid4().hex[:12]}")
    user_id: str = ""
    
    # Subject (entity or span)
    subject_entity_id: Optional[str] = None  # Resolved entity ID (Loophole #2 fix)
    subject_span: Dict[str, Any] = field(default_factory=dict)  # Fallback span
    
    # Predicate
    predicate: str = ""
    
    # Object (entity or span)
    object_entity_id: Optional[str] = None  # Resolved entity ID (Loophole #2 fix)
    object_span: Dict[str, Any] = field(default_factory=dict)  # Fallback span
    
    # Supporting mentions
    supporting_mentions: List[SupportingMention] = field(default_factory=list)
    
    # Aggregated features (for activation scoring)
    edge_evidence_count: int = 0
    distinct_session_count: int = 0
    recency_weight: float = 1.0
    frequency_rate: float = 0.0
    
    # Activation score (from LightGBM model)
    activation: float = 0.0
    confidence: float = 0.0
    status: str = "candidate"  # "candidate" | "tier2" | "tier3" | "promoted" | "rejected"
    
    # Timestamps
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_fired_at: datetime = field(default_factory=datetime.utcnow)
    
    # Decay
    decay_lambda: float = 0.000012
    
    # Contradiction score (aggregated penalty)
    contradiction_score: float = 0.0
    
    # Aggregated features (for activation scoring)
    aggregated_features: Dict[str, Any] = field(default_factory=dict)
    
    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB document format."""
        return {
            "_key": self.candidate_id,
            "_from": f"entities/{self.subject_entity_id}",  # Required for ArangoDB edges
            "_to": f"entities/{self.object_entity_id}",      # Required for ArangoDB edges
            "candidate_id": self.candidate_id,
            "user_id": self.user_id,
            "subject_entity_id": self.subject_entity_id,
            "subject_span": self.subject_span,
            "predicate": self.predicate,
            "object_entity_id": self.object_entity_id,
            "object_span": self.object_span,
            "supporting_mentions": [
                {
                    "mem_id": m.mem_id,
                    "srl_conf": m.srl_conf,
                    "coref_conf": m.coref_conf,
                    "ego": m.ego,
                    "observed_at": m.observed_at.isoformat() if m.observed_at else None
                }
                for m in self.supporting_mentions
            ],
            "edge_evidence_count": self.edge_evidence_count,
            "distinct_session_count": self.distinct_session_count,
            "recency_weight": self.recency_weight,
            "frequency_rate": self.frequency_rate,
            "activation": self.activation,
            "confidence": self.confidence,
            "status": self.status,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_fired_at": self.last_fired_at.isoformat() if self.last_fired_at else None,
            "decay_lambda": self.decay_lambda,
            "contradiction_score": self.contradiction_score,
        }
    
    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> 'CandidateEdge':
        """Create from ArangoDB document."""
        mentions = []
        for m in doc.get("supporting_mentions", []):
            mentions.append(SupportingMention(
                mem_id=m.get("mem_id", ""),
                srl_conf=m.get("srl_conf", 0.0),
                coref_conf=m.get("coref_conf", 0.0),
                ego=m.get("ego", 0.0),
                observed_at=datetime.fromisoformat(m["observed_at"]) if m.get("observed_at") else datetime.utcnow()
            ))
        
        return cls(
            candidate_id=doc.get("_key", doc.get("candidate_id", "")),
            user_id=doc.get("user_id", ""),
            subject_entity_id=doc.get("subject_entity_id"),
            subject_span=doc.get("subject_span", {}),
            predicate=doc.get("predicate", ""),
            object_entity_id=doc.get("object_entity_id"),
            object_span=doc.get("object_span", {}),
            supporting_mentions=mentions,
            edge_evidence_count=doc.get("edge_evidence_count", 0),
            distinct_session_count=doc.get("distinct_session_count", 0),
            recency_weight=doc.get("recency_weight", 1.0),
            frequency_rate=doc.get("frequency_rate", 0.0),
            activation=doc.get("activation", 0.0),
            confidence=doc.get("confidence", 0.0),
            status=doc.get("status", "candidate"),
            first_seen=datetime.fromisoformat(doc["first_seen"]) if doc.get("first_seen") else datetime.utcnow(),
            last_fired_at=datetime.fromisoformat(doc["last_fired_at"]) if doc.get("last_fired_at") else datetime.utcnow(),
            decay_lambda=doc.get("decay_lambda", 0.000012),
            contradiction_score=doc.get("contradiction_score", 0.0),
        )
    
    def get_aggregation_key(self) -> tuple:
        """
        Get the key for aggregating candidate edges.
        Uses entity IDs if available, falls back to span text.
        (Loophole #2 fix)
        """
        subject_key = self.subject_entity_id or self.subject_span.get("text", "")
        object_key = self.object_entity_id or self.object_span.get("text", "")
        return (subject_key, self.predicate, object_key)


@dataclass
class Entity:
    """
    Simplified Entity (no cluster membership).
    See Section 4.4 of DAPPY_UNIFIED_ARCHITECTURE.md
    
    Context is derived from edges, not cluster membership:
    - Family context: Query edges WHERE relation_category = 'family'
    - Professional context: Query edges WHERE relation_category = 'professional'
    """
    entity_id: str = field(default_factory=lambda: f"e_{uuid.uuid4().hex[:12]}")
    canonical_name: str = ""
    type: str = "unknown"  # person, organization, concept, location
    aliases: List[str] = field(default_factory=list)
    
    # Embedding for similarity matching
    embedding: List[float] = field(default_factory=list)
    
    # Linked memories
    linked_memories: List[str] = field(default_factory=list)
    
    # Stats
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "mention_frequency": 0,
        "avg_sentiment": 0.0,
        "avg_ego_score": 0.0
    })
    
    # User scope
    user_id: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_arango_doc(self) -> Dict[str, Any]:
        """Convert to ArangoDB document format."""
        return {
            "_key": self.entity_id,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "type": self.type,
            "aliases": self.aliases,
            "embedding": self.embedding,
            "linked_memories": self.linked_memories,
            "stats": self.stats,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_arango_doc(cls, doc: Dict[str, Any]) -> 'Entity':
        """Create from ArangoDB document."""
        return cls(
            entity_id=doc.get("_key", doc.get("entity_id", "")),
            canonical_name=doc.get("canonical_name", ""),
            type=doc.get("type", "unknown"),
            aliases=doc.get("aliases", []),
            embedding=doc.get("embedding", []),
            linked_memories=doc.get("linked_memories", []),
            stats=doc.get("stats", {}),
            user_id=doc.get("user_id", ""),
            created_at=datetime.fromisoformat(doc["created_at"]) if doc.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(doc["updated_at"]) if doc.get("updated_at") else datetime.utcnow(),
        )
    
    def add_alias(self, alias: str):
        """Add an alias if not already present."""
        alias_lower = alias.lower()
        if alias_lower not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)
            self.updated_at = datetime.utcnow()
    
    def link_memory(self, memory_id: str):
        """Link a memory to this entity."""
        if memory_id not in self.linked_memories:
            self.linked_memories.append(memory_id)
            self.stats["mention_frequency"] = len(self.linked_memories)
            self.updated_at = datetime.utcnow()
    
    def update_embedding(
        self,
        memory_embeddings: List[List[float]],
        ego_scores: Optional[List[float]] = None
    ):
        """
        Update entity embedding by aggregating memory embeddings (Phase 1F).
        
        Strategy: Weighted average by ego_score
        
        Args:
            memory_embeddings: List of embedding vectors from linked memories
            ego_scores: Optional list of ego scores for weighting (defaults to equal weights)
        """
        if not memory_embeddings:
            return
        
        try:
            import numpy as np
            
            # Convert to numpy array
            embeddings_array = np.array(memory_embeddings)
            
            # Use ego scores as weights, or equal weights
            if ego_scores and len(ego_scores) == len(memory_embeddings):
                weights = np.array(ego_scores)
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(memory_embeddings)) / len(memory_embeddings)
            
            # Weighted average
            aggregated = np.average(embeddings_array, axis=0, weights=weights)
            
            # Update embedding
            self.embedding = aggregated.tolist()
            self.updated_at = datetime.utcnow()
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update entity embedding: {e}")

