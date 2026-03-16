"""
DAPPY Edge Store
Manages ThoughtEdge and CandidateEdge collections in ArangoDB.

Phase 1 Implementation:
- Create candidate_edges collection
- Create thought_edges collection with relation_category
- Implement temporal validity filters (Loophole #6 fix)
- Implement bidirectional edge creation (Loophole #4 fix)
- Implement unified edge strength decay (Loophole #3 fix)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from math import exp

from arango import ArangoClient
from arango.database import StandardDatabase

from .schemas import (
    ThoughtEdge, 
    CandidateEdge, 
    SupportingMention,
    EdgeType,
    RelationCategory,
    get_relation_category,
    HIGH_IMPACT_RELATIONS
)

logger = logging.getLogger(__name__)


class CandidateEdgeStore:
    """
    Manages Tier 4 candidate edges in ArangoDB.
    
    Responsibilities:
    - Create/update candidate edges
    - Aggregate by (subject, predicate, object)
    - Track supporting mentions
    - Update activation scores
    """
    
    COLLECTION_NAME = "candidate_edges"
    
    def __init__(self, db: StandardDatabase):
        self.db = db
        self._init_collection()
    
    def _init_collection(self):
        """Create collection and indexes if they don't exist."""
        if not self.db.has_collection(self.COLLECTION_NAME):
            self.collection = self.db.create_collection(self.COLLECTION_NAME)
            logger.info(f"Created collection: {self.COLLECTION_NAME}")
        else:
            self.collection = self.db.collection(self.COLLECTION_NAME)
    
    async def get(self, candidate_id: str) -> Optional[CandidateEdge]:
        """
        Get a candidate edge by ID.
        
        Args:
            candidate_id: The candidate edge ID
        
        Returns:
            CandidateEdge or None if not found
        """
        try:
            doc = self.collection.get(candidate_id)
            if doc:
                return CandidateEdge.from_arango_doc(doc)
            return None
        except Exception as e:
            logger.error(f"Error fetching candidate edge {candidate_id}: {e}")
            return None
        
        # Create indexes
        try:
            self.collection.add_persistent_index(
                fields=["user_id", "status"],
                name="user_status_idx"
            )
            self.collection.add_persistent_index(
                fields=["activation"],
                name="activation_idx"
            )
            self.collection.add_persistent_index(
                fields=["first_seen"],
                name="first_seen_idx"
            )
            # Aggregation key index (Loophole #2 fix)
            self.collection.add_persistent_index(
                fields=["subject_entity_id", "predicate", "object_entity_id"],
                name="aggregation_key_idx"
            )
            logger.info("CandidateEdge indexes created")
        except Exception as e:
            logger.debug(f"Index creation warning (may exist): {e}")
    
    async def create_or_update(self, candidate: CandidateEdge) -> CandidateEdge:
        """
        Create or update a candidate edge.
        If exists with same aggregation key, merge supporting mentions.
        """
        # Check for existing candidate with same key
        existing = await self.find_by_aggregation_key(
            user_id=candidate.user_id,
            subject_key=candidate.subject_entity_id or candidate.subject_span.get("text", ""),
            predicate=candidate.predicate,
            object_key=candidate.object_entity_id or candidate.object_span.get("text", "")
        )
        
        if existing:
            # Merge supporting mentions
            existing_mem_ids = {m.mem_id for m in existing.supporting_mentions}
            for mention in candidate.supporting_mentions:
                if mention.mem_id not in existing_mem_ids:
                    existing.supporting_mentions.append(mention)
            
            # Update aggregated features
            existing.edge_evidence_count = len(existing.supporting_mentions)
            existing.last_fired_at = datetime.utcnow()
            
            # Update recency weight (exponential decay)
            delta_t = (datetime.utcnow() - existing.first_seen).total_seconds()
            existing.recency_weight = exp(-existing.decay_lambda * delta_t)
            
            # Save
            self.collection.update(existing.to_arango_doc())
            logger.info(f"🔄 CandidateEdgeStore: Updated edge {existing.candidate_id} (evidence={existing.edge_evidence_count}, total_mentions={len(existing.supporting_mentions)})")
            return existing
        else:
            # Create new
            candidate.edge_evidence_count = len(candidate.supporting_mentions)
            self.collection.insert(candidate.to_arango_doc())
            logger.info(f"💾 CandidateEdgeStore: Created edge {candidate.candidate_id} ({candidate.subject_entity_id} --[{candidate.predicate}]--> {candidate.object_entity_id}, mentions={len(candidate.supporting_mentions)})")
            return candidate
    
    async def find_by_aggregation_key(
        self,
        user_id: str,
        subject_key: str,
        predicate: str,
        object_key: str
    ) -> Optional[CandidateEdge]:
        """Find candidate edge by aggregation key."""
        query = """
        FOR c IN @@collection
        FILTER c.user_id == @user_id
        AND (c.subject_entity_id == @subject_key OR c.subject_span.text == @subject_key)
        AND c.predicate == @predicate
        AND (c.object_entity_id == @object_key OR c.object_span.text == @object_key)
        LIMIT 1
        RETURN c
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "user_id": user_id,
                "subject_key": subject_key,
                "predicate": predicate,
                "object_key": object_key
            }
        )
        
        results = list(cursor)
        if results:
            return CandidateEdge.from_arango_doc(results[0])
        return None
    
    async def get_pending_for_consolidation(
        self,
        limit: int = 100,
        priority: str = "LOW"
    ) -> List[CandidateEdge]:
        """
        Get candidate edges pending consolidation.
        Priority rules (Loophole #9 fix):
        - HIGH: evidence >= 3, contradiction > 0.7, tier1 mentions, high-impact relations
        - MED: evidence >= 2, session >= 2, tier2 mentions
        - LOW: everything else
        """
        if priority == "HIGH":
            query = """
            FOR c IN @@collection
            FILTER c.status == 'candidate'
            AND (
                c.edge_evidence_count >= 3 OR
                c.contradiction_score > 0.7 OR
                c.predicate IN @high_impact
            )
            SORT c.last_fired_at DESC
            LIMIT @limit
            RETURN c
            """
        elif priority == "MED":
            query = """
            FOR c IN @@collection
            FILTER c.status == 'candidate'
            AND c.edge_evidence_count >= 2
            AND c.distinct_session_count >= 2
            SORT c.last_fired_at DESC
            LIMIT @limit
            RETURN c
            """
        else:  # LOW
            query = """
            FOR c IN @@collection
            FILTER c.status == 'candidate'
            SORT c.first_seen ASC
            LIMIT @limit
            RETURN c
            """
        
        high_impact_values = [r.value for r in HIGH_IMPACT_RELATIONS]
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "limit": limit,
                "high_impact": high_impact_values
            }
        )
        
        return [CandidateEdge.from_arango_doc(doc) for doc in cursor]
    
    async def update_activation(
        self,
        candidate_id: str,
        activation: float,
        status: str
    ):
        """Update activation score and status."""
        self.collection.update({
            "_key": candidate_id,
            "activation": activation,
            "status": status,
            "last_fired_at": datetime.utcnow().isoformat()
        })
    
    async def delete_expired(self, older_than_days: int = 7) -> int:
        """Delete candidate edges that never promoted (Tier 4 TTL)."""
        query = """
        FOR c IN @@collection
        FILTER c.status == 'candidate'
        AND DATE_DIFF(c.first_seen, DATE_NOW(), 'd') > @days
        REMOVE c IN @@collection
        RETURN OLD
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "days": older_than_days
            }
        )
        deleted = len(list(cursor))
        logger.info(f"Deleted {deleted} expired candidate edges")
        return deleted


class ThoughtEdgeStore:
    """
    Manages canonical ThoughtEdges in ArangoDB.
    
    Responsibilities:
    - Create edges with relation_category (implicit clustering)
    - Create bidirectional contradiction edges (Loophole #4 fix)
    - Enforce temporal validity (Loophole #6 fix)
    - Apply unified edge strength decay (Loophole #3 fix)
    """
    
    COLLECTION_NAME = "thought_edges"
    
    def __init__(self, db: StandardDatabase):
        self.db = db
        self._init_collection()
    
    def _init_collection(self):
        """Create edge collection and indexes if they don't exist."""
        if not self.db.has_collection(self.COLLECTION_NAME):
            self.collection = self.db.create_collection(self.COLLECTION_NAME, edge=True)
            logger.info(f"Created edge collection: {self.COLLECTION_NAME}")
        else:
            self.collection = self.db.collection(self.COLLECTION_NAME)
        
        # Create indexes
        try:
            self.collection.add_persistent_index(
                fields=["relation", "is_active"],
                name="relation_active_idx"
            )
            # Implicit clustering index (Section 6)
            self.collection.add_persistent_index(
                fields=["relation_category", "is_active"],
                name="category_active_idx"
            )
            self.collection.add_persistent_index(
                fields=["user_id", "tier"],
                name="user_tier_idx"
            )
            self.collection.add_persistent_index(
                fields=["strength"],
                name="strength_idx"
            )
            # Temporal validity index (Loophole #6)
            self.collection.add_persistent_index(
                fields=["effective_from", "effective_to"],
                name="temporal_validity_idx"
            )
            # Bidirectional edge index (Loophole #4)
            self.collection.add_persistent_index(
                fields=["is_bidirectional", "pair_edge_id"],
                name="bidirectional_idx"
            )
            logger.info("ThoughtEdge indexes created")
        except Exception as e:
            logger.debug(f"Index creation warning (may exist): {e}")
        
        # Create graph if not exists
        self._init_graph()
    
    def _init_graph(self):
        """Create thought_graph if not exists."""
        graph_name = "thought_graph"
        if not self.db.has_graph(graph_name):
            self.db.create_graph(
                graph_name,
                edge_definitions=[{
                    "edge_collection": self.COLLECTION_NAME,
                    "from_vertex_collections": ["memories", "entities"],
                    "to_vertex_collections": ["memories", "entities"]
                }]
            )
            logger.info(f"Created graph: {graph_name}")
    
    async def create_edge(
        self,
        from_node: str,
        to_node: str,
        relation: str,
        strength: float = 0.5,
        supporting_mentions: List[str] = None,
        resolved_by: str = "auto",
        resolution_text: str = None,
        effective_from: datetime = None,
        effective_to: datetime = None,
        tier: int = 2
    ) -> ThoughtEdge:
        """
        Create a ThoughtEdge with relation_category (implicit clustering).
        """
        # Determine relation category
        relation_category = get_relation_category(relation)
        
        edge = ThoughtEdge(
            _from=from_node,
            _to=to_node,
            relation=relation,
            relation_category=relation_category.value if relation_category else "temporal",
            strength=strength,
            supporting_mentions=supporting_mentions or [],
            resolved_by=resolved_by,
            resolution_text=resolution_text,
            effective_from=effective_from or datetime.utcnow(),
            effective_to=effective_to,
            tier=tier
        )
        
        self.collection.insert(edge.to_arango_doc())
        logger.info(f"🎯 ThoughtEdgeStore: PROMOTED to ThoughtEdge {edge.id} ({edge._from} --[{relation}]--> {edge._to}, strength={edge.strength:.2f}, tier={edge.tier})")
        return edge
    
    async def create_bidirectional_edge(
        self,
        node1: str,
        node2: str,
        relation: str,
        strength: float = 0.5,
        supporting_mentions: List[str] = None,
        resolved_by: str = "auto"
    ) -> tuple:
        """
        Create bidirectional edges (Loophole #4 fix).
        Used for contradiction edges where A contradicts B ⟺ B contradicts A.
        
        Returns: (edge1, edge2)
        """
        relation_category = get_relation_category(relation)
        cat_value = relation_category.value if relation_category else "temporal"
        
        # Create edge 1: node1 → node2
        edge1 = ThoughtEdge(
            _from=node1,
            _to=node2,
            relation=relation,
            relation_category=cat_value,
            strength=strength,
            supporting_mentions=supporting_mentions or [],
            resolved_by=resolved_by,
            is_bidirectional=True
        )
        
        # Create edge 2: node2 → node1
        edge2 = ThoughtEdge(
            _from=node2,
            _to=node1,
            relation=relation,
            relation_category=cat_value,
            strength=strength,
            supporting_mentions=supporting_mentions or [],
            resolved_by=resolved_by,
            is_bidirectional=True
        )
        
        # Link them
        edge1.pair_edge_id = edge2.id
        edge2.pair_edge_id = edge1.id
        
        # Insert both
        self.collection.insert(edge1.to_arango_doc())
        self.collection.insert(edge2.to_arango_doc())
        
        logger.info(f"Created bidirectional edges: {edge1.id} ↔ {edge2.id} ({relation})")
        return (edge1, edge2)
    
    async def get_edges_by_context(
        self,
        entity_id: str,
        context: str,
        max_depth: int = 3,
        include_historical: bool = False,
        as_of_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Get edges filtered by context (implicit clustering).
        See Section 6.4 of DAPPY_UNIFIED_ARCHITECTURE.md
        
        Args:
            entity_id: Entity to start from (e.g., "entities/e_456")
            context: Context category ("family", "professional", etc.)
            max_depth: Max traversal depth
            include_historical: Include edges with effective_to in past
            as_of_date: Point-in-time query (default: now)
        """
        as_of = as_of_date or datetime.utcnow()
        
        # Temporal validity filter (Loophole #6 fix)
        temporal_filter = ""
        if not include_historical:
            temporal_filter = """
            AND (@as_of >= e.effective_from OR e.effective_from == null)
            AND (@as_of <= e.effective_to OR e.effective_to == null)
            """
        
        query = f"""
        FOR v, e IN 1..@max_depth ANY @entity_id
        GRAPH 'thought_graph'
        FILTER e.relation_category == @context
        AND e.is_active == true
        {temporal_filter}
        SORT e.strength DESC
        LIMIT 10
        RETURN {{node: v, edge: e}}
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "entity_id": entity_id,
                "context": context,
                "max_depth": max_depth,
                "as_of": as_of.isoformat()
            }
        )
        
        return list(cursor)
    
    async def update_edge_strength(
        self,
        edge_id: str,
        new_evidence: float = 0.1
    ) -> ThoughtEdge:
        """
        Update edge strength with unified decay formula (Loophole #3 fix).
        
        Formula:
        1. Apply decay since last update: strength * exp(-λ * Δt)
        2. Add new evidence (capped at 1.0)
        """
        edge_doc = self.collection.get(edge_id)
        if not edge_doc:
            raise ValueError(f"Edge not found: {edge_id}")
        
        edge = ThoughtEdge.from_arango_doc(edge_doc)
        
        # Step 1: Apply decay
        delta_t = (datetime.utcnow() - edge.last_fired_at).total_seconds()
        decayed_strength = edge.strength * exp(-edge.decay_lambda * delta_t)
        
        # Step 2: Add new evidence (capped at 1.0)
        new_strength = min(1.0, decayed_strength + new_evidence)
        
        # Step 3: Update
        edge.strength = new_strength
        edge.last_fired_at = datetime.utcnow()
        
        self.collection.update(edge.to_arango_doc())
        
        logger.debug(f"Updated edge strength: {edge_id} ({edge.strength:.3f})")
        return edge
    
    async def get_contradictions(
        self,
        user_id: str,
        min_strength: float = 0.5
    ) -> List[ThoughtEdge]:
        """Get active contradiction edges."""
        query = """
        FOR e IN @@collection
        FILTER e.relation == 'contradicts'
        AND e.is_active == true
        AND e.strength >= @min_strength
        RETURN e
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "min_strength": min_strength
            }
        )
        
        return [ThoughtEdge.from_arango_doc(doc) for doc in cursor]
    
    async def mark_superseded(
        self,
        edge_id: str,
        superseded_by: str
    ):
        """Mark an edge as superseded (historical)."""
        self.collection.update({
            "_key": edge_id,
            "is_active": False,
            "effective_to": datetime.utcnow().isoformat(),
            "superseded_by": superseded_by
        })
        logger.info(f"Marked edge as superseded: {edge_id}")
    
    async def apply_decay_batch(self, older_than_days: int = 7) -> int:
        """Apply decay to edges not accessed recently."""
        query = """
        FOR e IN @@collection
        FILTER e.is_active == true
        AND DATE_DIFF(e.last_fired_at, DATE_NOW(), 'd') > @days
        LET delta_t = DATE_DIFF(e.last_fired_at, DATE_NOW(), 's')
        LET decayed = e.strength * EXP(-e.decay_lambda * delta_t)
        UPDATE e WITH {
            strength: decayed,
            last_fired_at: DATE_NOW()
        } IN @@collection
        RETURN NEW
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "days": older_than_days
            }
        )
        updated = len(list(cursor))
        logger.info(f"Applied decay to {updated} edges")
        return updated

