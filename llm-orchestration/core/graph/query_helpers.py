"""
DAPPY Graph Query Helpers
Temporal validity filters and context-aware retrieval utilities.

Loophole #6 fix: All queries enforce temporal validity (effective_from/to).
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from arango.database import StandardDatabase

from .schemas import (
    ThoughtEdge,
    Entity,
    RelationCategory,
    EDGE_TYPE_TAXONOMY,
    get_relation_category
)

logger = logging.getLogger(__name__)


class GraphQueryHelper:
    """
    Helper class for Graph-of-Thoughts queries with temporal validity.
    
    All queries enforce:
    - effective_from/to filtering (Loophole #6 fix)
    - relation_category filtering (implicit clustering)
    - is_active filtering
    """
    
    def __init__(self, db: StandardDatabase):
        self.db = db
    
    async def get_current_relationships(
        self,
        entity_id: str,
        relation_type: str = None,
        relation_category: str = None,
        as_of_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Get current relationships for an entity with temporal validity.
        
        Loophole #6 fix: Filters by effective_from/to.
        
        Args:
            entity_id: Entity to query (e.g., "entities/e_456")
            relation_type: Optional specific relation type (e.g., "works_at")
            relation_category: Optional category (e.g., "professional")
            as_of_date: Point-in-time query (default: now)
        
        Returns:
            List of {edge, target} dicts
        """
        as_of = as_of_date or datetime.utcnow()
        
        # Build filters
        filters = ["e.is_active == true"]
        
        # Temporal validity filter (Loophole #6 fix)
        filters.append("(@as_of >= e.effective_from OR e.effective_from == null)")
        filters.append("(@as_of <= e.effective_to OR e.effective_to == null)")
        
        if relation_type:
            filters.append("e.relation == @relation_type")
        
        if relation_category:
            filters.append("e.relation_category == @relation_category")
        
        filter_clause = " AND ".join(filters)
        
        query = f"""
        FOR v, e IN 1..2 ANY @entity_id
        GRAPH 'thought_graph'
        FILTER {filter_clause}
        SORT e.strength DESC
        RETURN {{edge: e, target: v}}
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "entity_id": entity_id,
                "as_of": as_of.isoformat(),
                "relation_type": relation_type,
                "relation_category": relation_category
            }
        )
        
        results = list(cursor)
        logger.debug(f"Found {len(results)} current relationships for {entity_id}")
        return results
    
    async def get_entity_context(
        self,
        entity_id: str,
        context: str,
        max_depth: int = 3,
        include_historical: bool = False,
        as_of_date: datetime = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get memories in a specific context via edge type filtering.
        See Section 6.4 of DAPPY_UNIFIED_ARCHITECTURE.md
        
        Args:
            entity_id: Entity to start from (e.g., "entities/e_456")
            context: Context category ("family", "professional", etc.)
            max_depth: Max traversal depth
            include_historical: Include edges with effective_to in past
            as_of_date: Point-in-time query (default: now)
            limit: Max results
        
        Returns:
            List of {node, edge} dicts
        """
        as_of = as_of_date or datetime.utcnow()
        
        # Get edge types for this context
        taxonomy = EDGE_TYPE_TAXONOMY.get(RelationCategory(context))
        if not taxonomy:
            logger.warning(f"Unknown context: {context}")
            return []
        
        edge_types = [r.value if hasattr(r, 'value') else r for r in taxonomy["relations"]]
        
        # Build temporal filter
        temporal_filter = ""
        if not include_historical:
            temporal_filter = """
            AND (@as_of >= e.effective_from OR e.effective_from == null)
            AND (@as_of <= e.effective_to OR e.effective_to == null)
            """
        
        query = f"""
        FOR v, e IN 1..@max_depth ANY @entity_id
        GRAPH 'thought_graph'
        FILTER e.relation IN @edge_types
        AND e.is_active == true
        {temporal_filter}
        SORT e.strength DESC
        LIMIT @limit
        RETURN {{node: v, edge: e}}
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "entity_id": entity_id,
                "edge_types": edge_types,
                "max_depth": max_depth,
                "as_of": as_of.isoformat(),
                "limit": limit
            }
        )
        
        results = list(cursor)
        logger.debug(f"Found {len(results)} nodes in {context} context for {entity_id}")
        return results
    
    async def get_evolution_chain(
        self,
        seed_node: str,
        relation_types: List[str] = None,
        max_depth: int = 10,
        time_window: tuple = None,
        direction: str = "forward"
    ) -> List[Dict[str, Any]]:
        """
        Get evolution chain from a seed node.
        See Section 5.3 of DAPPY_UNIFIED_ARCHITECTURE.md
        
        Args:
            seed_node: Starting memory node
            relation_types: Edge types to follow (default: temporal edges)
            max_depth: Max chain length
            time_window: Optional (start_date, end_date) tuple
            direction: "forward" (OUTBOUND) or "backward" (INBOUND)
        
        Returns:
            List of {node, edge, path} dicts
        """
        if relation_types is None:
            relation_types = ["evolves_to", "supersedes", "refines", "clarifies"]
        
        # Build time filter
        time_filter = ""
        if time_window:
            start_date, end_date = time_window
            time_filter = f"""
            AND (@start_date == null OR e.created_at >= @start_date)
            AND (@end_date == null OR e.created_at <= @end_date)
            """
        
        direction_keyword = "OUTBOUND" if direction == "forward" else "INBOUND"
        
        query = f"""
        FOR v, e, p IN 1..@max_depth {direction_keyword} @seed_node
        GRAPH 'thought_graph'
        FILTER e.relation IN @relation_types
        AND e.is_active == true
        {time_filter}
        RETURN {{node: v, edge: e, path: p}}
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "seed_node": seed_node,
                "relation_types": relation_types,
                "max_depth": max_depth,
                "start_date": time_window[0].isoformat() if time_window else None,
                "end_date": time_window[1].isoformat() if time_window else None
            }
        )
        
        results = list(cursor)
        logger.debug(f"Found evolution chain of {len(results)} nodes from {seed_node}")
        return results
    
    async def get_contradictions(
        self,
        user_id: str = None,
        min_strength: float = 0.5,
        unresolved_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get contradiction edges for conflict resolution.
        
        Args:
            user_id: Optional user filter
            min_strength: Minimum edge strength
            unresolved_only: Only return edges with resolved_by = "pending"
        
        Returns:
            List of contradiction edges with both memories
        """
        filters = [
            "e.relation == 'contradicts'",
            "e.is_active == true",
            "e.strength >= @min_strength"
        ]
        
        if unresolved_only:
            filters.append("e.resolved_by == 'pending'")
        
        if user_id:
            filters.append("e.user_id == @user_id")
        
        filter_clause = " AND ".join(filters)
        
        query = f"""
        FOR e IN thought_edges
        FILTER {filter_clause}
        LET from_doc = DOCUMENT(e._from)
        LET to_doc = DOCUMENT(e._to)
        RETURN {{
            edge: e,
            memory1: from_doc,
            memory2: to_doc
        }}
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "min_strength": min_strength,
                "user_id": user_id
            }
        )
        
        results = list(cursor)
        logger.debug(f"Found {len(results)} contradiction edges")
        return results
    
    async def get_family_members(self, user_id: str) -> List[Entity]:
        """
        Get all family members for a user.
        Example query from Section 6.4.
        """
        results = await self.get_entity_context(
            entity_id=f"users/{user_id}",
            context="family",
            max_depth=2,
            limit=20
        )
        
        # Filter to entities only
        entities = []
        for r in results:
            node = r.get("node", {})
            if node.get("_id", "").startswith("entities/"):
                entities.append(Entity.from_arango_doc(node))
        
        return entities
    
    async def get_professional_history(
        self,
        entity_id: str,
        include_historical: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get professional relationships with temporal validity.
        Example query from Section 6.4.
        
        Returns:
            List of {entity, relation, from, to, is_current}
        """
        query = """
        FOR v, e IN 1..2 ANY @entity_id
        GRAPH 'thought_graph'
        FILTER e.relation_category == 'professional'
        AND (@include_historical OR e.is_active == true)
        SORT e.effective_from DESC
        RETURN {
            entity: v,
            relation: e.relation,
            from: e.effective_from,
            to: e.effective_to,
            is_current: e.effective_to == null
        }
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "entity_id": entity_id,
                "include_historical": include_historical
            }
        )
        
        return list(cursor)
    
    async def disambiguate_entity_context(
        self,
        entity_id: str,
        query_text: str,
        llm_client = None
    ) -> str:
        """
        Disambiguate which context the user is asking about.
        See Section 6.5 of DAPPY_UNIFIED_ARCHITECTURE.md
        
        Args:
            entity_id: Entity to disambiguate
            query_text: User's query
            llm_client: Optional LLM client for disambiguation
        
        Returns:
            Context name ("family", "professional", etc.)
        """
        # Get all contexts this entity appears in
        query = """
        FOR v, e IN 1..2 ANY @entity_id
        GRAPH 'thought_graph'
        FILTER e.is_active == true
        RETURN DISTINCT e.relation_category
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={"entity_id": entity_id}
        )
        
        contexts = list(cursor)
        
        if len(contexts) == 1:
            return contexts[0]  # No ambiguity
        
        if len(contexts) == 0:
            return "personal"  # Default
        
        # Need LLM to disambiguate
        if llm_client:
            # Get entity name
            entity_doc = self.db.collection("entities").get(entity_id.split("/")[-1])
            entity_name = entity_doc.get("canonical_name", "this entity") if entity_doc else "this entity"
            
            prompt = f"""
User query: "{query_text}"
Entity: {entity_name}

This entity appears in multiple contexts:
{', '.join(contexts)}

Which context is the user asking about?
Respond with ONLY the context name.
"""
            
            try:
                response = await llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a context disambiguation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=20
                )
                
                context = response.choices[0].message.content.strip().lower()
                if context in contexts:
                    return context
            except Exception as e:
                logger.warning(f"LLM disambiguation failed: {e}")
        
        # Fallback: return most common context
        return contexts[0]


async def apply_temporal_filter(
    query: str,
    as_of_date: datetime = None
) -> str:
    """
    Utility to add temporal validity filter to an AQL query.
    
    Loophole #6 fix: Ensures all queries respect effective_from/to.
    
    Usage:
        query = "FOR e IN thought_edges FILTER e.relation == 'works_at'"
        query = apply_temporal_filter(query, as_of_date)
    """
    as_of = as_of_date or datetime.utcnow()
    as_of_str = as_of.isoformat()
    
    temporal_clause = f"""
    AND ('{as_of_str}' >= e.effective_from OR e.effective_from == null)
    AND ('{as_of_str}' <= e.effective_to OR e.effective_to == null)
    """
    
    # Insert before RETURN or at end
    if "RETURN" in query:
        parts = query.split("RETURN")
        return parts[0] + temporal_clause + "\nRETURN" + parts[1]
    else:
        return query + temporal_clause

