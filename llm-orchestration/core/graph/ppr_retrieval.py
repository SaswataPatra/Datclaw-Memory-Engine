"""
DAPPY PPR-based Graph Retrieval

Implements Personalized PageRank for multi-hop retrieval.
Uses NetworkX for local graph processing.

Key Features:
1. Build local graph from ArangoDB
2. Run PPR from seed entities
3. Context-aware filtering by relation_category
4. Temporal validity filtering

Phase 3 Implementation
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Lazy import networkx to avoid ImportError when not installed
# import networkx as nx  # imported in __init__ when needed

from .schemas import ThoughtEdge, Entity, RelationCategory

logger = logging.getLogger(__name__)


@dataclass
class PPRResult:
    """Result of PPR retrieval."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    scores: Dict[str, float]
    hops: int
    context_filter: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PPRRetrieval:
    """
    Personalized PageRank retrieval for multi-hop reasoning.
    
    Uses NetworkX for local graph operations.
    ArangoDB is used only for loading the graph.
    
    Algorithm:
    1. Load relevant subgraph from ArangoDB
    2. Build NetworkX DiGraph
    3. Run PPR from seed nodes
    4. Return top-k nodes by PPR score
    """
    
    def __init__(
        self,
        db,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PPR retrieval.
        
        Args:
            db: ArangoDB database connection
            config: Configuration dict with:
                - ppr.alpha: Damping factor (default 0.85)
                - ppr.max_iter: Max iterations (default 100)
                - ppr.top_k: Return top k nodes (default 10)
        """
        # Import networkx here to avoid module-level import error
        try:
            import networkx as nx
            self.nx = nx
        except ImportError:
            raise ImportError(
                "PPR retrieval requires networkx. Install with: pip install networkx"
            )
        
        self.db = db
        self.config = config or {}
        
        # PPR settings
        ppr_config = self.config.get('ppr', {})
        self.alpha = ppr_config.get('alpha', 0.85)
        self.max_iter = ppr_config.get('max_iter', 100)
        self.default_top_k = ppr_config.get('top_k', 10)
        
        logger.info(f"✅ PPRRetrieval initialized")
        logger.info(f"   Alpha: {self.alpha}, Max iter: {self.max_iter}")
    
    async def retrieve(
        self,
        user_id: str,
        seed_entities: List[str],
        context_filter: Optional[str] = None,
        temporal_filter: Optional[datetime] = None,
        max_hops: int = 3,
        top_k: int = None
    ) -> PPRResult:
        """
        Retrieve related nodes using PPR.
        
        Args:
            user_id: User ID
            seed_entities: List of entity IDs to start from
            context_filter: Optional relation category filter
            temporal_filter: Optional timestamp for temporal validity
            max_hops: Maximum hops from seed nodes
            top_k: Number of top nodes to return
        
        Returns:
            PPRResult with nodes, edges, and scores
        """
        top_k = top_k or self.default_top_k
        
        # Step 1: Load subgraph from ArangoDB
        graph, node_data, edge_data = await self._load_subgraph(
            user_id=user_id,
            seed_entities=seed_entities,
            context_filter=context_filter,
            temporal_filter=temporal_filter,
            max_hops=max_hops
        )
        
        if not graph.nodes():
            logger.debug("Empty graph, returning empty result")
            return PPRResult(
                nodes=[],
                edges=[],
                scores={},
                hops=0,
                context_filter=context_filter
            )
        
        # Step 2: Run PPR
        personalization = {node: 1.0 / len(seed_entities) for node in seed_entities if node in graph}
        
        if not personalization:
            logger.debug("No seed entities in graph")
            return PPRResult(
                nodes=[],
                edges=[],
                scores={},
                hops=0,
                context_filter=context_filter
            )
        
        try:
            ppr_scores = self.nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter
            )
        except self.nx.PowerIterationFailedConvergence:
            logger.warning("PPR did not converge, using partial result")
            ppr_scores = self.nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=1e-4  # More lenient tolerance
            )
        
        # Step 3: Get top-k nodes
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = sorted_nodes[:top_k]
        
        # Build result
        result_nodes = []
        for node_id, score in top_nodes:
            node_info = node_data.get(node_id, {})
            # Ensure entity_id is always present
            if "entity_id" not in node_info:
                node_info["entity_id"] = node_id
            node_info["ppr_score"] = score
            result_nodes.append(node_info)
        
        # Get edges between top nodes
        top_node_ids = {node_id for node_id, _ in top_nodes}
        result_edges = [
            edge_data[edge]
            for edge in graph.edges()
            if edge[0] in top_node_ids and edge[1] in top_node_ids
            and edge in edge_data
        ]
        
        return PPRResult(
            nodes=result_nodes,
            edges=result_edges,
            scores=dict(top_nodes),
            hops=max_hops,
            context_filter=context_filter,
            metadata={
                "total_nodes": graph.number_of_nodes(),
                "total_edges": graph.number_of_edges()
            }
        )
    
    async def _load_subgraph(
        self,
        user_id: str,
        seed_entities: List[str],
        context_filter: Optional[str] = None,
        temporal_filter: Optional[datetime] = None,
        max_hops: int = 3
    ) -> Tuple[Any, Dict[str, Any], Dict[Tuple[str, str], Any]]:  # Returns (nx.DiGraph, node_data, edge_data)
        """
        Load subgraph from ArangoDB.
        
        Uses AQL traversal to get nodes within max_hops.
        """
        graph = self.nx.DiGraph()
        node_data = {}
        edge_data = {}
        
        # Build AQL query
        filters = ["e.user_id == @user_id"]
        bind_vars = {
            "user_id": user_id,
            "seeds": seed_entities,
            "max_hops": max_hops
        }
        
        if context_filter:
            filters.append("e.relation_category == @context_filter")
            bind_vars["context_filter"] = context_filter
        
        if temporal_filter:
            ts = temporal_filter.isoformat()
            filters.append(f"(e.effective_from == null OR e.effective_from <= '{ts}')")
            filters.append(f"(e.effective_to == null OR e.effective_to >= '{ts}')")
        
        filter_clause = " AND ".join(filters)
        
        # Step 1: Query thought_graph (promoted edges)
        thought_query = f"""
        FOR seed IN @seeds
            LET start_node = DOCUMENT(CONCAT('entities/', seed))
            FILTER start_node != null
            
            FOR v, e, p IN 1..@max_hops ANY start_node
                GRAPH 'thought_graph'
                FILTER {filter_clause}
                RETURN {{
                    vertex: v,
                    edge: e,
                    path_length: LENGTH(p.edges),
                    source: 'thought_edge'
                }}
        """
        
        # Step 2: Query candidate_edges (not yet promoted)
        # Use manual traversal since candidate_edges aren't in a named graph
        candidate_query = """
        FOR seed IN @seeds
            LET start_entity = DOCUMENT(CONCAT('entities/', seed))
            FILTER start_entity != null
            
            // Find all candidate edges where this entity is subject or object
            FOR edge IN candidate_edges
                FILTER edge.user_id == @user_id
                FILTER edge.subject_entity_id == seed OR edge.object_entity_id == seed
                
                // Get the other entity
                LET other_id = (edge.subject_entity_id == seed ? edge.object_entity_id : edge.subject_entity_id)
                LET other_entity = DOCUMENT(CONCAT('entities/', other_id))
                FILTER other_entity != null
                
                RETURN {
                    vertex: other_entity,
                    edge: edge,
                    path_length: 1,
                    source: 'candidate_edge'
                }
        """
        
        try:
            # Execute both queries with appropriate bind vars
            thought_cursor = self.db.aql.execute(thought_query, bind_vars=bind_vars)
            
            # Candidate query doesn't use max_hops
            candidate_bind_vars = {
                'user_id': user_id,
                'seeds': seed_entities
            }
            candidate_cursor = self.db.aql.execute(candidate_query, bind_vars=candidate_bind_vars)
            
            # Merge results
            thought_results = list(thought_cursor)
            candidate_results = list(candidate_cursor)
            all_results = thought_results + candidate_results
            
            logger.info(f"   Seeds: {seed_entities}")
            logger.info(f"   Loaded {len(thought_results)} results from thought_graph")
            logger.info(f"   Loaded {len(candidate_results)} results from candidate_edges")
            logger.info(f"   Total: {len(all_results)} results")
            
            if len(all_results) == 0:
                for seed in seed_entities:
                    doc = self.db.collection('entities').get(seed)
                    if doc:
                        logger.warning(f"   Seed '{seed}' exists: canonical_name='{doc.get('canonical_name')}', type={doc.get('type')}")
                    else:
                        logger.warning(f"   Seed '{seed}' NOT FOUND in entities collection")
                    
                    ce_count = self.db.aql.execute(
                        "RETURN LENGTH(FOR e IN candidate_edges FILTER e.subject_entity_id == @s OR e.object_entity_id == @s RETURN 1)",
                        bind_vars={"s": seed}
                    )
                    count = list(ce_count)[0]
                    logger.warning(f"   Seed '{seed}' has {count} candidate edges (any user)")
                    
                    ce_user_count = self.db.aql.execute(
                        "RETURN LENGTH(FOR e IN candidate_edges FILTER e.user_id == @uid AND (e.subject_entity_id == @s OR e.object_entity_id == @s) RETURN 1)",
                        bind_vars={"s": seed, "uid": user_id}
                    )
                    ucount = list(ce_user_count)[0]
                    logger.warning(f"   Seed '{seed}' has {ucount} candidate edges (user={user_id})")
            
            for doc in all_results:
                vertex = doc.get("vertex", {})
                edge = doc.get("edge", {})
                source = doc.get("source", "thought_edge")
                
                if vertex:
                    node_id = vertex.get("_key") or vertex.get("entity_id")
                    if node_id:
                        graph.add_node(node_id)
                        node_data[node_id] = {
                            "entity_id": node_id,
                            "canonical_name": vertex.get("canonical_name"),
                            "type": vertex.get("type"),
                            "metadata": vertex.get("metadata", {})
                        }
                
                if edge:
                    # Handle both thought_edges and candidate_edges
                    if source == "candidate_edge":
                        # Candidate edges use subject_entity_id/object_entity_id
                        from_id = edge.get("subject_entity_id")
                        to_id = edge.get("object_entity_id")
                        strength = edge.get("aggregated_features", {}).get("avg_confidence", 0.5)
                        predicate = edge.get("predicate")
                        relation_category = edge.get("aggregated_features", {}).get("relation_category", "factual")
                        is_bidirectional = False
                    else:
                        # Thought edges use _from/_to
                        from_id = edge.get("_from", "").split("/")[-1]
                        to_id = edge.get("_to", "").split("/")[-1]
                        strength = edge.get("strength", 0.5)
                        predicate = edge.get("predicate")
                        relation_category = edge.get("relation_category")
                        is_bidirectional = edge.get("is_bidirectional", False)
                    
                    if from_id and to_id:
                        # Add edge with weight based on strength
                        graph.add_edge(from_id, to_id, weight=strength)
                        
                        edge_data[(from_id, to_id)] = {
                            "from": from_id,
                            "to": to_id,
                            "predicate": predicate,
                            "relation_category": relation_category,
                            "strength": strength,
                            "source": source
                        }
                        
                        # Add reverse edge if bidirectional
                        if is_bidirectional:
                            graph.add_edge(to_id, from_id, weight=strength)
                            edge_data[(to_id, from_id)] = edge_data[(from_id, to_id)]
        
        except Exception as e:
            logger.error(f"Failed to load subgraph: {e}")
            # Return empty graph on error
        
        # Also add seed entities
        for seed in seed_entities:
            if seed not in graph:
                graph.add_node(seed)
                node_data[seed] = {"entity_id": seed}
        
        logger.debug(
            f"Loaded subgraph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        
        return graph, node_data, edge_data
    
    async def retrieve_context(
        self,
        user_id: str,
        query_entities: List[str],
        relation_category: str = None,
        as_of: datetime = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context for a query.
        
        Simplified interface for context retrieval.
        
        Args:
            user_id: User ID
            query_entities: Entities from the query
            relation_category: Filter by category (e.g., "family")
            as_of: Point-in-time for temporal queries
            limit: Max results
        
        Returns:
            List of context items (nodes + edges)
        """
        result = await self.retrieve(
            user_id=user_id,
            seed_entities=query_entities,
            context_filter=relation_category,
            temporal_filter=as_of,
            max_hops=2,
            top_k=limit
        )
        
        # Flatten to context items
        context = []
        
        # Add nodes as context
        for node in result.nodes:
            context.append({
                "type": "entity",
                "entity_id": node.get("entity_id"),
                "name": node.get("canonical_name"),
                "relevance": node.get("ppr_score", 0.5)
            })
        
        # Add edges as context
        for edge in result.edges:
            context.append({
                "type": "relation",
                "from": edge.get("from"),
                "to": edge.get("to"),
                "predicate": edge.get("predicate"),
                "relevance": edge.get("strength", 0.5)
            })
        
        return context
    
    async def find_path(
        self,
        user_id: str,
        source_entity: str,
        target_entity: str,
        context_filter: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path between two entities.
        
        Args:
            user_id: User ID
            source_entity: Source entity ID
            target_entity: Target entity ID
            context_filter: Optional relation category filter
        
        Returns:
            List of nodes/edges in path, or None if no path exists
        """
        # Load subgraph containing both entities
        graph, node_data, edge_data = await self._load_subgraph(
            user_id=user_id,
            seed_entities=[source_entity, target_entity],
            context_filter=context_filter,
            max_hops=5
        )
        
        if source_entity not in graph or target_entity not in graph:
            return None
        
        try:
            path = self.nx.shortest_path(graph, source_entity, target_entity)
            
            # Build path result
            result = []
            for i, node_id in enumerate(path):
                result.append({
                    "type": "entity",
                    "entity_id": node_id,
                    **node_data.get(node_id, {})
                })
                
                if i < len(path) - 1:
                    edge_key = (node_id, path[i + 1])
                    if edge_key in edge_data:
                        result.append({
                            "type": "edge",
                            **edge_data[edge_key]
                        })
            
            return result
            
        except self.nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return None
    
    async def get_entity_neighborhood(
        self,
        user_id: str,
        entity_id: str,
        hops: int = 1,
        context_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get immediate neighborhood of an entity.
        
        Args:
            user_id: User ID
            entity_id: Entity ID
            hops: Number of hops (default 1)
            context_filter: Optional relation category filter
        
        Returns:
            Dict with neighbors and relationships
        """
        graph, node_data, edge_data = await self._load_subgraph(
            user_id=user_id,
            seed_entities=[entity_id],
            context_filter=context_filter,
            max_hops=hops
        )
        
        if entity_id not in graph:
            return {"entity_id": entity_id, "neighbors": [], "edges": []}
        
        # Get neighbors
        neighbors = list(graph.neighbors(entity_id))
        
        # Get incoming edges too
        predecessors = list(graph.predecessors(entity_id))
        all_connected = set(neighbors + predecessors)
        
        return {
            "entity_id": entity_id,
            "neighbors": [
                node_data.get(n, {"entity_id": n})
                for n in all_connected
            ],
            "edges": [
                edge_data[e]
                for e in edge_data
                if e[0] == entity_id or e[1] == entity_id
            ]
        }





