"""
DAPPY ArangoDB Integration Module

Provides the connection layer between:
1. ArangoDB (canonical memory store + graph)
2. Chatbot Service (memory extraction)
3. Graph-of-Thoughts components

This module handles:
- ArangoDB connection management
- Graph collection initialization
- Integration with existing memory pipeline

Phase 1C: Essential Integration
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from arango import ArangoClient
    from arango.database import StandardDatabase
    ARANGO_AVAILABLE = True
except ImportError:
    ARANGO_AVAILABLE = False
    ArangoClient = None
    StandardDatabase = None

from .schemas import Entity, ThoughtEdge, CandidateEdge
from .edge_store import CandidateEdgeStore, ThoughtEdgeStore
from .entity_resolver import EntityResolver
from .entity_extractor import EntityExtractor
from .relation_classifier import RelationClassifier
from .relation_extractor import RelationExtractor
from .activation_scorer import ActivationScorer
from .ppr_retrieval import PPRRetrieval

logger = logging.getLogger(__name__)


class ArangoGraphManager:
    """
    Manages ArangoDB connection and graph collections.
    
    Collections managed:
    - entities: Canonical entities (nodes)
    - thought_edges: Promoted edges (Tier 1-3)
    - candidate_edges: Candidate edges (Tier 4)
    
    Graphs:
    - thought_graph: The main Graph-of-Thoughts
    """
    
    # Collection names
    ENTITIES_COLLECTION = "entities"
    THOUGHT_EDGES_COLLECTION = "thought_edges"
    CANDIDATE_EDGES_COLLECTION = "candidate_edges"
    GRAPH_NAME = "thought_graph"
    
    def __init__(
        self,
        config: Dict[str, Any],
        db: Optional[StandardDatabase] = None
    ):
        """
        Initialize ArangoDB graph manager.
        
        Args:
            config: Configuration dict with arangodb section
            db: Optional existing database connection
        """
        self.config = config
        self.arango_config = config.get('arangodb', {})
        
        if not ARANGO_AVAILABLE:
            logger.warning("python-arango not installed. Graph features disabled.")
            self.db = None
            self.enabled = False
            return
        
        # Use provided db or create new connection
        if db:
            self.db = db
        else:
            self.db = self._connect()
        
        self.enabled = self.db is not None
        
        if self.enabled:
            self._init_collections()
            self._init_graph()
            logger.info("✅ ArangoGraphManager initialized")
    
    def _connect(self) -> Optional[StandardDatabase]:
        """Connect to ArangoDB."""
        try:
            url = self.arango_config.get('url', 'http://localhost:8529')
            username = self.arango_config.get('username', 'root')
            password = self.arango_config.get('password', '')
            database = self.arango_config.get('database', 'dappy_memories')
            
            client = ArangoClient(hosts=url)
            sys_db = client.db('_system', username=username, password=password)
            
            # Create database if not exists
            if not sys_db.has_database(database):
                sys_db.create_database(database)
                logger.info(f"Created database: {database}")
            
            db = client.db(database, username=username, password=password)
            logger.info(f"Connected to ArangoDB: {url}/{database}")
            return db
            
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            return None
    
    def _init_collections(self):
        """Initialize required collections."""
        if not self.db:
            return
        
        # Document collections
        for collection_name in [self.ENTITIES_COLLECTION]:
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
        
        # Edge collections
        for collection_name in [self.THOUGHT_EDGES_COLLECTION, self.CANDIDATE_EDGES_COLLECTION]:
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name, edge=True)
                logger.info(f"Created edge collection: {collection_name}")
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for efficient queries."""
        if not self.db:
            return
        
        try:
            # Entity indexes
            entities = self.db.collection(self.ENTITIES_COLLECTION)
            entities.add_persistent_index(fields=["user_id"], name="user_idx")
            entities.add_persistent_index(fields=["canonical_name"], name="name_idx")
            entities.add_persistent_index(fields=["type"], name="type_idx")
            
            # Thought edge indexes
            thought_edges = self.db.collection(self.THOUGHT_EDGES_COLLECTION)
            thought_edges.add_persistent_index(fields=["user_id"], name="user_idx")
            thought_edges.add_persistent_index(fields=["relation_category"], name="category_idx")
            thought_edges.add_persistent_index(fields=["effective_from", "effective_to"], name="temporal_idx")
            
            # Candidate edge indexes
            candidate_edges = self.db.collection(self.CANDIDATE_EDGES_COLLECTION)
            candidate_edges.add_persistent_index(fields=["user_id", "status"], name="user_status_idx")
            candidate_edges.add_persistent_index(fields=["activation"], name="activation_idx")
            
            logger.debug("Created graph indexes")
            
        except Exception as e:
            logger.debug(f"Index creation (may already exist): {e}")
    
    def _init_graph(self):
        """Initialize the thought_graph."""
        if not self.db:
            return
        
        try:
            if not self.db.has_graph(self.GRAPH_NAME):
                self.db.create_graph(
                    self.GRAPH_NAME,
                    edge_definitions=[
                        {
                            "edge_collection": self.THOUGHT_EDGES_COLLECTION,
                            "from_vertex_collections": [self.ENTITIES_COLLECTION],
                            "to_vertex_collections": [self.ENTITIES_COLLECTION]
                        }
                    ]
                )
                logger.info(f"Created graph: {self.GRAPH_NAME}")
            else:
                logger.debug(f"Graph already exists: {self.GRAPH_NAME}")
                
        except Exception as e:
            logger.error(f"Failed to create graph: {e}")
    
    def get_database(self) -> Optional[StandardDatabase]:
        """Get the database connection."""
        return self.db
    
    def is_enabled(self) -> bool:
        """Check if graph features are enabled."""
        return self.enabled


class GraphPipeline:
    """
    Unified pipeline for Graph-of-Thoughts processing.
    
    Orchestrates:
    1. Entity extraction (spaCy)
    2. Entity resolution (fuzzy + embedding)
    3. Relation classification (DeBERTa + LLM fallback)
    4. Candidate edge creation
    5. Activation scoring
    6. PPR retrieval
    
    This is the main entry point for chatbot service integration.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        db: Optional[StandardDatabase] = None,
        embedding_service = None,
        llm_provider = None,
    ):
        """
        Initialize the graph pipeline.
        
        Args:
            config: Full configuration dict
            db: ArangoDB database connection
            embedding_service: Embedding service for entity resolution
        """
        self.config = config
        self.embedding_service = embedding_service
        self.llm_provider = llm_provider
        
        # Initialize ArangoDB manager
        self.arango_manager = ArangoGraphManager(config, db)
        self.db = self.arango_manager.get_database()
        self.enabled = self.arango_manager.is_enabled()
        
        if not self.enabled:
            logger.warning("Graph pipeline disabled (no ArangoDB connection)")
            return
        
        # Initialize components
        self._init_components()
        
        logger.info("✅ GraphPipeline initialized")
    
    def _init_components(self):
        """Initialize all graph components."""
        # Entity extraction
        self.entity_extractor = EntityExtractor(
            config=self.config
        )
        
        # Entity resolution
        self.entity_resolver = EntityResolver(
            db=self.db,
            embedding_service=self.embedding_service,
            config=self.config
        )
        
        # Relation classification
        self.relation_classifier = RelationClassifier(config=self.config)
        
        # Relation extraction (orchestrates entity + relation)
        self.relation_extractor = RelationExtractor(
            db=self.db,
            config=self.config,
            embedding_service=self.embedding_service,
            llm_provider=self.llm_provider,
        )
        
        # Edge stores
        self.candidate_edge_store = CandidateEdgeStore(db=self.db)
        self.thought_edge_store = ThoughtEdgeStore(db=self.db)
        
        # Activation scoring
        self.activation_scorer = ActivationScorer(
            config=self.config,
            db=self.db,
            embedding_service=self.embedding_service
        )
        
        # PPR retrieval
        self.ppr_retrieval = PPRRetrieval(
            db=self.db,
            config=self.config
        )
    
    async def process_memory(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        ego_score: float,
        tier: int,
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a memory through the graph pipeline.
        
        This is called by chatbot_service after memory scoring.
        
        Args:
            user_id: User ID
            memory_id: Memory ID (from Qdrant/ArangoDB)
            content: Memory content text
            ego_score: Ego score from ML scorer
            tier: Memory tier (1-4)
            session_id: Optional session ID
            metadata: Optional additional metadata
        
        Returns:
            Dict with extracted entities, relations, and edges
        """
        if not self.enabled:
            return {"enabled": False}
        
        result = {
            "memory_id": memory_id,
            "entities": [],
            "relations": [],
            "candidate_edges": [],
            "errors": []
        }
        
        try:
            # Step 1: Extract relations (includes entity extraction + resolution)
            relations = await self.relation_extractor.extract(
                text=content,
                user_id=user_id,
                memory_id=memory_id,
                ego_score=ego_score,
                session_id=session_id,
                metadata=metadata  # Pass metadata down
            )
            
            result["relations"] = [r.to_dict() for r in relations]
            
            # Step 2: Create candidate edges
            for relation in relations:
                candidate = relation.to_candidate_edge(
                    user_id=user_id,
                    ego_score=ego_score
                )
                
                # Store candidate edge
                stored = await self.candidate_edge_store.create_or_update(candidate)
                result["candidate_edges"].append({
                    "candidate_id": stored.candidate_id,
                    "predicate": stored.predicate,
                    "subject": relation.subject_text,
                    "object": relation.object_text
                })
            
            # Step 3: Score candidates for potential promotion
            # (Only for high-tier memories to avoid noise)
            if tier <= 2 and result["candidate_edges"]:
                logger.info(f"🎲 GraphPipeline: Tier {tier} detected, triggering activation scoring for {len(result['candidate_edges'])} edges")
                await self._score_and_maybe_promote(
                    user_id=user_id,
                    candidate_ids=[ce["candidate_id"] for ce in result["candidate_edges"]]
                )
            elif tier > 2:
                logger.info(f"⏭️  GraphPipeline: Tier {tier} - skipping activation scoring (low priority)")
            
            logger.info(f"✅ GraphPipeline: Processed memory {memory_id}")
            logger.info(f"   → Extracted {len(relations)} relations")
            logger.info(f"   → Created {len(result['candidate_edges'])} candidate edges")
            logger.info(f"   → Tier {tier} (ego={ego_score:.2f})")
            
            # Step 4: Write back graph extraction metadata to memory (Phase 1F)
            await self._update_memory_with_graph_data(
                memory_id=memory_id,
                relations=relations,
                candidate_edges=result["candidate_edges"]
            )
            
        except Exception as e:
            logger.error(f"Graph pipeline error: {e}", exc_info=True)
            result["errors"].append(str(e))
        
        return result
    
    async def _update_memory_with_graph_data(
        self,
        memory_id: str,
        relations: List,
        candidate_edges: List[Dict[str, Any]]
    ):
        """
        Update memory node with bidirectional links to graph elements (Phase 1F).
        
        This enables:
        - Memory → Entities lookup
        - Memory → Edges lookup
        - Provenance tracking
        - Context verification during PPR retrieval
        """
        try:
            # Build graph extraction metadata
            extracted_entities = []
            entity_ids_seen = set()
            
            for relation in relations:
                # Add subject entity (avoid duplicates)
                if relation.subject_entity_id not in entity_ids_seen:
                    extracted_entities.append({
                        "entity_id": relation.subject_entity_id,
                        "text": relation.subject_text,
                        "type": getattr(relation, 'subject_type', None)
                    })
                    entity_ids_seen.add(relation.subject_entity_id)
                
                # Add object entity (avoid duplicates)
                if relation.object_entity_id not in entity_ids_seen:
                    extracted_entities.append({
                        "entity_id": relation.object_entity_id,
                        "text": relation.object_text,
                        "type": getattr(relation, 'object_type', None)
                    })
                    entity_ids_seen.add(relation.object_entity_id)
            
            extracted_edges = [
                {
                    "edge_id": edge["candidate_id"],
                    "relation": edge["predicate"],
                    "subject": edge["subject"],
                    "object": edge["object"]
                }
                for edge in candidate_edges
            ]
            
            graph_extraction = {
                "extracted_entities": extracted_entities,
                "extracted_edges": extracted_edges,
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "entity_count": len(extracted_entities),
                "edge_count": len(extracted_edges)
            }
            
            # Update memory node in ArangoDB
            memories_collection = self.db.collection("memories")
            memories_collection.update({
                "_key": memory_id,
                "graph_extraction": graph_extraction
            })
            
            logger.info(f"🔗 Updated memory {memory_id} with graph metadata:")
            logger.info(f"   → {len(extracted_entities)} entities linked")
            logger.info(f"   → {len(extracted_edges)} edges linked")
            
        except Exception as e:
            # Non-fatal error - log but don't fail the pipeline
            logger.warning(f"⚠️  Failed to update memory with graph data: {e}")
    
    async def _score_and_maybe_promote(
        self,
        user_id: str,
        candidate_ids: List[str]
    ):
        """Score candidates and promote if above threshold."""
        logger.info(f"🎯 Scoring {len(candidate_ids)} candidate edges for promotion...")
        
        for candidate_id in candidate_ids:
            try:
                # Get candidate
                logger.debug(f"Fetching candidate edge: {candidate_id}")
                candidate = await self.candidate_edge_store.get(candidate_id)
                if not candidate:
                    logger.warning(f"⚠️  Candidate edge {candidate_id} not found in store")
                    continue
                
                # Score
                logger.debug(f"Scoring candidate: {candidate_id}")
                result = await self.activation_scorer.score(candidate)
                
                # Update candidate with score
                await self.candidate_edge_store.update_activation(
                    candidate_id=candidate_id,
                    activation=result.activation_score,
                    status=result.decision  # "promote", "keep", or "demote"
                )
                
                # Promote if above threshold
                if result.decision == "promote":
                    logger.info(f"✅ Decision: PROMOTE edge {candidate_id}")
                    await self._promote_candidate(candidate, result)
                else:
                    logger.info(f"⏸️  Decision: {result.decision.upper()} edge {candidate_id} (score={result.activation_score:.3f})")
                    
            except Exception as e:
                logger.error(f"❌ Scoring failed for {candidate_id}: {e}", exc_info=True)
    
    async def _promote_candidate(
        self,
        candidate: CandidateEdge,
        activation_result
    ):
        """Promote a candidate edge to the canonical KG."""
        try:
            # Create ThoughtEdge from candidate
            thought_edge = ThoughtEdge(
                user_id=candidate.user_id,
                predicate=candidate.predicate,
                relation_category=self._infer_category(candidate.predicate),
                strength=activation_result.activation_score,
                effective_from=candidate.first_seen,
                is_bidirectional=self._is_symmetric(candidate.predicate),
                supporting_memories=[m.mem_id for m in candidate.supporting_mentions]
            )
            
            # Store in thought_edges
            await self.thought_edge_store.create(
                thought_edge,
                from_entity_id=candidate.subject_entity_id,
                to_entity_id=candidate.object_entity_id
            )
            
            # Update candidate status
            await self.candidate_edge_store.update_status(
                candidate.candidate_id, 
                "promoted"
            )
            
            logger.info(f"Promoted edge: {candidate.predicate}")
            
        except Exception as e:
            logger.error(f"Failed to promote candidate: {e}")
    
    def _infer_category(self, predicate: str) -> str:
        """Infer relation category from predicate."""
        family_relations = ["sister_of", "brother_of", "parent_of", "child_of", "spouse_of", "family_of"]
        professional_relations = ["works_at", "works_with", "colleague_of", "manages", "employed_by"]
        temporal_relations = ["contradicts", "supersedes", "evolves_to", "replaces"]
        
        if predicate in family_relations:
            return "family"
        elif predicate in professional_relations:
            return "professional"
        elif predicate in temporal_relations:
            return "temporal"
        else:
            return "general"
    
    def _is_symmetric(self, predicate: str) -> bool:
        """Check if relation is symmetric (bidirectional)."""
        symmetric_relations = [
            "friend_of", "knows", "colleague_of", "works_with",
            "sibling_of", "married_to", "partner_of"
        ]
        return predicate in symmetric_relations
    
    async def retrieve_context(
        self,
        user_id: str,
        query_text: str,
        relation_category: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve graph context for a query.
        
        Uses PPR to find relevant entities and relationships.
        
        Args:
            user_id: User ID
            query_text: Query text
            relation_category: Optional filter by category
            limit: Max results
        
        Returns:
            List of context items
        """
        if not self.enabled:
            return []
        
        try:
            # Extract entities from query
            entities = self.entity_extractor.extract(query_text)
            
            if not entities:
                return []
            
            # Resolve entities to get IDs
            entity_ids = []
            for entity in entities:
                resolved = await self.entity_resolver.resolve(
                    text=entity.text,
                    user_id=user_id,
                    create_if_missing=False
                )
                if resolved:
                    entity_ids.append(resolved.entity_id)
            
            if not entity_ids:
                return []
            
            # Use PPR to retrieve context
            context = await self.ppr_retrieval.retrieve_context(
                user_id=user_id,
                query_entities=entity_ids,
                relation_category=relation_category,
                limit=limit
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            return {
                "enabled": True,
                "entities_count": self.db.collection("entities").count(),
                "thought_edges_count": self.db.collection("thought_edges").count(),
                "candidate_edges_count": self.db.collection("candidate_edges").count()
            }
        except:
            return {"enabled": True, "stats": "unavailable"}


# Factory function for easy initialization
def create_graph_pipeline(
    config: Dict[str, Any],
    embedding_service = None,
    llm_provider = None,
) -> GraphPipeline:
    """
    Create a GraphPipeline instance.
    
    Usage:
        from core.graph.arango_integration import create_graph_pipeline
        
        pipeline = create_graph_pipeline(config, embedding_service)
        
        # In chatbot_service._extract_and_score_memories:
        result = await pipeline.process_memory(
            user_id=user_id,
            memory_id=node_id,
            content=user_message,
            ego_score=ego_result.ego_score,
            tier=ego_result.tier
        )
    """
    return GraphPipeline(config, embedding_service=embedding_service, llm_provider=llm_provider)



