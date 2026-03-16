"""
DAPPY Entity Resolver (Revised)
Resolves entity mentions to canonical entities BEFORE candidate edge creation.

Key improvements:
1. Uses spaCy KnowledgeBase pattern for O(1) alias lookups
2. BK-Tree for O(log n) fuzzy matching (Phase 1.5 - Quick Win)
3. Incremental KB sync (only new/updated entities)
4. Embedding similarity fallback
5. All thresholds are hyperparameters (not hardcoded!)

This fixes Loophole #2: Candidate Edge Aggregation is Broken
- Entity resolution runs FIRST (before candidate edge creation)
- Store resolved entity IDs in candidate edges

Phase 1A Implementation (Revised)
Phase 1.5: BK-Tree for scalability (100K+ entities/user)
Scalability still an issue, but we can live with it for now.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from arango.database import StandardDatabase

from .schemas import Entity

logger = logging.getLogger(__name__)

# BK-Tree for fast fuzzy matching
try:
    from pybktree import BKTree
    BK_TREE_AVAILABLE = True
except ImportError:
    BK_TREE_AVAILABLE = False
    logger.warning("pybktree not installed. Fuzzy matching will use O(n) fallback.")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance between two strings.
    Used by BK-Tree for fuzzy matching.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class EntityResolver:
    """
    Resolves entity mentions to canonical entities using multiple strategies.
    
    Resolution strategy (ordered by speed):
    1. spaCy KB lookup (O(1) alias lookup)
    2. Fuzzy match using Levenshtein (rapidfuzz)
    3. Embedding similarity (if embeddings available)
    4. Create new entity if no match
    
    All thresholds are configurable hyperparameters.
    """
    
    COLLECTION_NAME = "entities"
    
    def __init__(
        self,
        db: StandardDatabase,
        config: Optional[Dict[str, Any]] = None,
        embedding_service = None
    ):
        """
        Initialize entity resolver.
        
        Args:
            db: ArangoDB database connection
            config: Configuration dict with:
                - fuzzy_threshold: Levenshtein ratio threshold (default: 0.85)
                - embedding_threshold: Cosine similarity threshold (default: 0.90)
                - kb_sync_interval: Seconds between KB syncs (default: 300)
            embedding_service: Optional embedding service for similarity matching
        """
        self.db = db
        self.config = config or {}
        self.embedding_service = embedding_service
        
        # Configurable thresholds (hyperparameters, not hardcoded!)
        entity_config = self.config.get('entity_resolution', {})
        self.fuzzy_threshold = entity_config.get('fuzzy_threshold', 0.85)
        self.embedding_threshold = entity_config.get('embedding_threshold', 0.90)
        self.kb_sync_interval = entity_config.get('kb_sync_interval', 300)
        
        # Initialize collection
        self._init_collection()
        
        # spaCy KB for fast alias lookups (lazy loaded, per-user)
        self._kb_cache = {}  # user_id -> KB
        self._kb_last_sync = {}  # user_id -> timestamp
        
        # BK-Tree for O(log n) fuzzy matching (per-user)
        self._bk_trees = {}  # user_id -> BKTree
        self._bk_tree_last_build = {}  # user_id -> timestamp
        
        logger.info(f"✅ EntityResolver initialized")
        logger.info(f"   Fuzzy threshold: {self.fuzzy_threshold}")
        logger.info(f"   Embedding threshold: {self.embedding_threshold}")
        logger.info(f"   BK-Tree: {'enabled' if BK_TREE_AVAILABLE else 'disabled (using O(n) fallback)'}")
    
    def _init_collection(self):
        """Create collection and indexes if they don't exist."""
        if not self.db.has_collection(self.COLLECTION_NAME):
            self.collection = self.db.create_collection(self.COLLECTION_NAME)
            logger.info(f"Created collection: {self.COLLECTION_NAME}")
        else:
            self.collection = self.db.collection(self.COLLECTION_NAME)
        
        # Create indexes
        try:
            self.collection.add_persistent_index(
                fields=["user_id", "type"],
                name="user_type_idx"
            )
            self.collection.add_persistent_index(
                fields=["canonical_name"],
                name="canonical_name_idx"
            )
            # Alias array index for fast lookup
            self.collection.add_persistent_index(
                fields=["aliases[*]"],
                name="aliases_idx"
            )
            logger.debug("Entity indexes created")
        except Exception as e:
            logger.debug(f"Index creation warning (may exist): {e}")
    
    def _init_kb(self, user_id: str):
        """
        Initialize spaCy KnowledgeBase for a specific user.
        Now per-user instead of global.
        """
        try:
            import spacy
            from spacy.vocab import Vocab
            from spacy.kb import InMemoryLookupKB
            
            # Create a minimal vocab for the KB
            nlp = spacy.blank("en")
            kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=300)
            self._kb_cache[user_id] = kb
            logger.debug(f"spaCy KnowledgeBase initialized for user {user_id}")
            return kb
        except ImportError:
            logger.warning("spaCy not available, KB lookup disabled")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize KB: {e}")
            return None
    
    async def sync_kb(self, user_id: str, incremental: bool = True):
        """
        Sync entities from ArangoDB to spaCy KnowledgeBase.
        Now supports incremental sync for better performance.
        
        Args:
            user_id: User ID to sync entities for
            incremental: If True, only sync new/updated entities
        """
        # Get or create KB for this user
        kb = self._kb_cache.get(user_id)
        if kb is None:
            kb = self._init_kb(user_id)
        
        if kb is None:
            logger.debug("KB not available, skipping sync")
            return
        
        try:
            # Determine which entities to sync
            last_sync = self._kb_last_sync.get(user_id, 0)
            
            if incremental and last_sync > 0:
                # Incremental: only entities updated since last sync
                query = """
                FOR e IN @@collection
                FILTER e.user_id == @user_id
                AND e.updated_at > @since
                RETURN e
                """
                cursor = self.db.aql.execute(
                    query,
                    bind_vars={
                        "@collection": self.COLLECTION_NAME,
                        "user_id": user_id,
                        "since": datetime.fromtimestamp(last_sync).isoformat()
                    }
                )
                entities = [Entity.from_arango_doc(doc) for doc in cursor]
                logger.debug(f"Incremental KB sync: {len(entities)} new/updated entities")
            else:
                # Full sync: all entities (first time or forced)
                entities = await self.get_user_entities(user_id, limit=None)
                logger.debug(f"Full KB sync: {len(entities)} total entities")
            
            # Add entities to KB
            for entity in entities:
                # Add entity to KB with frequency as prior
                try:
                    kb.add_entity(
                        entity=entity.entity_id,
                        freq=entity.stats.get("mention_frequency", 1),
                        entity_vector=entity.embedding if entity.embedding else [0.0] * 300
                    )
                except Exception:
                    pass  # Entity may already exist
                
                # Add canonical name as primary alias
                try:
                    kb.add_alias(
                        alias=entity.canonical_name.lower(),
                        entities=[entity.entity_id],
                        probabilities=[1.0]
                    )
                except Exception:
                    pass  # Alias may already exist
                
                # Add other aliases with slightly lower probability
                for alias in entity.aliases:
                    try:
                        kb.add_alias(
                            alias=alias.lower(),
                            entities=[entity.entity_id],
                            probabilities=[0.95]
                        )
                    except Exception:
                        pass  # Alias may already exist
            
            # Update sync timestamp
            self._kb_last_sync[user_id] = time.time()
            logger.info(f"KB synced for user {user_id}: {len(entities)} entities")
            
        except Exception as e:
            logger.warning(f"KB sync failed: {e}")
    
    async def build_bk_tree(self, user_id: str, incremental: bool = True):
        """
        Build BK-Tree for fast fuzzy matching (O(log n) instead of O(n)).
        
        BK-Tree allows finding all strings within edit distance k in O(log n) time.
        This is much faster than checking every entity for large entity counts.
        
        Args:
            user_id: User ID to build tree for
            incremental: If True, only rebuild if stale
        """
        if not BK_TREE_AVAILABLE:
            logger.debug("BK-Tree not available, skipping")
            return
        
        try:
            # Check if tree needs rebuild
            last_build = self._bk_tree_last_build.get(user_id, 0)
            now = time.time()
            
            if incremental and (now - last_build) < self.kb_sync_interval:
                logger.debug(f"BK-Tree for user {user_id} is fresh, skipping rebuild")
                return
            
            # Get all entities for this user
            entities = await self.get_user_entities(user_id, limit=None)
            
            if not entities:
                logger.debug(f"No entities for user {user_id}, skipping BK-Tree build")
                return
            
            # Build tree with (string, entity) pairs
            tree_data = []
            for entity in entities:
                # Add canonical name
                tree_data.append((entity.canonical_name.lower(), entity))
                
                # Add all aliases
                for alias in entity.aliases:
                    tree_data.append((alias.lower(), entity))
            
            # Create BK-Tree
            self._bk_trees[user_id] = BKTree(levenshtein_distance, tree_data)
            self._bk_tree_last_build[user_id] = now
            
            logger.info(
                f"Built BK-Tree for user {user_id}: "
                f"{len(entities)} entities, {len(tree_data)} total strings"
            )
            
        except Exception as e:
            logger.warning(f"BK-Tree build failed: {e}")
    
    async def resolve(
        self,
        text: str,
        user_id: str,
        context: str = "",
        entity_type: str = None,
        create_if_missing: bool = True
    ) -> Optional[Entity]:
        """
        Resolve a text mention to a canonical entity.
        
        Strategy (ordered by speed):
        1. spaCy KB lookup (O(1) alias lookup)
        2. Exact match on canonical_name (ArangoDB)
        3. Alias match (ArangoDB)
        4. Fuzzy match using Levenshtein (rapidfuzz)
        5. Embedding similarity (if available)
        6. Create new entity if no match
        
        Args:
            text: The entity mention text (e.g., "Sarah", "sarah@acme.com")
            user_id: User scope
            context: Surrounding context for disambiguation
            entity_type: Optional type hint ("person", "organization", etc.)
            create_if_missing: Create new entity if no match found
        
        Returns:
            Resolved Entity or None
        """
        if not text or not text.strip():
            return None
        
        text_normalized = text.strip().lower()
        
        # Check if KB/BK-Tree need sync/rebuild (incremental)
        now = time.time()
        last_sync = self._kb_last_sync.get(user_id, 0)
        
        if (now - last_sync) > self.kb_sync_interval:
            # Incremental sync (only new/updated entities)
            await self.sync_kb(user_id, incremental=True)
            await self.build_bk_tree(user_id, incremental=True)
        
        # Step 1: spaCy KB lookup (fastest - O(1))
        entity = await self._find_by_kb(text_normalized, user_id)
        if entity:
            logger.info(f"🔗 EntityResolver: Resolved '{text}' → {entity.entity_id} (method=KB, canonical={entity.canonical_name})")
            return entity
        
        # Step 2: Exact match on canonical_name (ArangoDB)
        entity = await self._find_by_canonical_name(text_normalized, user_id)
        if entity:
            logger.info(f"🔗 EntityResolver: Resolved '{text}' → {entity.entity_id} (method=Canonical, name={entity.canonical_name})")
            return entity
        
        # Step 3: Alias match (ArangoDB)
        entity = await self._find_by_alias(text_normalized, user_id)
        if entity:
            logger.info(f"🔗 EntityResolver: Resolved '{text}' → {entity.entity_id} (method=Alias, canonical={entity.canonical_name})")
            return entity
        
        # Step 4: Fuzzy match (BK-Tree)
        entity, score = await self._find_by_fuzzy(text_normalized, user_id, entity_type)
        if entity:
            # Add as alias for future lookups
            await self._add_alias(entity, text)
            logger.info(f"🔗 EntityResolver: Resolved '{text}' → {entity.entity_id} (method=Fuzzy, score={score:.2f}, canonical={entity.canonical_name})")
            return entity
        
        # Step 5: Embedding similarity (if available)
        if self.embedding_service:
            entity, score = await self._find_by_embedding(text, user_id, entity_type)
            if entity and score >= self.embedding_threshold:
                await self._add_alias(entity, text)
                logger.info(f"🔗 EntityResolver: Resolved '{text}' → {entity.entity_id} (method=Embedding, score={score:.2f}, canonical={entity.canonical_name})")
                return entity
        
        # Step 6: Create new entity if no match
        if create_if_missing:
            entity = await self._create_entity(
                text=text,
                user_id=user_id,
                entity_type=entity_type or self._infer_type(text, context)
            )
            logger.info(f"🆕 EntityResolver: Created NEW entity '{text}' → {entity.entity_id} (type={entity_type})")
            return entity
        
        return None
    
    async def _find_by_kb(
        self,
        text: str,
        user_id: str
    ) -> Optional[Entity]:
        """Find entity using spaCy KB (O(1) lookup)."""
        kb = self._kb_cache.get(user_id)
        if kb is None:
            return None
        
        try:
            candidates = kb.get_alias_candidates(text)
            if candidates:
                # Get best candidate by prior probability
                best = max(candidates, key=lambda c: c.prior_prob)
                # Fetch full entity from ArangoDB
                return await self.get_entity(best.entity_)
        except Exception as e:
            logger.debug(f"KB lookup failed: {e}")
        
        return None
    
    async def _find_by_canonical_name(
        self,
        name: str,
        user_id: str
    ) -> Optional[Entity]:
        """Find entity by exact canonical name match."""
        query = """
        FOR e IN @@collection
        FILTER e.user_id == @user_id
        AND LOWER(e.canonical_name) == @name
        LIMIT 1
        RETURN e
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "user_id": user_id,
                "name": name.lower()
            }
        )
        
        results = list[Any](cursor)
        if results:
            return Entity.from_arango_doc(results[0])
        return None
    
    async def _find_by_alias(
        self,
        alias: str,
        user_id: str
    ) -> Optional[Entity]:
        """Find entity by alias match."""
        query = """
        FOR e IN @@collection
        FILTER e.user_id == @user_id
        AND @alias IN e.aliases
        LIMIT 1
        RETURN e
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "user_id": user_id,
                "alias": alias.lower()
            }
        )
        
        results = list[Any](cursor)
        if results:
            return Entity.from_arango_doc(results[0])
        return None
    
    async def _find_by_fuzzy(
        self,
        text: str,
        user_id: str,
        entity_type: str = None
    ) -> Tuple[Optional[Entity], float]:
        """
        Fuzzy match using BK-Tree (O(log n)) or fallback to O(n) scan.
        Returns (entity, score) tuple.
        
        Phase 1.5: Uses BK-Tree for 100x speedup on large entity counts.
        """
        # Try BK-Tree first (O(log n))
        if BK_TREE_AVAILABLE and user_id in self._bk_trees:
            return await self._find_by_bk_tree(text, user_id, entity_type)
        
        # Fallback to O(n) scan if BK-Tree not available
        return await self._find_by_fuzzy_fallback(text, user_id, entity_type)
    
    async def _find_by_bk_tree(
        self,
        text: str,
        user_id: str,
        entity_type: str = None
    ) -> Tuple[Optional[Entity], float]:
        """
        BK-Tree fuzzy matching (O(log n)).
        
        Finds all strings within edit distance threshold in logarithmic time.
        """
        tree = self._bk_trees.get(user_id)
        if tree is None:
            return None, 0.0
        
        try:
            # Calculate max edit distance from threshold
            # threshold=0.85 means we allow 15% edits
            max_distance = int((1 - self.fuzzy_threshold) * len(text))
            max_distance = max(1, max_distance)  # At least 1 edit allowed
            
            # Find all matches within edit distance
            matches = tree.find(text, max_distance)
            
            if not matches:
                return None, 0.0
            
            # Filter by entity type if specified
            if entity_type:
                matches = [(dist, (string, entity)) for dist, (string, entity) in matches 
                          if entity.type == entity_type]
            
            if not matches:
                return None, 0.0
            
            # Get best match (lowest edit distance)
            best_distance, (best_string, best_entity) = min(matches, key=lambda m: m[0])
            
            # Convert edit distance to similarity score
            # score = 1 - (distance / max_length)
            score = 1.0 - (best_distance / max(len(text), len(best_string)))
            
            logger.debug(
                f"BK-Tree match: '{text}' → '{best_string}' "
                f"(distance={best_distance}, score={score:.2f})"
            )
            
            return best_entity, score
            
        except Exception as e:
            logger.warning(f"BK-Tree lookup failed: {e}")
            return None, 0.0
    
    async def _find_by_fuzzy_fallback(
        self,
        text: str,
        user_id: str,
        entity_type: str = None
    ) -> Tuple[Optional[Entity], float]:
        """
        Fallback O(n) fuzzy matching using rapidfuzz.
        Used when BK-Tree is not available.
        
        NOTE: This is slow for >1000 entities. Use BK-Tree for production.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("rapidfuzz not installed, fuzzy matching disabled")
            return None, 0.0
        
        # Limit to 1000 entities for O(n) scan (was 500)
        candidates = await self.get_user_entities(user_id, entity_type=entity_type, limit=1000)
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            # Check canonical name
            score = fuzz.ratio(text, candidate.canonical_name.lower()) / 100
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = candidate
            
            # Check aliases
            for alias in candidate.aliases:
                score = fuzz.ratio(text, alias.lower()) / 100
                if score > best_score and score >= self.fuzzy_threshold:
                    best_score = score
                    best_match = candidate
        
        # Log near-misses for threshold tuning
        if best_match is None and best_score > 0.6:
            logger.debug(
                f"Fuzzy near-miss: '{text}' best={best_score:.2f} < threshold={self.fuzzy_threshold}"
            )
        
        return best_match, best_score
    
    async def _find_by_embedding(
        self,
        text: str,
        user_id: str,
        entity_type: str = None
    ) -> Tuple[Optional[Entity], float]:
        """
        Find entity by embedding similarity.
        Returns (entity, score) tuple.
        """
        if not self.embedding_service:
            return None, 0.0
        
        try:
            # Generate embedding for the text
            text_embedding = await self.embedding_service.generate(text)
            
            # Query entities with embeddings
            type_filter = f"AND e.type == '{entity_type}'" if entity_type else ""
            query = f"""
            FOR e IN @@collection
            FILTER e.user_id == @user_id
            AND LENGTH(e.embedding) > 0
            {type_filter}
            LET similarity = (
                SUM(
                    FOR i IN 0..LENGTH(e.embedding)-1
                    RETURN e.embedding[i] * @query_embedding[i]
                ) / (
                    SQRT(SUM(FOR x IN e.embedding RETURN x * x)) *
                    SQRT(SUM(FOR x IN @query_embedding RETURN x * x))
                )
            )
            FILTER similarity >= @threshold
            SORT similarity DESC
            LIMIT 1
            RETURN {{entity: e, similarity: similarity}}
            """
            
            cursor = self.db.aql.execute(
                query,
                bind_vars={
                    "@collection": self.COLLECTION_NAME,
                    "user_id": user_id,
                    "query_embedding": text_embedding,
                    "threshold": self.embedding_threshold * 0.9  # Slightly lower for candidates
                }
            )
            
            results = list[Any](cursor)
            if results:
                entity = Entity.from_arango_doc(results[0]["entity"])
                similarity = results[0]["similarity"]
                return entity, similarity
            
        except Exception as e:
            logger.warning(f"Embedding-based entity resolution failed: {e}")
        
        return None, 0.0
    
    async def _add_alias(self, entity: Entity, alias: str):
        """Add an alias to an entity for future lookups."""
        alias_lower = alias.strip().lower()
        if alias_lower not in entity.aliases:
            entity.add_alias(alias_lower)
            self.collection.update(entity.to_arango_doc())
            
            # Update KB if available
            kb = self._kb_cache.get(user_id)
            if kb is not None:
                try:
                    kb.add_alias(
                        alias=alias_lower,
                        entities=[entity.entity_id],
                        probabilities=[0.9]
                    )
                except Exception:
                    pass
    
    async def _create_entity(
        self,
        text: str,
        user_id: str,
        entity_type: str = "unknown"
    ) -> Entity:
        """Create a new entity."""
        # Generate embedding if service available
        embedding = []
        if self.embedding_service:
            try:
                embedding = await self.embedding_service.generate(text)
            except Exception as e:
                logger.warning(f"Failed to generate entity embedding: {e}")
        
        entity = Entity(
            canonical_name=text,
            type=entity_type,
            aliases=[text.lower()],
            embedding=embedding,
            user_id=user_id
        )
        
        self.collection.insert(entity.to_arango_doc())
        
        # Add to KB if available
        kb = self._kb_cache.get(user_id)
        if kb is not None:
            try:
                kb.add_entity(
                    entity=entity.entity_id,
                    freq=1,
                    entity_vector=embedding if embedding else [0.0] * 300
                )
                kb.add_alias(
                    alias=text.lower(),
                    entities=[entity.entity_id],
                    probabilities=[1.0]
                )
            except Exception:
                pass
        
        return entity
    
    def _infer_type(self, text: str, context: str = "") -> str:
        """
        Infer entity type from text and context.
        Simple heuristics - enhanced by EntityExtractor when used together.
        """
        text_lower = text.lower()
        context_lower = context.lower()
        
        # Organization indicators
        org_indicators = ["corp", "inc", "llc", "ltd", "company", "org", "foundation", "university"]
        if any(ind in text_lower for ind in org_indicators):
            return "organization"
        
        # Location indicators
        loc_indicators = ["city", "country", "street", "avenue", "road", "state", "county"]
        if any(ind in text_lower for ind in loc_indicators):
            return "location"
        
        # Check context for hints
        if "works at" in context_lower or "employee" in context_lower:
            if "works at" in context_lower:
                parts = context_lower.split("works at")
                if len(parts) > 1 and text_lower in parts[-1]:
                    return "organization"
            return "person"
        
        if any(rel in context_lower for rel in ["my sister", "my brother", "my mom", "my dad", "my friend"]):
            return "person"
        
        # Default to person (most common in personal memory)
        return "person"
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        try:
            doc = self.collection.get(entity_id)
            if doc:
                return Entity.from_arango_doc(doc)
        except Exception:
            pass
        return None
    
    async def link_memory_to_entity(
        self,
        entity_id: str,
        memory_id: str
    ):
        """Link a memory to an entity."""
        entity = await self.get_entity(entity_id)
        if entity:
            entity.link_memory(memory_id)
            self.collection.update(entity.to_arango_doc())
    
    async def merge_entities(
        self,
        source_id: str,
        target_id: str
    ) -> Entity:
        """
        Merge source entity into target entity.
        Used when duplicate entities are detected.
        """
        source = await self.get_entity(source_id)
        target = await self.get_entity(target_id)
        
        if not source or not target:
            raise ValueError("Source or target entity not found")
        
        # Merge aliases
        for alias in source.aliases:
            target.add_alias(alias)
        
        # Merge linked memories
        for mem_id in source.linked_memories:
            target.link_memory(mem_id)
        
        # Update stats
        target.stats["mention_frequency"] = len(target.linked_memories)
        
        # Save target
        self.collection.update(target.to_arango_doc())
        
        # Delete source
        self.collection.delete(source_id)
        
        logger.info(f"Merged entity {source_id} into {target_id}")
        return target
    
    async def get_user_entities(
        self,
        user_id: str,
        entity_type: str = None,
        limit: Optional[int] = 100
    ) -> List[Entity]:
        """
        Get entities for a user.
        
        Args:
            user_id: User ID
            entity_type: Optional filter by entity type
            limit: Max entities to return (None = all entities)
        
        Returns:
            List of Entity objects
        """
        type_filter = f"AND e.type == '{entity_type}'" if entity_type else ""
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        
        query = f"""
        FOR e IN @@collection
        FILTER e.user_id == @user_id
        {type_filter}
        SORT e.stats.mention_frequency DESC
        {limit_clause}
        RETURN e
        """
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                "@collection": self.COLLECTION_NAME,
                "user_id": user_id
            }
        )
        
        return [Entity.from_arango_doc(doc) for doc in cursor]
