"""
Knowledge Graph Store
Stores entity-entity relations (knowledge graph) for structured context in retrieval.
Relations link entities to memories via memory_id field.
"""

import logging
from typing import List, Set, Optional, Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

ENTITY_RELATIONS_COLLECTION = "entity_relations"


class KnowledgeGraphStore:
    """
    Manages the user's knowledge graph (entity-entity relations).
    
    Collection:
    - entity_relations: {user_id, subject, predicate, object, memory_id, confidence}
    
    Relations serve dual purpose:
    1. Provide structured context to LLM during retrieval
    2. Link entities to memories (via memory_id field)
    """

    def __init__(self, db, config: Optional[dict] = None):
        self.db = db
        self.config = config or {}
        self._relations_collection = None
        if db:
            self._init_collection()

    def _init_collection(self):
        """Create entity_relations collection if it doesn't exist."""
        try:
            logger.debug(f"Initializing KnowledgeGraphStore with db: {self.db}")
            
            if not self.db.has_collection(ENTITY_RELATIONS_COLLECTION):
                self.db.create_collection(ENTITY_RELATIONS_COLLECTION)
                logger.info(f"Created collection: {ENTITY_RELATIONS_COLLECTION}")
            else:
                logger.debug(f"Collection {ENTITY_RELATIONS_COLLECTION} already exists")
            
            self._relations_collection = self.db.collection(ENTITY_RELATIONS_COLLECTION)
            logger.debug(f"Got collection reference: {self._relations_collection}, type: {type(self._relations_collection)}")
            
            try:
                self._relations_collection.add_persistent_index(
                    fields=["user_id", "subject"],
                    name="user_subject_idx"
                )
                self._relations_collection.add_persistent_index(
                    fields=["user_id", "object"],
                    name="user_object_idx"
                )
                self._relations_collection.add_persistent_index(
                    fields=["user_id", "memory_id"],
                    name="user_memory_idx"
                )
            except Exception as idx_err:
                logger.debug(f"Index creation (may exist): {idx_err}")
            
            logger.info(f"KnowledgeGraphStore ready: {ENTITY_RELATIONS_COLLECTION}")
            logger.debug(f"Collection object set: {self._relations_collection is not None}")
        except Exception as e:
            logger.error(f"KnowledgeGraphStore init failed: {e}", exc_info=True)
            self._relations_collection = None

    def store_relations(
        self,
        user_id: str,
        memory_id: str,
        relations: List[Dict[str, Any]]
    ) -> int:
        """
        Store entity-entity relations for a memory.

        Args:
            user_id: User ID
            memory_id: Memory node ID
            relations: List of {"subject": str, "predicate": str, "object": str, "confidence": float}

        Returns:
            Number of relations stored
        """
        if self._relations_collection is None:
            logger.warning("KnowledgeGraphStore: _relations_collection is None, attempting re-init")
            try:
                self._init_collection()
            except Exception as e:
                logger.error(f"Re-init failed: {e}")
                return 0
            
            if self._relations_collection is None:
                logger.error("KnowledgeGraphStore: Re-init failed, collection still None")
                return 0
        if not relations:
            return 0

        count = 0
        
        
        for rel in relations:
            subject = rel.get("subject", "").lower().strip()
            predicate = rel.get("predicate", "").lower().strip()
            obj = rel.get("object", "").lower().strip()
            confidence = rel.get("confidence", 0.5)

            if not subject or not predicate or not obj:
                logger.debug(f"Skipping incomplete relation: {rel}")
                continue

            try:
                # Check if relation already exists (deduplication)
                existing_query = """
                FOR rel IN entity_relations
                    FILTER rel.user_id == @user_id
                    FILTER rel.subject == @subject
                    FILTER rel.predicate == @predicate
                    FILTER rel.object == @object
                    LIMIT 1
                    RETURN rel
                """
                
                cursor = self.db.aql.execute(
                    existing_query,
                    bind_vars={
                        "user_id": user_id,
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj
                    }
                )
                existing = list(cursor)
                
                if existing:
                    # Relation exists - reinforce it (supporting_mentions++)
                    existing_rel = existing[0]
                    update_query = """
                    FOR rel IN entity_relations
                        FILTER rel._key == @key
                        UPDATE rel WITH {
                            supporting_mentions: (rel.supporting_mentions || 0) + 1,
                            confidence: MIN([rel.confidence + 0.05, 1.0]),
                            last_mentioned: @last_mentioned,
                            memory_id: @memory_id
                        } IN entity_relations
                        RETURN NEW
                    """
                    
                    self.db.aql.execute(
                        update_query,
                        bind_vars={
                            "key": existing_rel["_key"],
                            "last_mentioned": datetime.now().isoformat(),
                            "memory_id": memory_id
                        }
                    )
                    count += 1
                    logger.debug(f"Reinforced relation: {subject} --{predicate}--> {obj} (mentions: {existing_rel.get('supporting_mentions', 0) + 1})")
                else:
                    # New relation - insert
                    now = datetime.now().isoformat()
                    self._relations_collection.insert({
                        "user_id": user_id,
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "memory_id": memory_id,
                        "confidence": float(confidence),
                        "supporting_mentions": 0,
                        "created_at": now,
                        "last_mentioned": now,
                        "effective_from": now,
                        "effective_to": None
                    })
                    count += 1
                    logger.debug(f"Stored new relation: {subject} --{predicate}--> {obj} (confidence: {confidence:.2f})")
            except Exception as e:
                logger.warning(f"Relation upsert failed for '{subject} --{predicate}--> {obj}': {e}")

        return count

    def get_memories_for_entities(
        self,
        user_id: str,
        entity_names: List[str],
        limit: int = 20
    ) -> List[str]:
        """
        Get memory IDs for entities by querying the knowledge graph.
        Finds memories where the entity appears as subject or object in any relation.

        Args:
            user_id: User ID
            entity_names: Entity names to search for
            limit: Max memory IDs to return

        Returns:
            List of memory_id (distinct)
        """
        if not self._relations_collection or not entity_names:
            return []

        names_lower = [str(n).lower().strip() for n in entity_names if n]
        if not names_lower:
            return []

        try:
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.subject IN @names OR rel.object IN @names
                LIMIT @limit
                RETURN DISTINCT rel.memory_id
            """
            cursor = self.db.aql.execute(
                query,
                bind_vars={
                    "user_id": user_id,
                    "names": names_lower,
                    "limit": limit
                }
            )
            return list(cursor)
        except Exception as e:
            logger.warning(f"Entity-to-memory lookup via KG failed: {e}")
            return []

    def get_user_relations(
        self,
        user_id: str,
        entity_names: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get relations for a user, optionally filtered by entity names.
        Used to provide relation context to LLM during retrieval.
        
        Args:
            user_id: User ID
            entity_names: Optional list of entity names to filter by
            limit: Max relations to return
        
        Returns:
            List of {"subject": str, "predicate": str, "object": str, "confidence": float, "memory_id": str,
                     "created_at": str, "last_mentioned": str, "supporting_mentions": int,
                     "effective_from": str, "effective_to": str}
        """
        if not self._relations_collection:
            return []

        try:
            if entity_names:
                names_lower = [str(n).lower().strip() for n in entity_names if n]
                query = """
                FOR rel IN entity_relations
                    FILTER rel.user_id == @user_id
                    FILTER rel.subject IN @names OR rel.object IN @names
                    FILTER rel.effective_to == null
                    SORT rel.confidence DESC
                    LIMIT @limit
                    RETURN {
                        subject: rel.subject,
                        predicate: rel.predicate,
                        object: rel.object,
                        confidence: rel.confidence,
                        memory_id: rel.memory_id,
                        created_at: rel.created_at,
                        last_mentioned: rel.last_mentioned,
                        supporting_mentions: rel.supporting_mentions,
                        effective_from: rel.effective_from,
                        effective_to: rel.effective_to
                    }
                """
                bind_vars = {"user_id": user_id, "names": names_lower, "limit": limit}
            else:
                query = """
                FOR rel IN entity_relations
                    FILTER rel.user_id == @user_id
                    FILTER rel.effective_to == null
                    SORT rel.confidence DESC
                    LIMIT @limit
                    RETURN {
                        subject: rel.subject,
                        predicate: rel.predicate,
                        object: rel.object,
                        confidence: rel.confidence,
                        memory_id: rel.memory_id,
                        created_at: rel.created_at,
                        last_mentioned: rel.last_mentioned,
                        supporting_mentions: rel.supporting_mentions,
                        effective_from: rel.effective_from,
                        effective_to: rel.effective_to
                    }
                """
                bind_vars = {"user_id": user_id, "limit": limit}
            
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            return list(cursor)
        except Exception as e:
            logger.warning(f"Failed to get user relations: {e}")
            return []

