"""
KG Re-consolidation Service

Periodic background process that maintains KG health by:
1. Merging duplicate relations
2. Re-processing high-tier memories without relations
3. Decaying stale relations
4. (Future) Promoting high-confidence relations to thought_graph
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class KGReconsolidationService:
    """
    Background service for periodic KG cleanup and enhancement.
    """

    def __init__(self, knowledge_graph_store, consolidation_service, arango_db):
        self.kg_store = knowledge_graph_store
        self.consolidation_service = consolidation_service
        self.arango_db = arango_db

    async def merge_duplicate_relations(self, user_id: str) -> Dict[str, int]:
        """
        Find and merge duplicate relations for a user.
        
        Duplicates are relations with same (user_id, subject, predicate, object).
        Merge strategy: Keep highest confidence, sum supporting_mentions, use earliest created_at.
        
        Returns:
            {"duplicates_found": int, "relations_merged": int}
        """
        logger.info(f"🔄 Merging duplicate relations for user {user_id}")
        
        try:
            # Find duplicate groups
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.effective_to == null
                COLLECT 
                    subject = rel.subject,
                    predicate = rel.predicate,
                    object = rel.object
                INTO group
                LET count = LENGTH(group)
                LET relations = group[*].rel
                FILTER count > 1
                RETURN {
                    subject: subject,
                    predicate: predicate,
                    object: object,
                    count: count,
                    relations: relations
                }
            """
            
            cursor = self.kg_store.db.aql.execute(query, bind_vars={"user_id": user_id})
            duplicate_groups = list(cursor)
            
            if not duplicate_groups:
                logger.info(f"   ✅ No duplicate relations found")
                return {"duplicates_found": 0, "relations_merged": 0}
            
            logger.info(f"   📊 Found {len(duplicate_groups)} duplicate groups")
            
            merged_count = 0
            for group in duplicate_groups:
                relations = group["relations"]
                
                # Find best relation (highest confidence)
                best_rel = max(relations, key=lambda r: r.get("confidence", 0.5))
                
                # Calculate merged values
                total_mentions = sum(r.get("supporting_mentions", 0) for r in relations)
                earliest_created = min(r.get("created_at", datetime.now().isoformat()) for r in relations)
                latest_mentioned = max(r.get("last_mentioned", datetime.now().isoformat()) for r in relations)
                
                # Update best relation with merged values
                update_query = """
                FOR rel IN entity_relations
                    FILTER rel._key == @key
                    UPDATE rel WITH {
                        supporting_mentions: @mentions,
                        created_at: @created_at,
                        last_mentioned: @last_mentioned
                    } IN entity_relations
                    RETURN NEW
                """
                
                self.kg_store.db.aql.execute(
                    update_query,
                    bind_vars={
                        "key": best_rel["_key"],
                        "mentions": total_mentions,
                        "created_at": earliest_created,
                        "last_mentioned": latest_mentioned
                    }
                )
                
                # Delete other duplicates
                for rel in relations:
                    if rel["_key"] != best_rel["_key"]:
                        delete_query = """
                        FOR rel IN entity_relations
                            FILTER rel._key == @key
                            REMOVE rel IN entity_relations
                        """
                        self.kg_store.db.aql.execute(delete_query, bind_vars={"key": rel["_key"]})
                
                merged_count += 1
                logger.info(f"   ✅ Merged {len(relations)} duplicates: {group['subject']} --{group['predicate']}--> {group['object']}")
            
            logger.info(f"   📈 Merged {merged_count} duplicate groups")
            return {"duplicates_found": len(duplicate_groups), "relations_merged": merged_count}
            
        except Exception as e:
            logger.error(f"Failed to merge duplicate relations: {e}", exc_info=True)
            return {"duplicates_found": 0, "relations_merged": 0}

    async def reprocess_unextracted_memories(self, user_id: str, tier_threshold: int = 2) -> Dict[str, int]:
        """
        Re-process high-tier memories that have no associated relations.
        
        Args:
            user_id: User ID
            tier_threshold: Only process memories with tier <= this value (1=core, 2=long-term)
        
        Returns:
            {"memories_processed": int, "relations_extracted": int}
        """
        logger.info(f"🔄 Re-processing unextracted memories for user {user_id} (tier <= {tier_threshold})")
        
        try:
            # Find high-tier memories without relations
            query = """
            FOR mem IN memories
                FILTER mem.user_id == @user_id
                FILTER mem.tier <= @tier_threshold
                LET rel_count = LENGTH(
                    FOR rel IN entity_relations
                        FILTER rel.user_id == @user_id
                        FILTER rel.memory_id == mem._key
                        RETURN 1
                )
                FILTER rel_count == 0
                SORT mem.ego_score DESC
                LIMIT 50
                RETURN {
                    memory_id: mem._key,
                    content: mem.content,
                    ego_score: mem.ego_score,
                    tier: mem.tier
                }
            """
            
            cursor = self.arango_db.aql.execute(
                query,
                bind_vars={"user_id": user_id, "tier_threshold": tier_threshold}
            )
            memories = list(cursor)
            
            if not memories:
                logger.info(f"   ✅ No unextracted high-tier memories found")
                return {"memories_processed": 0, "relations_extracted": 0}
            
            logger.info(f"   📊 Found {len(memories)} high-tier memories without relations")
            
            total_relations = 0
            processed = 0
            
            for mem in memories:
                try:
                    # Run consolidation on this memory
                    result = await self.consolidation_service.consolidate(
                        text=mem["content"],
                        ego_score=mem["ego_score"],
                        tier=mem["tier"]
                    )
                    
                    # Store extracted relations
                    if result.get("relations"):
                        self.kg_store.store_relations(
                            user_id=user_id,
                            memory_id=mem["memory_id"],
                            relations=result["relations"]
                        )
                    
                    relations_count = len(result.get("relations", []))
                    if relations_count > 0:
                        total_relations += relations_count
                        logger.info(f"   ✅ Extracted {relations_count} relations from memory {mem['memory_id'][:8]}")
                    
                    processed += 1
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to process memory {mem['memory_id']}: {e}")
            
            logger.info(f"   📈 Processed {processed} memories, extracted {total_relations} relations")
            return {"memories_processed": processed, "relations_extracted": total_relations}
            
        except Exception as e:
            logger.error(f"Failed to reprocess unextracted memories: {e}", exc_info=True)
            return {"memories_processed": 0, "relations_extracted": 0}

    async def decay_stale_relations(
        self, 
        user_id: str, 
        stale_days: int = 90,
        min_mentions: int = 2,
        confidence_threshold: float = 0.3
    ) -> Dict[str, int]:
        """
        Decay confidence of stale relations and remove those below threshold.
        
        A relation is stale if:
        - last_mentioned is older than stale_days
        - supporting_mentions < min_mentions
        
        Args:
            user_id: User ID
            stale_days: Days since last_mentioned to consider stale
            min_mentions: Minimum supporting_mentions to avoid decay
            confidence_threshold: Remove relations below this confidence after decay
        
        Returns:
            {"relations_decayed": int, "relations_removed": int}
        """
        logger.info(f"🔄 Decaying stale relations for user {user_id}")
        logger.info(f"   Criteria: last_mentioned > {stale_days} days ago, mentions < {min_mentions}")
        
        try:
            stale_date = (datetime.now() - timedelta(days=stale_days)).isoformat()
            
            # Find stale relations
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.effective_to == null
                FILTER rel.last_mentioned < @stale_date
                FILTER (rel.supporting_mentions || 0) < @min_mentions
                RETURN rel
            """
            
            cursor = self.kg_store.db.aql.execute(
                query,
                bind_vars={
                    "user_id": user_id,
                    "stale_date": stale_date,
                    "min_mentions": min_mentions
                }
            )
            stale_relations = list(cursor)
            
            if not stale_relations:
                logger.info(f"   ✅ No stale relations found")
                return {"relations_decayed": 0, "relations_removed": 0}
            
            logger.info(f"   📊 Found {len(stale_relations)} stale relations")
            
            decayed_count = 0
            removed_count = 0
            
            for rel in stale_relations:
                # Decay confidence by 20%
                new_confidence = rel.get("confidence", 0.5) * 0.8
                
                if new_confidence < confidence_threshold:
                    # Remove relation (set effective_to)
                    update_query = """
                    FOR r IN entity_relations
                        FILTER r._key == @key
                        UPDATE r WITH {
                            effective_to: @effective_to
                        } IN entity_relations
                    """
                    self.kg_store.db.aql.execute(
                        update_query,
                        bind_vars={
                            "key": rel["_key"],
                            "effective_to": datetime.now().isoformat()
                        }
                    )
                    removed_count += 1
                    logger.info(f"   🗑️  Removed: {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {new_confidence:.2f})")
                else:
                    # Update confidence
                    update_query = """
                    FOR r IN entity_relations
                        FILTER r._key == @key
                        UPDATE r WITH {
                            confidence: @confidence
                        } IN entity_relations
                    """
                    self.kg_store.db.aql.execute(
                        update_query,
                        bind_vars={
                            "key": rel["_key"],
                            "confidence": new_confidence
                        }
                    )
                    decayed_count += 1
            
            logger.info(f"   📈 Decayed {decayed_count} relations, removed {removed_count}")
            return {"relations_decayed": decayed_count, "relations_removed": removed_count}
            
        except Exception as e:
            logger.error(f"Failed to decay stale relations: {e}", exc_info=True)
            return {"relations_decayed": 0, "relations_removed": 0}

    async def run_full_reconsolidation(self, user_id: str) -> Dict[str, Any]:
        """
        Run all re-consolidation steps for a user.
        
        Returns:
            Summary of all operations
        """
        logger.info(f"🔄 Starting full KG re-consolidation for user {user_id}")
        logger.info("="*80)
        
        # Step 1: Merge duplicates
        merge_result = await self.merge_duplicate_relations(user_id)
        
        # Step 2: Re-process unextracted memories
        reprocess_result = await self.reprocess_unextracted_memories(user_id)
        
        # Step 3: Decay stale relations
        decay_result = await self.decay_stale_relations(user_id)
        
        summary = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "merge": merge_result,
            "reprocess": reprocess_result,
            "decay": decay_result
        }
        
        logger.info("="*80)
        logger.info(f"✅ Re-consolidation complete for user {user_id}")
        logger.info(f"   Duplicates merged: {merge_result['relations_merged']}")
        logger.info(f"   Memories re-processed: {reprocess_result['memories_processed']}")
        logger.info(f"   Relations extracted: {reprocess_result['relations_extracted']}")
        logger.info(f"   Relations decayed: {decay_result['relations_decayed']}")
        logger.info(f"   Relations removed: {decay_result['relations_removed']}")
        logger.info("="*80)
        
        return summary
