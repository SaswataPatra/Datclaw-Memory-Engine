"""
Memory Ingestion Service

Format-agnostic core service that orchestrates memory ingestion from various sources.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.event_bus import Event, EventBus
from .models import BaseParser, ConversationChunk

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Core ingestion service that:
    1. Manages a registry of parsers
    2. Scores ingested chunks (using ML or trigger-based scoring)
    3. Publishes memory.upsert events
    4. Triggers graph processing
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        ego_scorer,
        ml_scorer=None,
        graph_pipeline=None,
        regex_fallback=None,
        consolidation_service=None,
        knowledge_graph_store=None,
        background_consolidation=None,
        shadow_tier=None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.event_bus = event_bus
        self.ego_scorer = ego_scorer
        self.ml_scorer = ml_scorer
        self.graph_pipeline = graph_pipeline
        self.regex_fallback = regex_fallback
        self.consolidation_service = consolidation_service
        self.knowledge_graph_store = knowledge_graph_store
        self.background_consolidation = background_consolidation
        self.shadow_tier = shadow_tier
        self.config = config or {}
        
        self._parsers: Dict[str, BaseParser] = {}
        
        self.use_ml_scoring = ml_scorer is not None
        # Graph pipeline: DISABLED (files kept for future reference)
        self.use_graph_pipeline = False
        self.use_consolidation = consolidation_service is not None and knowledge_graph_store is not None

        self.default_ego_floor = self.config.get('ingestion', {}).get('default_ego_floor', 0.5)
    
    def register_parser(self, source_type: str, parser: BaseParser):
        """Register a parser for a specific source type."""
        self._parsers[source_type] = parser
        logger.info(f"Registered parser for source type: {source_type}")
    
    def get_parser(self, source_type: str) -> Optional[BaseParser]:
        """Get a parser for a specific source type."""
        return self._parsers.get(source_type)
    
    async def ingest(
        self,
        user_id: str,
        source_type: str,
        source: str,
        session_id: Optional[str] = None,
        **parser_kwargs
    ) -> Dict[str, Any]:
        """
        Ingest memories from a source.
        
        Args:
            user_id: User identifier
            source_type: Type of source (e.g., "chatgpt", "zip", "text")
            source: Source identifier (URL, file path, etc.)
            session_id: Optional session identifier (generated if not provided)
            **parser_kwargs: Additional arguments passed to the parser
            
        Returns:
            Dictionary with ingestion results:
            {
                "status": "success" | "error",
                "chunks_parsed": int,
                "memories_created": int,
                "errors": List[str],
                "session_id": str
            }
        """
        parser = self.get_parser(source_type)
        if not parser:
            raise ValueError(f"No parser registered for source type: {source_type}")
        
        if not session_id:
            session_id = f"ingestion_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting ingestion: user={user_id}, type={source_type}, session={session_id}")
        
        import time
        start_time = time.time()
        
        result = {
            "status": "success",
            "chunks_parsed": 0,
            "memories_created": 0,
            "errors": [],
            "session_id": session_id,
        }
        
        try:
            # Parse the source
            chunks = await parser.parse(source, **parser_kwargs)
            result["chunks_parsed"] = len(chunks)
            
            logger.info(f"Parsed {len(chunks)} chunks from {source_type} source")
            
            # Pre-compute batch consolidation (1 LLM call per 10 memories instead of N)
            # Toggle this to switch between batch mode (faster, less logs) and per-memory mode (slower, detailed logs)
            consolidation_results = {}
            USE_BATCH_CONSOLIDATION = True  # ← CHANGE THIS: True = batch mode, False = per-memory mode
            if self.use_consolidation and len(chunks) > 1 and USE_BATCH_CONSOLIDATION:
                try:
                    logger.info(f"📦 Preparing batch consolidation for {len(chunks)} chunks...")
                    batch_input = []
                    for chunk in chunks:
                        timestamp = chunk.timestamp or datetime.utcnow().isoformat()
                        batch_input.append({
                            "content": chunk.content,
                            "context": chunk.context or "",
                            "ego_score": 0.5,  # Placeholder; actual score comes after fast scoring
                            "tier": 2,
                            "document_date": timestamp,
                        })
                    
                    logger.info(f"🚀 Starting batch consolidation (this will take ~10-15 minutes)...")
                    import time
                    batch_start_time = time.time()
                    
                    batch_results_list = await self.consolidation_service.consolidate_batch(
                        memories=batch_input,
                        max_batch_size=10
                    )
                    
                    batch_elapsed = time.time() - batch_start_time
                    
                    for idx, br in enumerate(batch_results_list):
                        consolidation_results[idx] = br
                    
                    logger.info(f"✅ Batch consolidation complete: {len(batch_results_list)} results in {batch_elapsed:.1f}s")
                except Exception as e:
                    logger.warning(f"❌ Batch consolidation failed, falling back to per-memory: {e}", exc_info=True)
            
            # Process chunks in parallel batches for better performance
            import asyncio
            batch_size = 10  # Process 10 chunks at a time
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                tasks = [
                    self._process_chunk(
                        user_id=user_id,
                        session_id=session_id,
                        chunk=chunk,
                        chunk_index=i + idx,
                        precomputed_consolidation=consolidation_results.get(i + idx)
                    )
                    for idx, chunk in enumerate(batch)
                ]
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successes and errors
                for idx, res in enumerate(batch_results):
                    if isinstance(res, Exception):
                        error_msg = f"Failed to process chunk {i + idx}: {str(res)}"
                        logger.error(error_msg)
                        result["errors"].append(error_msg)
                    else:
                        result["memories_created"] += 1
            
            elapsed = time.time() - start_time
            per_memory = elapsed / max(1, result['memories_created'])
            result["elapsed_seconds"] = round(elapsed, 1)
            result["seconds_per_memory"] = round(per_memory, 1)
            
            logger.info(
                f"Ingestion complete: {result['memories_created']}/{result['chunks_parsed']} "
                f"memories created, {len(result['errors'])} errors, "
                f"{elapsed:.1f}s total ({per_memory:.1f}s/memory)"
            )
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Parser error: {str(e)}")
            logger.error(f"Ingestion failed: {e}", exc_info=True)
        
        return result
    
    async def _process_chunk(
        self,
        user_id: str,
        session_id: str,
        chunk: ConversationChunk,
        chunk_index: int,
        precomputed_consolidation: Optional[Dict[str, Any]] = None
    ):
        """
        Sequential processing per chunk:
        
        1. Trigger-based scoring (fast, no API calls)
        2. Publish memory.upsert v1 (stores in ArangoDB + Qdrant)
        3. Full ML scoring (classification, novelty, frequency)
        4. Update memory if score changed significantly
        5. KG consolidation (entities + relations) using final ego/tier
        6. Publish v2 with entity tags + ego_scoring_complete for KG maintenance
        """
        node_id = f"mem_{uuid.uuid4().hex}"
        timestamp = chunk.timestamp or datetime.utcnow().isoformat()
        
        # ── Step 1: Fast trigger-based scoring ──
        ego_score, confidence, triggers, tier = await self._score_chunk_fast(
            chunk=chunk
        )
        
        logger.info(
            f"[Step 1] Chunk {chunk_index}: ego={ego_score:.2f}, tier={tier}, "
            f"triggers={triggers[:3]}"
        )
        
        # ── Step 2: Publish memory.upsert v1 (ArangoDB + Qdrant) ──
        event = Event(
            topic="memory.upsert",
            event_type="memory.ingested",
            payload={
                "node_id": node_id,
                "user_id": user_id,
                "session_id": session_id,
                "content": chunk.content,
                "ego_score": ego_score,
                "tier": tier,
                "confidence": confidence,
                "source": f"ingestion_{chunk.source_type}",
                "observed_at": timestamp,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed_at": datetime.utcnow().isoformat(),
                "version": 1,
                "metadata": {
                    "triggers": triggers,
                    "assistant_response_length": len(chunk.context) if chunk.context else 0,
                    "ingestion_source": chunk.source_type,
                    "ingestion_metadata": chunk.metadata,
                    "chunk_index": chunk_index,
                    "scoring_phase": "trigger_based",
                }
            }
        )
        await self.event_bus.publish(event)
        
        # ── Step 3: Full ML scoring ──
        if self.use_ml_scoring and self.ml_scorer:
            try:
                ml_ego, ml_confidence, ml_triggers, ml_tier = await self.ml_scorer.score_memory(
                    user_id=user_id,
                    user_message=chunk.content,
                    assistant_response=chunk.context or ""
                )
                
                tier_changed = ml_tier != tier
                score_delta = abs(ml_ego - ego_score)
                
                logger.info(
                    f"[Step 3] Chunk {chunk_index}: ego={ml_ego:.2f} (was {ego_score:.2f}), "
                    f"tier={ml_tier} (was {tier}), delta={score_delta:.2f}"
                )
                
                # ── Step 4: Update memory if scoring changed ──
                if tier_changed or score_delta > 0.15:
                    update_event = Event(
                        topic="memory.upsert",
                        event_type="memory.rescored",
                        payload={
                            "node_id": node_id,
                            "user_id": user_id,
                            "session_id": session_id,
                            "content": chunk.content,
                            "ego_score": ml_ego,
                            "tier": ml_tier,
                            "confidence": ml_confidence,
                            "source": f"ingestion_{chunk.source_type}",
                            "observed_at": timestamp,
                            "created_at": datetime.utcnow().isoformat(),
                            "last_accessed_at": datetime.utcnow().isoformat(),
                            "version": 2,
                            "metadata": {
                                "triggers": ml_triggers,
                                "assistant_response_length": len(chunk.context) if chunk.context else 0,
                                "ingestion_source": chunk.source_type,
                                "ingestion_metadata": chunk.metadata,
                                "chunk_index": chunk_index,
                                "scoring_phase": "ml_rescored",
                                "phase1_ego": ego_score,
                                "phase1_tier": tier,
                            }
                        }
                    )
                    await self.event_bus.publish(update_event)
                    logger.info(f"[Step 4] Updated memory {node_id}: tier {tier}→{ml_tier}")
                
                # Use final ML scores going forward
                ego_score = ml_ego
                confidence = ml_confidence
                triggers = ml_triggers
                tier = ml_tier
                
            except Exception as e:
                logger.warning(f"[Step 3] ML scoring failed for chunk {chunk_index} (non-fatal): {e}")
        
        # ── Step 4.5: Shadow tier (Tier 1 → Tier 2 for user confirmation) ──
        shadow_tier_applied = False
        if tier == 1 and self.shadow_tier:
            try:
                use_shadow = await self.shadow_tier.should_use_shadow_tier(
                    ego_score=ego_score,
                    confidence=confidence
                )
                if use_shadow:
                    shadow_tier_applied = True
                    tier = 2
                    clarification_id, question = await self.shadow_tier.propose_core_memory({
                        'node_id': node_id,
                        'user_id': user_id,
                        'content': chunk.content,
                        'summary': chunk.content[:200] + ('...' if len(chunk.content) > 200 else ''),
                        'ego_score': ego_score,
                        'confidence': confidence,
                        'sources': [f"ingestion_{chunk.source_type}"],
                    })
                    shadow_event = Event(
                        topic="memory.upsert",
                        event_type="memory.shadow_tier",
                        payload={
                            "node_id": node_id,
                            "user_id": user_id,
                            "session_id": session_id,
                            "content": chunk.content,
                            "ego_score": ego_score,
                            "tier": tier,
                            "confidence": confidence,
                            "source": f"ingestion_{chunk.source_type}",
                            "observed_at": timestamp,
                            "created_at": datetime.utcnow().isoformat(),
                            "last_accessed_at": datetime.utcnow().isoformat(),
                            "version": 3,
                            "metadata": {
                                "triggers": triggers,
                                "ingestion_source": chunk.source_type,
                                "ingestion_metadata": chunk.metadata,
                                "chunk_index": chunk_index,
                                "shadow_tier_candidate": True,
                                "original_tier": 1,
                                "clarification_id": clarification_id,
                            }
                        }
                    )
                    await self.event_bus.publish(shadow_event)
                    logger.info(f"[Step 4.5] Chunk {chunk_index}: Tier 1 → Shadow Tier (clarification_id={clarification_id})")
            except Exception as e:
                logger.warning(f"[Step 4.5] Shadow tier failed for chunk {chunk_index} (non-fatal): {e}")
        
        # ── Step 5: KG consolidation (entities + relations + temporal) with final scores ──
        relations = []
        entities = []
        temporal_data = {"event_dates": [], "time_expressions": []}
        context_summary = ""
        if self.use_consolidation:
            try:
                if precomputed_consolidation:
                    # Use batch-precomputed results (includes context from assistant)
                    result = precomputed_consolidation
                    context_summary = result.get("context_summary", "")
                    logger.info(f"   📦 Using precomputed batch consolidation for chunk {chunk_index}")
                else:
                    # Fallback: single-memory consolidation (includes context if available)
                    result = await self.consolidation_service.consolidate(
                        text=chunk.content,
                        ego_score=ego_score,
                        tier=tier,
                        document_date=timestamp,
                        context=chunk.context  # Pass assistant response for context_summary
                    )
                    context_summary = result.get("context_summary", "")
                
                entities = result.get("entities", [])
                relations = result.get("relations", [])
                temporal_data = result.get("temporal", {"event_dates": [], "time_expressions": []})
                
                logger.info(f"   ✅ KG consolidation: {len(entities)} entities, {len(relations)} relations extracted")
                
                if relations:
                    stored = self.knowledge_graph_store.store_relations(
                        user_id=user_id,
                        memory_id=node_id,
                        relations=relations
                    )
                    logger.info(f"   💾 Stored {stored}/{len(relations)} relations in KG")
                
            except Exception as e:
                logger.warning(f"KG consolidation failed (non-fatal): {e}")
        
        # ── Step 6: Publish entity tags + KG maintenance event ──
        entity_names = [e.get('name', e) if isinstance(e, dict) else e for e in entities]
        entity_names = [str(e).lower() for e in entity_names if e]
        
        if entity_names:
            entity_metadata = {
                "triggers": triggers,
                "assistant_response_length": len(chunk.context) if chunk.context else 0,
                "ingestion_source": chunk.source_type,
                "ingestion_metadata": chunk.metadata,
                "chunk_index": chunk_index,
                "scoring_phase": "consolidation",
                "temporal": temporal_data,
                "context_summary": context_summary,
            }
            if shadow_tier_applied:
                entity_metadata["shadow_tier_candidate"] = True
            entity_update_event = Event(
                topic="memory.upsert",
                event_type="memory.entities_extracted",
                payload={
                    "node_id": node_id,
                    "user_id": user_id,
                    "content": chunk.content,
                    "ego_score": ego_score,
                    "tier": tier,
                    "confidence": confidence,
                    "source": f"ingestion_{chunk.source_type}",
                    "observed_at": timestamp,
                    "created_at": datetime.utcnow().isoformat(),
                    "last_accessed_at": datetime.utcnow().isoformat(),
                    "entities": entity_names,
                    "version": 3,
                    "metadata": entity_metadata
                }
            )
            await self.event_bus.publish(entity_update_event)
            logger.info(f"   📡 Published entity tags: {entity_names}")
        else:
            # Still store temporal + context_summary even without entities
            if temporal_data.get("event_dates") or context_summary:
                meta_update_event = Event(
                    topic="memory.upsert",
                    event_type="memory.metadata_enriched",
                    payload={
                        "node_id": node_id,
                        "user_id": user_id,
                        "content": chunk.content,
                        "ego_score": ego_score,
                        "tier": tier,
                        "confidence": confidence,
                        "source": f"ingestion_{chunk.source_type}",
                        "observed_at": timestamp,
                        "created_at": datetime.utcnow().isoformat(),
                        "last_accessed_at": datetime.utcnow().isoformat(),
                        "entities": [],
                        "version": 3,
                        "metadata": {
                            "triggers": triggers,
                            "ingestion_source": chunk.source_type,
                            "ingestion_metadata": chunk.metadata,
                            "chunk_index": chunk_index,
                            "scoring_phase": "consolidation",
                            "temporal": temporal_data,
                            "context_summary": context_summary,
                        }
                    }
                )
                await self.event_bus.publish(meta_update_event)
                logger.info(f"   📡 Published metadata enrichment (temporal/context_summary)")
            else:
                logger.info(f"   ℹ️  No entities or metadata to store")
        
        # Always emit ego_scoring_complete for KG maintenance
        kg_event = Event(
            topic="events:ego_scoring_complete",
            event_type="ego_scoring_complete",
            payload={
                "user_id": user_id,
                "memory_id": node_id,
                "memory_content": chunk.content,
                "ego_score": ego_score,
                "tier": tier,
                "confidence": confidence,
                "new_relations": relations,
            }
        )
        await self.event_bus.publish(kg_event)
        logger.info(f"   📡 Published ego_scoring_complete for KG maintenance")
    
    async def _score_chunk_fast(
        self,
        chunk: ConversationChunk
    ) -> tuple[float, float, List[str], int]:
        """
        Fast trigger-based scoring for Phase 1 (immediate storage).
        
        Uses regex pattern matching and the ego scorer with whatever
        signals we can compute locally without API calls.
        
        Returns:
            (ego_score, confidence, triggers, tier)
        """
        triggers = []
        if self.regex_fallback:
            triggers = self.regex_fallback.detect_triggers(chunk.content)
        
        if triggers:
            importance_map = {
                'identity': 1.0,
                'family': 1.0,
                'high_value': 0.95,
                'preference': 0.9,
                'fact': 0.7
            }
            explicit_importance = max(
                importance_map.get(trigger, 0.5) for trigger in triggers
            )
            
            memory_data = {
                'explicit_importance': explicit_importance,
                'observed_at': chunk.timestamp or datetime.utcnow().isoformat(),
                'user_response_length': len(chunk.context) if chunk.context else 0,
                'llm_confidence': 0.7,  # Lower confidence since this is fast scoring
                'source_weight': 0.6,   # External source, not live conversation
            }
            
            ego_result = self.ego_scorer.calculate(memory_data)
            ego_score = ego_result.ego_score
            confidence = 0.7
        else:
            ego_score = self.default_ego_floor
            confidence = 0.5
        
        tier = self._determine_tier(ego_score, confidence)
        
        return ego_score, confidence, triggers, tier
    
    def _determine_tier(self, ego_score: float, confidence: float) -> int:
        """
        Determine memory tier based on ego score.
        
        Simplified version without shadow tier (ingested memories bypass shadow tier).
        """
        if ego_score >= 0.8:
            return 1
        elif ego_score >= 0.6:
            return 2
        elif ego_score >= 0.3:
            return 3
        else:
            return 4
