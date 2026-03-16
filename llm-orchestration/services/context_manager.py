"""
DAPPY - Context Memory Manager
Intelligent LLM context window management with ego-based flushing
Includes PPR-based graph retrieval for relevant memories
"""

import tiktoken
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import asyncio
import spacy

from core.scoring.ego_scorer import TemporalEgoScorer
from core.event_bus import Event, EventBus
from adapters.redis_message_bus import RedisMessageBus

# Qdrant for vector search
try:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# PPR retrieval for graph-based memory retrieval
try:
    from core.graph.ppr_retrieval import PPRRetrieval
    PPR_AVAILABLE = True
except ImportError:
    PPR_AVAILABLE = False
    PPRRetrieval = None

logger = logging.getLogger(__name__)


class ContextMemoryManager:
    """
    Manage LLM context window with intelligent ego-based flushing
    
    Responsibilities:
    - Monitor context token usage
    - Intelligent flush at 50% capacity
    - Emergency flush at 80% capacity
    - Ego-based prioritization (keep/summarize/drop)
    - Store messages to hot tier
    - Flush to Tier 4 for consolidation
    """
    
    def __init__(
        self,
        redis_client,
        config: Dict[str, Any],
        ego_scorer: TemporalEgoScorer,
        event_bus: EventBus,
        message_store: RedisMessageBus,
        chatbot_service=None,  # Optional: for accessing processing state
        arango_db=None,  # Optional: ArangoDB for memory storage
        qdrant_client=None,  # Optional: Qdrant for vector search
        embedding_service=None,  # Optional: for query embedding in vector search
        knowledge_graph_store=None,  # Optional: for relation context in retrieval
        query_understanding_service=None  # Optional: for entity extraction from queries
    ):
        self.redis = redis_client
        self.config = config
        self.ego_scorer = ego_scorer
        self.event_bus = event_bus
        self.message_store = message_store
        self.chatbot_service = chatbot_service  # For processing state tracking
        self.arango_db = arango_db
        self.qdrant_client = qdrant_client
        self.embedding_service = embedding_service
        self.knowledge_graph_store = knowledge_graph_store
        self.query_understanding_service = query_understanding_service
        
        # Initialize PPR retrieval if ArangoDB is available
        self.ppr_retrieval = None
        if arango_db and PPR_AVAILABLE:
            try:
                self.ppr_retrieval = PPRRetrieval(db=arango_db, config=config)
                logger.info("PPR retrieval initialized for graph-based memory retrieval")
            except Exception as e:
                logger.warning(f"Failed to initialize PPR retrieval: {e}")
        
        # Initialize spaCy for entity extraction
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy loaded for entity extraction from queries")
        except Exception as e:
            logger.warning(f"Failed to load spaCy: {e}")
        
        # Context limits
        context_config = config.get('context_memory', {})
        
        # ⚙️ STAYS - LLM model limit (determined by OpenAI/provider)
        self.max_tokens = context_config.get('max_tokens', 128000)
        
        # ⚙️ STAYS - Safety thresholds for proactive and emergency flushing
        # 0.5 = flush at 50% to avoid hitting limits during long conversations
        # 0.8 = emergency flush at 80% to prevent context overflow
        self.flush_threshold = context_config.get('flush_threshold', 0.5)
        self.emergency_threshold = context_config.get('emergency_threshold', 0.8)
        
        # 🔄 HYPERPARAMETER - Will be TUNED in Phase 2 (per-user preference)
        # Flushing config for hybrid strategy
        # Current: Keep last 20 messages (balance between context and freshness)
        # Future: Learned per-user (some want more context, others want faster responses)
        self.keep_recent_messages = context_config.get('keep_recent_messages', 20)
        
        # ⚙️ STAYS - Computational efficiency limit for summarization
        # Maximum messages to summarize in a single batch (prevents LLM timeout)
        self.summarize_batch_size = context_config.get('summarize_batch_size', 50)
        
        # Token counter
        llm_model = config.get('llm', {}).get('model', 'gpt-4-turbo-preview')
        try:
            self.encoding = tiktoken.encoding_for_model(llm_model)
        except:
            # Fallback to cl100k_base (GPT-4)
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Tier 4 config
        tier4_config = config.get('tiers', {}).get('tier4', {})
        self.tier4_ttl = tier4_config.get('ttl_seconds', 600)  # 10 minutes
        self.tier4_prefix = tier4_config.get('redis_key_prefix', 'tier4:')
        
        logger.info(
            f"Context Manager initialized",
            extra={
                "max_tokens": self.max_tokens,
                "flush_threshold": self.flush_threshold,
                "emergency_threshold": self.emergency_threshold,
                "keep_recent_messages": self.keep_recent_messages
            }
        )
    
    def set_chatbot_service(self, chatbot_service):
        """
        Set chatbot service reference (for processing state tracking)
        Called after both services are initialized to avoid circular dependency
        """
        self.chatbot_service = chatbot_service
        logger.info("ChatbotService reference set in ContextManager")
    
    async def manage_context(
        self,
        user_id: str,
        session_id: str,
        conversation_history: List[Dict],
        new_message: Optional[Dict] = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Manage context before LLM call
        
        Args:
            user_id: User ID
            session_id: Session ID
            conversation_history: Current conversation messages
            new_message: New message to add (if any)
        
        Returns:
            Tuple of (optimized_history, metadata)
        """
        
        # Store new message if provided
        if new_message:
            await self._store_message(user_id, session_id, new_message)
            conversation_history.append(new_message)
        
        # Count current tokens
        current_tokens = self._count_tokens(conversation_history)
        usage_percent = current_tokens / self.max_tokens
        
        logger.info(
            f"Context usage: {current_tokens}/{self.max_tokens} ({usage_percent:.1%})",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "tokens": current_tokens,
                "usage_percent": usage_percent
            }
        )
        
        metadata = {
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": usage_percent,
            "flushed": False,
            "emergency_flush": False
        }
        
        # Emergency flush
        if usage_percent > self.emergency_threshold:
            logger.warning(
                f"EMERGENCY FLUSH at {usage_percent:.1%}",
                extra={"user_id": user_id, "session_id": session_id}
            )
            
            optimized_history = await self._emergency_flush(
                user_id,
                session_id,
                conversation_history
            )
            
            metadata["flushed"] = True
            metadata["emergency_flush"] = True
            metadata["tokens_after_flush"] = self._count_tokens(optimized_history)
            
            return optimized_history, metadata
        
        # Intelligent flush
        elif usage_percent > self.flush_threshold:
            logger.info(
                f"Intelligent flush at {usage_percent:.1%}",
                extra={"user_id": user_id, "session_id": session_id}
            )
            
            optimized_history = await self._intelligent_flush(
                user_id,
                session_id,
                conversation_history
            )
            
            metadata["flushed"] = True
            metadata["emergency_flush"] = False
            metadata["tokens_after_flush"] = self._count_tokens(optimized_history)
            
            return optimized_history, metadata
        
        # No flush needed
        return conversation_history, metadata
    
    async def _emergency_flush(
        self,
        user_id: str,
        session_id: str,
        conversation_history: List[Dict]
    ) -> List[Dict]:
        """
        Emergency flush: Aggressively remove messages to free space
        
        Strategy:
        1. Keep system message (first)
        2. Keep last N messages (recent context)
        3. Flush everything else to Tier 4
        """
        
        if not conversation_history:
            return conversation_history
        
        # Keep system message
        system_message = conversation_history[0] if conversation_history[0].get('role') == 'system' else None
        
        # Keep last 10 messages (5 exchanges)
        keep_count = min(10, len(conversation_history))
        recent_messages = conversation_history[-keep_count:]
        
        # Flush the middle messages
        messages_to_flush = conversation_history[1:-keep_count] if len(conversation_history) > keep_count + 1 else []
        
        if messages_to_flush:
            await self._flush_to_tier4(user_id, session_id, messages_to_flush, priority="HIGH")
        
        # Build optimized history
        optimized = []
        if system_message:
            optimized.append(system_message)
        optimized.extend(recent_messages)
        
        logger.info(
            f"Emergency flush: kept {len(optimized)}, flushed {len(messages_to_flush)}",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "kept": len(optimized),
                "flushed": len(messages_to_flush)
            }
        )
        
        return optimized
    
    async def _intelligent_flush(
        self,
        user_id: str,
        session_id: str,
        conversation_history: List[Dict]
    ) -> List[Dict]:
        """
        Intelligent flush: 3-tier strategy based on processing state
        
        Strategy:
        1. Keep last N messages (recent context)
        2. Flush completed messages (already in DB)
        3. Summarize + flush unprocessed messages (not in DB yet)
        """
        
        if not conversation_history:
            return conversation_history
        
        # Keep system message
        system_message = conversation_history[0] if conversation_history[0].get('role') == 'system' else None
        start_idx = 1 if system_message else 0
        
        # Step 1: Keep last N messages
        keep_count = min(self.keep_recent_messages, len(conversation_history) - start_idx)
        recent_messages = conversation_history[-keep_count:]
        old_messages = conversation_history[start_idx:-keep_count] if len(conversation_history) > start_idx + keep_count else []
        
        if not old_messages:
            # Nothing to flush
            return conversation_history
        
        # Step 2: Categorize old messages by processing state
        completed_messages = []
        unprocessed_messages = []
        
        if self.chatbot_service:
            processing_state = self.chatbot_service.get_processing_state(session_id)
            
            for msg in old_messages:
                msg_id = msg.get('message_id')
                state = processing_state.get(msg_id, 'unknown')
                
                if state == 'completed':
                    completed_messages.append(msg)
                else:
                    # pending, processing, or unknown - treat as unprocessed
                    unprocessed_messages.append(msg)
        else:
            # No chatbot_service - treat all as unprocessed (safe default)
            unprocessed_messages = old_messages
        
        # Step 3: Flush completed messages (background)
        if completed_messages:
            asyncio.create_task(
                self._flush_completed_messages(user_id, session_id, completed_messages)
            )
        
        # Step 4: Summarize + flush unprocessed messages
        summary_message = None
        if unprocessed_messages:
            summary_message = await self._summarize_and_flush(
                user_id, session_id, unprocessed_messages
            )
        
        # Step 5: Build new history
        optimized = []
        if system_message:
            optimized.append(system_message)
        if summary_message:
            optimized.append(summary_message)
        optimized.extend(recent_messages)
        
        logger.info(
            f"Intelligent flush: completed={len(completed_messages)}, "
            f"unprocessed={len(unprocessed_messages)}, recent={len(recent_messages)}",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "completed": len(completed_messages),
                "unprocessed": len(unprocessed_messages),
                "recent": len(recent_messages)
            }
        )
        
        return optimized
    
    async def _flush_completed_messages(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict]
    ) -> None:
        """
        Flush messages that are already processed and stored in DB
        These can be safely removed from context
        """
        for msg in messages:
            await self._flush_to_tier4(
                user_id, session_id, [msg],
                priority="LOW"  # Already in DB, low priority for consolidation
            )
        
        logger.info(
            f"Flushed {len(messages)} completed messages",
            extra={"user_id": user_id, "session_id": session_id, "count": len(messages)}
        )
    
    async def _summarize_and_flush(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict]
    ) -> Optional[Dict]:
        """
        Summarize unprocessed messages and flush to Redis
        Returns summary message to inject into conversation history
        """
        if not messages:
            return None
        
        # TODO Phase 1.5: Use LLM to generate summary
        # For now, create a simple summary
        summary_content = f"[Previous context: {len(messages)} messages from earlier in the conversation]"
        
        # Flush all unprocessed messages to Tier 4
        for msg in messages:
            await self._flush_to_tier4(
                user_id, session_id, [msg],
                priority="HIGH"  # Not yet processed, high priority
            )
        
        logger.info(
            f"Summarized and flushed {len(messages)} unprocessed messages",
            extra={"user_id": user_id, "session_id": session_id, "count": len(messages)}
        )
        
        # Return summary message
        return {
            "message_id": f"summary_{session_id}_{datetime.utcnow().timestamp()}",
            "role": "system",
            "content": summary_content,
            "timestamp": datetime.utcnow().isoformat(),
            "is_summary": True
        }
    
    async def _flush_to_tier4(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict],
        priority: str = "MED"
    ) -> None:
        """
        Flush messages to Tier 4 (Redis hot buffer) for consolidation
        
        Args:
            messages: Messages to flush
            priority: Priority for consolidation queue (HIGH/MED/LOW)
        """
        
        for msg in messages:
            # Create Tier 4 key
            message_id = msg.get('message_id', f"msg_{datetime.utcnow().timestamp()}")
            tier4_key = f"{self.tier4_prefix}{user_id}:{session_id}:{message_id}"
            
            # Store in Redis with TTL
            tier4_data = {
                **msg,
                "user_id": user_id,
                "session_id": session_id,
                "flushed_at": datetime.utcnow().isoformat(),
                "priority": priority
            }
            
            try:
                await self.redis.setex(
                    tier4_key,
                    self.tier4_ttl,
                    str(tier4_data)
                )
                
                # Publish event for consolidation
                event = Event(
                    topic="tier4.flush",
                    event_type="context.flush",
                    payload={
                        "user_id": user_id,
                        "session_id": session_id,
                        "tier4_key": tier4_key,
                        "priority": priority,
                        "message_id": message_id
                    }
                )
                
                await self.event_bus.publish(event)
                
                logger.debug(
                    f"Flushed message to Tier 4: {tier4_key}",
                    extra={"tier4_key": tier4_key, "priority": priority}
                )
                
            except Exception as e:
                logger.error(f"Failed to flush to Tier 4: {e}", exc_info=True)
    
    async def _store_message(
        self,
        user_id: str,
        session_id: str,
        message: Dict
    ) -> None:
        """Store message in hot tier (Redis Streams)"""
        
        try:
            await self.message_store.append(
                message_id=message.get('message_id', f"msg_{datetime.utcnow().timestamp()}"),
                user_id=user_id,
                session_id=session_id,
                role=message.get('role', 'user'),
                content=message.get('content', ''),
                observed_at=datetime.fromisoformat(message.get('timestamp', datetime.utcnow().isoformat())),
                sequence=message.get('sequence', 0),
                metadata=message.get('metadata', {})
            )
        except Exception as e:
            logger.error(f"Failed to store message: {e}", exc_info=True)
    
    async def _enqueue_consolidation(
        self,
        user_id: str,
        tier: int,
        memory_data: Dict[str, Any],
        priority: str = "MED"
    ) -> None:
        """
        Enqueue memory for consolidation processing
        
        Args:
            user_id: User ID
            tier: Target tier (2, 3, or 4)
            memory_data: Memory data to consolidate
            priority: Priority level (HIGH/MED/LOW)
        """
        queue_name = f"consolidation:{priority.lower()}"
        
        payload = {
            "user_id": user_id,
            "tier": tier,
            **memory_data,
            "priority": priority,
            "enqueued_at": datetime.utcnow().isoformat()
        }
        
        try:
            # Add to consolidation queue (Redis Stream)
            await self.redis.xadd(queue_name, payload)
            
            logger.debug(
                f"Enqueued for consolidation: tier={tier}, priority={priority}",
                extra={"user_id": user_id, "tier": tier, "priority": priority}
            )
        except Exception as e:
            logger.error(f"Failed to enqueue consolidation: {e}", exc_info=True)
    
    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract named entities from user query for PPR seeding and entity expansion.
        Uses LLM (query_understanding_service) when available, else spaCy fallback.
        
        Args:
            query: User's query text
            
        Returns:
            List of entity canonical names
        """
        # Prefer LLM-based extraction (better quality)
        if self.query_understanding_service:
            try:
                entities = await self.query_understanding_service.extract_entities(query)
                if entities:
                    logger.debug(f"LLM extracted {len(entities)} entities: {entities[:5]}")
                    return entities
            except Exception as e:
                logger.warning(f"LLM entity extraction failed, falling back to spaCy: {e}")
        
        # Fallback: spaCy
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(query)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']:
                    entities.append(ent.text.lower())
            
            for chunk in doc.noun_chunks:
                if chunk.root.pos_ in ['NOUN', 'PROPN']:
                    entities.append(chunk.text.lower())
            
            seen = set()
            unique_entities = []
            for entity in entities:
                if entity not in seen:
                    seen.add(entity)
                    unique_entities.append(entity)
            
            logger.debug(f"spaCy extracted {len(unique_entities)} entities: {unique_entities[:5]}")
            return unique_entities[:10]
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _resolve_entity_ids(self, user_id: str, entity_names: List[str]) -> List[str]:
        """
        Resolve entity names to entity IDs in the graph
        
        Args:
            user_id: User ID
            entity_names: List of entity canonical names
            
        Returns:
            List of entity IDs
        """
        if not self.arango_db or not entity_names:
            return []
        
        try:
            # Query ArangoDB for entities matching these names
            query = """
            FOR entity IN entities
                FILTER entity.user_id == @user_id
                FILTER LOWER(entity.canonical_name) IN @names OR
                       LENGTH(
                           FOR alias IN entity.aliases
                               FILTER LOWER(alias) IN @names
                               RETURN 1
                       ) > 0
                RETURN entity._key
            """
            
            cursor = self.arango_db.aql.execute(
                query,
                bind_vars={
                    'user_id': user_id,
                    'names': entity_names
                }
            )
            
            entity_ids = list(cursor)
            logger.debug(f"Resolved {len(entity_ids)} entity IDs from {len(entity_names)} names")
            return entity_ids
            
        except Exception as e:
            logger.warning(f"Entity ID resolution failed: {e}")
            return []
    
    async def _find_kg_relevant_memories(
        self,
        user_id: str,
        query: str,
        vector_results: List[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[tuple[str, float]]:
        """
        Find memory IDs that are semantically relevant to the query by having the LLM
        reason over KG relations and floating high-tier nodes.
        
        Provides the LLM with contextual signals:
        - Whether localized KG was possible (entity tags from vector search)
        - Entity tags found in vector results (if any)
        - KG relations (structured knowledge)
        - Floating high-tier nodes: tier 1/2 memories with no KG relations
        
        Args:
            user_id: User ID
            query: User's search query
            vector_results: Memories from vector search (with 'entities' and 'tier' fields)
            limit: Max memory IDs to return
            
        Returns:
            List of (memory_id, relevance_score) tuples
        """
        if not query or not self.arango_db:
            return []
        
        try:
            # --- Signal 1: Entity tags from vector search ---
            vector_entities = set()
            if vector_results:
                for mem in vector_results:
                    for ent in mem.get('entities', []):
                        if isinstance(ent, str) and ent:
                            vector_entities.add(ent.lower())
                        elif isinstance(ent, dict) and ent.get('name'):
                            vector_entities.add(ent['name'].lower())
            
            is_localized = len(vector_entities) > 0
            
            # --- Fetch KG relations ---
            relations_query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.deprecated != true
                FILTER rel.confidence >= 0.7
                SORT rel.confidence DESC, rel.supporting_mentions DESC
                LIMIT 100
                RETURN {
                    subject: rel.subject,
                    predicate: rel.predicate,
                    object: rel.object,
                    confidence: rel.confidence,
                    memory_id: rel.memory_id
                }
            """
            cursor = self.arango_db.aql.execute(
                relations_query,
                bind_vars={'user_id': user_id}
            )
            relations = list(cursor)
            
            # --- Fetch floating high-tier nodes ---
            # Tier 1/2 memories that have NO KG relations (orphaned important memories)
            related_mem_ids = set(r['memory_id'] for r in relations if r.get('memory_id'))
            
            floating_query = """
            FOR m IN memories
                FILTER m.user_id == @user_id
                FILTER m.tier IN [1, 2]
                SORT m.ego_score DESC
                LIMIT 15
                RETURN {
                    memory_id: m._key,
                    content: m.content,
                    ego_score: m.ego_score,
                    tier: m.tier
                }
            """
            cursor = self.arango_db.aql.execute(
                floating_query,
                bind_vars={'user_id': user_id}
            )
            all_high_tier = list(cursor)
            floating_nodes = [m for m in all_high_tier if m['memory_id'] not in related_mem_ids]
            
            if not relations and not floating_nodes:
                logger.debug(f"No KG relations or floating nodes for user {user_id}")
                return []
            
            logger.info(f"   🔍 KG agent context: {len(relations)} relations, {len(floating_nodes)} floating nodes, localized={is_localized}")
            
            # --- Build LLM prompt with signals ---
            from openai import AsyncOpenAI
            import json
            client = AsyncOpenAI()
            
            # Format relations
            relations_section = ""
            if relations:
                relations_text = "\n".join([
                    f"  R{i}. {r['subject']} --{r['predicate']}--> {r['object']} (confidence: {r['confidence']:.2f})"
                    for i, r in enumerate(relations)
                ])
                relations_section = f"KNOWLEDGE GRAPH RELATIONS:\n{relations_text}"
            else:
                relations_section = "KNOWLEDGE GRAPH RELATIONS:\n  (none found)"
            
            # Format floating nodes (truncated content)
            floating_section = ""
            if floating_nodes:
                floating_text = "\n".join([
                    f"  F{i}. [tier {m['tier']}, ego={m['ego_score']:.2f}] {m['content'][:120]}..."
                    for i, m in enumerate(floating_nodes)
                ])
                floating_section = f"\nFLOATING HIGH-PRIORITY MEMORIES (no KG relations yet, but high importance):\n{floating_text}"
            
            # Format signals
            signal_lines = []
            if is_localized:
                signal_lines.append(f"- LOCALIZED KG: Entity tags were found in vector search results: {sorted(vector_entities)}")
                signal_lines.append("- This means the entity extraction pipeline worked for some memories.")
            else:
                signal_lines.append("- NON-LOCALIZED KG: No entity tags were found in vector search results.")
                signal_lines.append("- This means the entity extraction pipeline did not populate tags for these memories. Relations and floating nodes are especially important for finding relevant content.")
            signals_section = "\n".join(signal_lines)
            
            prompt = f"""You are a KG retrieval agent. Your job is to examine the user's query against the available knowledge graph data and identify which pieces of knowledge help answer the query.

QUERY: {query}

SIGNALS:
{signals_section}

{relations_section}
{floating_section}

TASK: Return which relations (R-indices) and/or floating memories (F-indices) are relevant to answering the query.
- For relations: consider the FULL triple (subject, predicate, object) and whether the relation provides information that answers the query.
- For floating memories: consider whether the memory content is relevant to the query.
- Think about semantic connections, not just keyword overlap.

Return a JSON object:
{{"relations": [{{"index": 0, "relevance": 0.95}}], "floating": [{{"index": 0, "relevance": 0.90}}]}}

Return {{"relations": [], "floating": []}} if nothing is relevant.

JSON:"""
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a KG retrieval agent that identifies relevant knowledge for answering queries. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            logger.info(f"   🤖 KG agent response: {content[:300]}")
            result = json.loads(content)
            
            if not isinstance(result, dict):
                logger.warning(f"KG agent returned non-dict: {content}")
                return []
            
            # --- Extract memory_ids with relevance scores ---
            memory_relevance = {}
            
            # From relations
            for item in result.get('relations', []):
                if isinstance(item, dict):
                    idx = item.get('index')
                    relevance = item.get('relevance', 0.85)
                    if isinstance(idx, int) and 0 <= idx < len(relations):
                        mem_id = relations[idx]['memory_id']
                        if mem_id:
                            memory_relevance[mem_id] = max(memory_relevance.get(mem_id, 0), relevance)
            
            # From floating nodes
            for item in result.get('floating', []):
                if isinstance(item, dict):
                    idx = item.get('index')
                    relevance = item.get('relevance', 0.85)
                    if isinstance(idx, int) and 0 <= idx < len(floating_nodes):
                        mem_id = floating_nodes[idx]['memory_id']
                        memory_relevance[mem_id] = max(memory_relevance.get(mem_id, 0), relevance)
            
            sorted_memories = sorted(memory_relevance.items(), key=lambda x: x[1], reverse=True)
            
            rel_count = len(result.get('relations', []))
            float_count = len(result.get('floating', []))
            logger.info(f"   🔍 KG agent: {rel_count} relations + {float_count} floating → {len(sorted_memories)} unique memories")
            
            return sorted_memories[:limit * 2]
            
        except Exception as e:
            logger.warning(f"KG relevance matching failed: {e}", exc_info=True)
            return []
    
    async def _vector_search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search via Qdrant. Embeds query, searches vectors, fetches full memories from ArangoDB.
        
        Returns:
            List of memory dicts with content, ego_score, tier, relevance_score (from Qdrant)
        """
        if not self.qdrant_client or not self.embedding_service or not self.arango_db:
            return []
        if not QDRANT_AVAILABLE:
            return []
        
        try:
            # Embed query
            query_embedding = await self.embedding_service.generate(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding for vector search")
                return []
            
            qdrant_config = self.config.get('qdrant', {})
            collection_name = qdrant_config.get('collection_name', 'memories')
            
            # Search Qdrant with user_id filter
            query_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
            
            hits = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            
            if not hits:
                logger.debug("Vector search returned no hits")
                return []
            
            # Extract node_ids, scores, and entity tags from Qdrant payloads
            node_ids = [h.payload.get("node_id") for h in hits if h.payload.get("node_id")]
            scores = {h.payload.get("node_id"): h.score for h in hits if h.payload.get("node_id")}
            entity_tags = {h.payload.get("node_id"): h.payload.get("entities", []) for h in hits if h.payload.get("node_id")}
            
            if not node_ids:
                return []
            
            # Fetch full memories from ArangoDB
            memories_query = """
            FOR memory IN memories
                FILTER memory._key IN @node_ids
                FILTER memory.user_id == @user_id
                RETURN {
                    memory_id: memory._key,
                    content: memory.content,
                    ego_score: memory.ego_score,
                    tier: memory.tier,
                    created_at: memory.created_at,
                    observed_at: memory.observed_at,
                    metadata: memory.metadata,
                    source: 'vector_search',
                    relevance_score: null
                }
            """
            
            cursor = self.arango_db.aql.execute(
                memories_query,
                bind_vars={"node_ids": node_ids, "user_id": user_id}
            )
            memories = list(cursor)
            
            # Attach relevance scores and entity tags from Qdrant payload
            for mem in memories:
                mem["relevance_score"] = scores.get(mem["memory_id"], 0.0)
                mem["entities"] = entity_tags.get(mem["memory_id"], [])
            
            # Sort by relevance (Qdrant order)
            memories.sort(key=lambda m: m.get("relevance_score", 0), reverse=True)
            
            logger.info(f"Vector search retrieved {len(memories)} memories for query")
            return memories
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}", exc_info=True)
            return []
    
    async def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        max_memories: int = 5,
        use_ppr: bool = True,
        use_vector: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using KG-boosted vector search
        
        Args:
            user_id: User ID
            query: User's query text
            max_memories: Maximum memories to retrieve
            use_ppr: Enable KG-based boosting (repurposed from PPR)
            use_vector: Enable vector similarity search
            
        Returns:
            List of relevant memory documents with scores
        """
        all_memories = []
        
        # Vector search (primary retrieval path)
        if use_vector and self.qdrant_client and self.embedding_service:
            try:
                vector_memories = await self._vector_search_memories(
                    user_id=user_id,
                    query=query,
                    limit=max_memories * 3  # Fetch extra for KG boosting
                )
                all_memories.extend(vector_memories)
                if vector_memories:
                    logger.info(f"   Retrieved {len(vector_memories)} memories via vector search")
            except Exception as e:
                logger.warning(f"Vector search failed: {e}", exc_info=True)
        
        # KG-based boosting: Use LLM to find memories connected to semantically relevant KG relations
        if use_ppr and self.knowledge_graph_store:
            try:
                kg_results = await self._find_kg_relevant_memories(
                    user_id=user_id,
                    query=query,
                    vector_results=all_memories,
                    limit=max_memories
                )
                
                if kg_results:
                    # kg_results is list of (memory_id, relevance_score) tuples
                    kg_memory_ids = [mem_id for mem_id, _ in kg_results]
                    kg_relevance_map = {mem_id: relevance for mem_id, relevance in kg_results}
                    
                    logger.info(f"   🔗 Found {len(kg_memory_ids)} KG-connected memories")
                    
                    # Fetch these memories from ArangoDB
                    kg_memories_query = """
                    FOR memory IN memories
                        FILTER memory._key IN @memory_ids
                        FILTER memory.user_id == @user_id
                        RETURN {
                            memory_id: memory._key,
                            content: memory.content,
                            ego_score: memory.ego_score,
                            tier: memory.tier,
                            created_at: memory.created_at,
                            observed_at: memory.observed_at,
                            metadata: memory.metadata,
                            source: 'kg_boost'
                        }
                    """
                    cursor = self.arango_db.aql.execute(
                        kg_memories_query,
                        bind_vars={"memory_ids": kg_memory_ids, "user_id": user_id}
                    )
                    kg_memories = list(cursor)
                    
                    # Merge with vector results, using LLM relevance scores
                    for kg_mem in kg_memories:
                        mem_id = kg_mem["memory_id"]
                        llm_relevance = kg_relevance_map.get(mem_id, 0.85)
                        
                        # Check if already in vector results
                        existing = next((m for m in all_memories if m["memory_id"] == mem_id), None)
                        if existing:
                            # Boost using LLM relevance score
                            existing["relevance_score"] = min(1.0, existing["relevance_score"] + (llm_relevance * 0.4))
                            existing["source"] = "vector+kg_boost"
                            logger.debug(f"   ⬆️  Boosted memory {mem_id[:12]} (LLM relevance: {llm_relevance:.2f})")
                        else:
                            # Add as new result with LLM relevance score
                            kg_mem["relevance_score"] = llm_relevance
                            all_memories.append(kg_mem)
                            logger.debug(f"   ➕ Added KG memory {mem_id[:12]} (LLM relevance: {llm_relevance:.2f})")
                    
                    logger.info(f"   🔗 KG boosting: +{len(kg_memories)} memories, {sum(1 for m in all_memories if 'kg_boost' in m.get('source', ''))} boosted")
            except Exception as e:
                logger.warning(f"KG boosting failed: {e}", exc_info=True)
        
        # PPR-based graph retrieval (DEPRECATED - dead code path, kept for reference)
        if False and use_ppr and self.ppr_retrieval and entity_names:
            try:
                entity_ids = await self._resolve_entity_ids(user_id, entity_names)
                logger.info(f"🔍 Resolved entity IDs: {entity_ids} (from names: {entity_names})")
                
                if entity_ids:
                    logger.info(f"🔍 PPR retrieval with {len(entity_ids)} seed entities: {entity_ids}")
                    
                    # Run PPR
                    ppr_result = await self.ppr_retrieval.retrieve(
                        user_id=user_id,
                        seed_entities=entity_ids[:3],
                        max_hops=2,
                        top_k=max_memories * 2
                    )
                    
                    # Collect memory IDs from edges
                    memory_ids = set()
                    for edge in ppr_result.edges:
                        supporting_memories = edge.get('supporting_memories', [])
                        if supporting_memories:
                            for mem in supporting_memories:
                                memory_ids.add(mem['memory_id'])
                    
                    if not memory_ids:
                        for edge in ppr_result.edges:
                            source_memory_ids = edge.get('source_memory_ids', [])
                            if source_memory_ids:
                                memory_ids.update(source_memory_ids)
                    
                    if not memory_ids:
                        entity_ids_in_graph = {n['entity_id'] for n in ppr_result.nodes}
                        if entity_ids_in_graph:
                            entity_memory_query = """
                            FOR memory IN memories
                                FILTER memory.user_id == @user_id
                                FILTER memory.graph_extraction != null
                                FILTER LENGTH(
                                    FOR extracted_entity IN memory.graph_extraction.extracted_entities
                                        FILTER extracted_entity.entity_id IN @entity_ids
                                        RETURN 1
                                ) > 0
                                LIMIT 10
                                RETURN memory._key
                            """
                            mem_cursor = self.arango_db.aql.execute(
                                entity_memory_query,
                                bind_vars={'user_id': user_id, 'entity_ids': list(entity_ids_in_graph)}
                            )
                            memory_ids.update(list(mem_cursor))
                    
                    if memory_ids:
                        memories_query = """
                        FOR memory IN memories
                            FILTER memory._key IN @memory_ids
                            FILTER memory.user_id == @user_id
                            RETURN {
                                memory_id: memory._key,
                                content: memory.content,
                                ego_score: memory.ego_score,
                                tier: memory.tier,
                                created_at: memory.created_at,
                                observed_at: memory.observed_at,
                                metadata: memory.metadata,
                                source: 'ppr_graph'
                            }
                        """
                        cursor = self.arango_db.aql.execute(
                            memories_query,
                            bind_vars={'memory_ids': list(memory_ids), 'user_id': user_id}
                        )
                        ppr_memories = list(cursor)
                        all_memories.extend(ppr_memories)
                        logger.info(f"   Retrieved {len(ppr_memories)} memories via PPR")
                else:
                    logger.debug("No entity IDs resolved for PPR")
                    
            except Exception as e:
                logger.warning(f"PPR retrieval failed: {e}", exc_info=True)
        
        # 2. Fallback: Simple text-based retrieval from ArangoDB if PPR found nothing
        if len(all_memories) == 0:
            try:
                logger.info("🔄 PPR returned no results, falling back to recent high-ego memories")
                
                # Fallback strategy: Get recent high-ego memories containing query keywords
                query_keywords = query.lower().split()[:5]  # Use first 5 words
                
                fallback_query = """
                FOR memory IN memories
                    FILTER memory.user_id == @user_id
                    FILTER memory.tier IN [1, 2]
                    FILTER memory.ego_score >= 0.5
                    SORT memory.created_at DESC
                    LIMIT @limit
                    RETURN {
                        memory_id: memory._key,
                        content: memory.content,
                        ego_score: memory.ego_score,
                        tier: memory.tier,
                        created_at: memory.created_at,
                        observed_at: memory.observed_at,
                        metadata: memory.metadata,
                        source: 'fallback_recent'
                    }
                """
                
                cursor = self.arango_db.aql.execute(
                    fallback_query,
                    bind_vars={
                        'user_id': user_id,
                        'limit': max_memories * 2
                    }
                )
                
                fallback_memories = list(cursor)
                
                # Simple keyword matching for relevance
                scored_memories = []
                for mem in fallback_memories:
                    content_lower = mem['content'].lower()
                    keyword_matches = sum(1 for kw in query_keywords if kw in content_lower)
                    if keyword_matches > 0:
                        mem['relevance_score'] = keyword_matches
                        scored_memories.append(mem)
                
                # Sort by relevance, then ego score
                scored_memories.sort(key=lambda m: (m.get('relevance_score', 0), m.get('ego_score', 0)), reverse=True)
                
                all_memories.extend(scored_memories[:max_memories])
                logger.info(f"   Retrieved {len(scored_memories[:max_memories])} memories via fallback")
                
            except Exception as e:
                logger.warning(f"Fallback retrieval failed: {e}", exc_info=True)
        
        # 3. Deduplicate and rank (prefer vector relevance, then ego)
        unique_memories = {}
        for memory in all_memories:
            mem_id = memory['memory_id']
            if mem_id not in unique_memories:
                unique_memories[mem_id] = memory
            else:
                # Keep the one with higher relevance_score if present
                existing = unique_memories[mem_id]
                new_rel = memory.get('relevance_score') or 0
                existing_rel = existing.get('relevance_score') or 0
                if new_rel > existing_rel:
                    unique_memories[mem_id] = memory
        
        # Sort by relevance (vector) first, then ego score
        def _rank_key(m):
            rel = m.get('relevance_score')
            ego = m.get('ego_score', 0.0)
            return (rel if rel is not None else 0.0, ego)
        
        ranked_memories = sorted(
            unique_memories.values(),
            key=_rank_key,
            reverse=True
        )
        
        # Return top N
        result = ranked_memories[:max_memories]
        
        if result:
            logger.info(f"✅ Retrieved {len(result)} relevant memories for query")
        else:
            logger.info("No relevant memories found")
        
        return result
    
    def get_localized_relations_context(
        self,
        user_id: str,
        retrieved_memories: List[Dict[str, Any]],
        limit: int = 30
    ) -> str:
        """
        Get LOCALIZED user relations based on entity tags from vector search results.
        
        Entities are extracted ONCE at ingestion and stored in Qdrant payload.
        At retrieval, vector search returns memories with entity tags.
        We collect those entities and filter the KG to the relevant subgraph.
        
        Args:
            user_id: User ID
            retrieved_memories: Memories from vector search (with 'entities' in payload)
            limit: Max relations to return
        
        Returns:
            Formatted string of relevant relations for LLM context
        """
        if not self.knowledge_graph_store:
            return ""
        
        try:
            # Collect entity names from retrieved memory payloads
            matched_entities = set()
            matched_entities.add("user")  # Always include "user"
            
            for mem in retrieved_memories:
                entities = mem.get('entities', [])
                for ent in entities:
                    if isinstance(ent, str) and ent:
                        matched_entities.add(ent.lower())
                    elif isinstance(ent, dict):
                        name = ent.get('name', '')
                        if name:
                            matched_entities.add(name.lower())
            
            # Get full KG size for comparison
            full_kg = self.knowledge_graph_store.get_user_relations(user_id=user_id, entity_names=None, limit=500)
            full_kg_size = len(full_kg)
            
            if len(matched_entities) <= 1:
                # Only "user" found, no specific entities - fall back to full KG
                logger.info(f"   🔍 Localized KG: No entity tags found, falling back to full KG ({full_kg_size} relations)")
                return self.get_user_relations_context(user_id=user_id, limit=limit)
            
            logger.info(f"   🔍 Entity tags from vector search: {sorted(matched_entities)}")
            
            # Filter KG to only relations involving matched entities
            relations = self.knowledge_graph_store.get_user_relations(
                user_id=user_id,
                entity_names=list(matched_entities),
                limit=limit
            )
            
            if not relations:
                logger.info(f"   🔍 Localized KG: {len(matched_entities)} entities matched but no relations found (full KG: {full_kg_size} relations)")
                return ""
            
            logger.info(f"   🔍 Localized KG: {len(matched_entities)} entities → {len(relations)} relations (filtered from {full_kg_size} total)")
            logger.info(f"   📊 Localized Relations:")
            for rel in relations[:15]:
                logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel.get('confidence', 0.5):.2f})")
            if len(relations) > 15:
                logger.info(f"      ... and {len(relations) - 15} more relations")
            
            lines = ["Known relationships:"]
            for rel in relations:
                subject = rel['subject']
                predicate = rel['predicate'].replace('_', ' ')
                obj = rel['object']
                confidence = rel.get('confidence', 0.5)
                lines.append(f"  - {subject} {predicate} {obj} (confidence: {confidence:.2f})")
            
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to get localized relation context: {e}")
            return ""

    def get_user_relations_context(
        self,
        user_id: str,
        entity_names: Optional[List[str]] = None,
        limit: int = 50
    ) -> str:
        """
        Get user's relations formatted as context for LLM (full KG, no localization).
        Used by KG Maintenance Agent which needs the full KG view.
        Also used as fallback when no entity tags are available.
        
        Args:
            user_id: User ID
            entity_names: Optional entity names to filter relations
            limit: Max relations to return
        
        Returns:
            Formatted string of relations for LLM context
        """
        if not self.knowledge_graph_store:
            return ""
        
        try:
            relations = self.knowledge_graph_store.get_user_relations(
                user_id=user_id,
                entity_names=entity_names,
                limit=limit
            )
            
            if not relations:
                return ""
            
            # Format relations for LLM with confidence scores
            lines = ["Known relationships:"]
            for rel in relations:
                subject = rel['subject']
                predicate = rel['predicate'].replace('_', ' ')
                obj = rel['object']
                confidence = rel.get('confidence', 0.5)
                lines.append(f"  - {subject} {predicate} {obj} (confidence: {confidence:.2f})")
            
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to get relation context: {e}")
            return ""
    
    def _count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in conversation history"""
        total = 0
        for msg in messages:
            content = msg.get('content', '')
            try:
                total += len(self.encoding.encode(content))
            except Exception as e:
                # Fallback: rough estimate (4 chars per token)
                total += len(content) // 4
        return total

