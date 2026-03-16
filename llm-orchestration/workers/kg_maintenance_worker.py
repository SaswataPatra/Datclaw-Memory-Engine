"""
KG Maintenance Worker
Listens for ego scoring completion events and triggers KG maintenance.

Flow:
1. Listen for 'ego_scoring_complete' events
2. Fetch the memory and its relations
3. Run KG maintenance agent to detect contradictions
4. Update the KG (remove old relations, increment supporting mentions)
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class KGMaintenanceWorker:
    """
    Background worker that maintains the knowledge graph after ego scoring completes.
    """

    def __init__(
        self,
        event_bus,
        arango_db,
        kg_maintenance_agent,
        config: Optional[dict] = None
    ):
        self.event_bus = event_bus
        self.arango_db = arango_db
        self.kg_maintenance_agent = kg_maintenance_agent
        self.config = config or {}
        self.running = False
        self._task = None
        self.event_queue = asyncio.Queue()
        self.processor_task = None
        self.batch_size = 5  # Process 5 events in parallel

    async def start(self):
        """Start the worker."""
        if self.running:
            logger.warning("KG Maintenance Worker already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._run())
        self.processor_task = asyncio.create_task(self._process_events_in_parallel())
        logger.info("✅ KG Maintenance Worker started (parallel batch processing enabled)")

    async def stop(self):
        """Stop the worker."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        logger.info("✅ KG Maintenance Worker stopped")

    async def _run(self):
        """Main worker loop - uses EventBus subscription."""
        logger.info("🔧 KG Maintenance Worker listening for ego_scoring_complete events...")
        
        # Subscribe using EventBus (Redis Streams, not PubSub)
        await self.event_bus.subscribe(
            topic="events:ego_scoring_complete",
            handler=self._handle_event,
            consumer_group="kg-maintenance-worker"
        )
        
        # Keep running until stopped
        try:
            while self.running:
                await asyncio.sleep(1)
        finally:
            await self.event_bus.unsubscribe("events:ego_scoring_complete")

    async def _handle_event(self, event):
        """
        Handle ego_scoring_complete event by queuing it for parallel processing.
        
        Args:
            event: Event object from EventBus
        """
        try:
            await self.event_queue.put(event)
        except Exception as e:
            logger.error(f"Error queuing event: {e}", exc_info=True)
    
    async def _process_events_in_parallel(self):
        """
        Process events from the queue in parallel batches.
        Drains up to batch_size events at a time and processes them concurrently.
        """
        logger.info(f"🚀 Parallel event processor started (batch_size={self.batch_size})")
        
        while self.running:
            try:
                # Collect a batch of events (up to batch_size)
                batch = []
                try:
                    # Wait for at least one event (with timeout)
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    batch.append(event)
                    
                    # Try to get more events without blocking (up to batch_size)
                    while len(batch) < self.batch_size:
                        try:
                            event = self.event_queue.get_nowait()
                            batch.append(event)
                        except asyncio.QueueEmpty:
                            break
                            
                except asyncio.TimeoutError:
                    # No events available, continue loop
                    continue
                
                if not batch:
                    continue
                
                # Process batch in parallel
                logger.info(f"📦 Processing batch of {len(batch)} KG maintenance events...")
                tasks = [self._process_single_event(event) for event in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any errors
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch processing event {idx}: {result}")
                
                logger.info(f"✅ Batch complete: {len(batch)} events processed")
                
            except asyncio.CancelledError:
                logger.info("Parallel event processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in parallel event processor: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _process_single_event(self, event):
        """
        Process a single ego_scoring_complete event.
        
        Args:
            event: Event object from EventBus
        """
        try:
            # Event is already an Event object from EventBus, extract payload
            payload = event.payload
            
            user_id = payload.get("user_id")
            memory_id = payload.get("memory_id")
            memory_content = payload.get("memory_content", "")
            ego_score = payload.get("ego_score", 0.5)
            new_relations = payload.get("new_relations", [])
            
            if not user_id or not memory_id:
                logger.warning(f"Invalid event payload: missing user_id or memory_id")
                return
            
            logger.debug(f"🔔 Processing: memory={memory_id}, ego={ego_score:.2f}")
            
            # If memory_content not in payload, fetch from ArangoDB
            if not memory_content:
                memory = await self._fetch_memory(memory_id)
                if not memory:
                    logger.warning(f"Memory {memory_id} not found in ArangoDB")
                    return
                memory_content = memory.get("content", "")
            
            # Fetch embedding from Qdrant for contradiction signal detection
            embedding = await self._fetch_embedding(memory_id)
            
            # Run KG maintenance (ALWAYS, even if new_relations is empty)
            # The agent will analyze the entire KG against this memory
            result = await self.kg_maintenance_agent.process_memory(
                user_id=user_id,
                memory_id=memory_id,
                memory_content=memory_content,
                new_relations=new_relations,
                ego_score=ego_score,
                embedding=embedding
            )
            
            logger.debug(
                f"✅ KG maintenance complete for {memory_id}: "
                f"{result['contradictions_found']} contradictions, "
                f"{result['relations_updated']} updated, "
                f"{result['relations_removed']} removed"
            )
            
        except Exception as e:
            logger.error(f"Error processing event for memory {payload.get('memory_id', 'unknown')}: {e}", exc_info=True)

    async def _fetch_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Fetch memory from ArangoDB."""
        try:
            memories_collection = self.arango_db.collection("memories")
            memory = memories_collection.get(memory_id)
            return memory
        except Exception as e:
            logger.warning(f"Failed to fetch memory {memory_id}: {e}")
            return None

    async def _fetch_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Fetch embedding from Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            qdrant_config = self.config.get("qdrant", {})
            qdrant_client = QdrantClient(
                host=qdrant_config.get("host", "localhost"),
                port=qdrant_config.get("port", 6333)
            )
            collection_name = qdrant_config.get("collection_name", "dappy_memories")
            
            # Search for the point by memory_id in payload
            results = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="node_id",
                            match=MatchValue(value=memory_id)
                        )
                    ]
                ),
                limit=1,
                with_vectors=True
            )
            
            points, _ = results
            if points and len(points) > 0:
                return points[0].vector
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch embedding for {memory_id}: {e}")
            return None

    async def _fetch_memory_relations(self, user_id: str, memory_id: str) -> List[Dict[str, Any]]:
        """Fetch relations associated with a memory."""
        try:
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.memory_id == @memory_id
                RETURN {
                    subject: rel.subject,
                    predicate: rel.predicate,
                    object: rel.object,
                    confidence: rel.confidence
                }
            """
            
            cursor = self.arango_db.aql.execute(
                query,
                bind_vars={"user_id": user_id, "memory_id": memory_id}
            )
            
            return list(cursor)
            
        except Exception as e:
            logger.warning(f"Failed to fetch memory relations: {e}")
            return []
