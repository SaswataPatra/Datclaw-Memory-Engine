"""
DAPPY - Async Prioritized Consolidation Worker
Background consolidation with priority queues (HIGH/MED/LOW)
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import logging
import json

from core.event_bus import Event, EventBus
from core.scoring.ego_scorer import TemporalEgoScorer
from arango import ArangoClient
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class ConsolidationWorker:
    """
    Async consolidation with priority queues
    
    HIGH: <100ms (single item, urgent)
    MED/LOW: Batched (128 items, 30s window)
    
    Responsibilities:
    - Read from Redis consolidation queues
    - Cluster related memories
    - Create parent nodes (summaries)
    - Check spaced repetition triggers
    - Publish memory.upsert events
    """
    
    def __init__(
        self,
        redis_client,
        arango_client,
        qdrant_client,
        config: Dict[str, Any],
        ego_scorer: TemporalEgoScorer,
        event_bus: EventBus
    ):
        self.redis = redis_client
        self.arango = arango_client
        self.qdrant = qdrant_client
        self.config = config
        self.ego_scorer = ego_scorer
        self.event_bus = event_bus
        
        # Consolidation config
        consolidation_config = config.get('consolidation', {})
        self.enabled = consolidation_config.get('enabled', True)
        
        # Priority thresholds
        self.high_threshold = consolidation_config.get('high_priority_threshold', 0.75)
        self.med_threshold = consolidation_config.get('med_priority_threshold', 0.45)
        
        # Worker settings
        fast_worker_config = consolidation_config.get('fast_worker', {})
        self.fast_worker_enabled = fast_worker_config.get('enabled', True)
        self.fast_worker_target_latency_ms = fast_worker_config.get('target_latency_ms', 100)
        self.fast_worker_concurrency = fast_worker_config.get('concurrency', 5)
        
        batch_worker_config = consolidation_config.get('batch_worker', {})
        self.batch_worker_enabled = batch_worker_config.get('enabled', True)
        self.batch_size = batch_worker_config.get('batch_size', 128)
        self.batch_wait_seconds = batch_worker_config.get('wait_seconds', 30)
        self.batch_worker_concurrency = batch_worker_config.get('concurrency', 3)
        
        # Backpressure
        self.max_queue_depth = consolidation_config.get('max_queue_depth', 2000)
        self.ttl_bump_on_backlog = consolidation_config.get('ttl_bump_on_backlog', True)
        
        # Spaced repetition windows
        self.spacing_windows = consolidation_config.get('spacing_windows', [])
        
        # Stream keys
        self.high_queue = "consolidation:HIGH"
        self.med_queue = "consolidation:MED"
        self.low_queue = "consolidation:LOW"
        
        # Worker tasks
        self._fast_worker_tasks = []
        self._batch_worker_tasks = []
        self._running = False
        
        logger.info("Consolidation Worker initialized", extra={
            "fast_worker": self.fast_worker_enabled,
            "batch_worker": self.batch_worker_enabled,
            "high_threshold": self.high_threshold
        })
    
    async def start(self):
        """Start consolidation workers"""
        if not self.enabled:
            logger.warning("Consolidation worker is disabled")
            return
        
        self._running = True
        
        # Start fast workers (HIGH priority)
        if self.fast_worker_enabled:
            for i in range(self.fast_worker_concurrency):
                task = asyncio.create_task(self._fast_worker(worker_id=i))
                self._fast_worker_tasks.append(task)
            logger.info(f"Started {self.fast_worker_concurrency} fast workers")
        
        # Start batch workers (MED/LOW priority)
        if self.batch_worker_enabled:
            for i in range(self.batch_worker_concurrency):
                task = asyncio.create_task(self._batch_worker(worker_id=i))
                self._batch_worker_tasks.append(task)
            logger.info(f"Started {self.batch_worker_concurrency} batch workers")
        
        logger.info("Consolidation workers started")
    
    async def stop(self):
        """Stop consolidation workers"""
        self._running = False
        
        # Cancel all tasks
        for task in self._fast_worker_tasks + self._batch_worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("Consolidation workers stopped")
    
    async def _fast_worker(self, worker_id: int):
        """
        Fast worker for HIGH priority items
        Target latency: <100ms
        """
        logger.info(f"Fast worker {worker_id} started")
        
        while self._running:
            try:
                # Read single item from HIGH queue (blocking with timeout)
                items = await self.redis.xreadgroup(
                    groupname="consolidation",
                    consumername=f"fast-{worker_id}",
                    streams={self.high_queue: '>'},
                    count=1,
                    block=1000  # 1 second timeout
                )
                
                if not items:
                    continue
                
                # Process item
                for stream, message_list in items:
                    for message_id, item_data in message_list:
                        start_time = datetime.now(timezone.utc)
                        
                        try:
                            await self._consolidate_item(item_data)
                            
                            # ACK message
                            await self.redis.xack(self.high_queue, "consolidation", message_id)
                            
                            # Track latency
                            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                            
                            if latency_ms > self.fast_worker_target_latency_ms:
                                logger.warning(
                                    f"Fast worker latency exceeded target",
                                    extra={"latency_ms": latency_ms, "target_ms": self.fast_worker_target_latency_ms}
                                )
                            
                        except Exception as e:
                            logger.error(f"Fast worker error: {e}", exc_info=True)
                            # ACK anyway to prevent retry loop
                            await self.redis.xack(self.high_queue, "consolidation", message_id)
            
            except asyncio.CancelledError:
                logger.info(f"Fast worker {worker_id} cancelled")
                break
            
            except Exception as e:
                # Suppress expected NOGROUP errors (queues lazy-created on first publish)
                if "NOGROUP" in str(e):
                    await asyncio.sleep(5)
                    continue
                logger.error(f"Fast worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info(f"Fast worker {worker_id} stopped")
    
    async def _batch_worker(self, worker_id: int):
        """
        Batch worker for MED/LOW priority items
        Batches up to 128 items or waits 30 seconds
        """
        logger.info(f"Batch worker {worker_id} started")
        
        while self._running:
            try:
                # Read batch from MED queue
                med_items = await self.redis.xreadgroup(
                    groupname="consolidation",
                    consumername=f"batch-{worker_id}",
                    streams={self.med_queue: '>'},
                    count=self.batch_size,
                    block=self.batch_wait_seconds * 1000
                )
                
                # Read batch from LOW queue
                low_items = await self.redis.xreadgroup(
                    groupname="consolidation",
                    consumername=f"batch-{worker_id}",
                    streams={self.low_queue: '>'},
                    count=self.batch_size,
                    block=100  # Quick check
                )
                
                # Combine batches
                all_items = (med_items or []) + (low_items or [])
                
                if not all_items:
                    continue
                
                # Process batch
                batch_data = []
                message_ids = {}
                
                for stream, message_list in all_items:
                    for message_id, item_data in message_list:
                        batch_data.append(item_data)
                        
                        if stream not in message_ids:
                            message_ids[stream] = []
                        message_ids[stream].append(message_id)
                
                if batch_data:
                    try:
                        await self._consolidate_batch(batch_data)
                        
                        # ACK all messages
                        for stream, ids in message_ids.items():
                            for msg_id in ids:
                                await self.redis.xack(stream, "consolidation", msg_id)
                        
                        logger.info(f"Batch worker {worker_id} processed {len(batch_data)} items")
                        
                    except Exception as e:
                        logger.error(f"Batch consolidation error: {e}", exc_info=True)
                        # ACK to prevent retry loop
                        for stream, ids in message_ids.items():
                            for msg_id in ids:
                                await self.redis.xack(stream, "consolidation", msg_id)
            
            except asyncio.CancelledError:
                logger.info(f"Batch worker {worker_id} cancelled")
                break
            
            except Exception as e:
                # Suppress expected NOGROUP errors (queues lazy-created on first publish)
                if "NOGROUP" in str(e):
                    await asyncio.sleep(5)
                    continue
                logger.error(f"Batch worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info(f"Batch worker {worker_id} stopped")
    
    def _determine_tier(self, ego_score: float, confidence: float) -> int:
        """
        Determine memory tier based on ego score and confidence
        
        Returns:
            1: Core memory (Tier 1) - permanent, high ego + high confidence
            0.5: Shadow tier - needs user confirmation
            2: Long-term (Tier 2) - weeks-months retention
            3: Short-term (Tier 3) - hours-days retention
            None: Forget - too low value to store
        """
        ego_config = self.config.get('ego_scoring', {})
        tier1_threshold = ego_config.get('tier1_threshold', 0.75)
        tier2_threshold = ego_config.get('tier2_threshold', 0.45)
        tier3_threshold = ego_config.get('tier3_threshold', 0.25)
        
        shadow_config = self.config.get('shadow_tier', {})
        shadow_confidence_threshold = shadow_config.get('confidence_threshold', 0.85)
        
        # Core memory: high ego + high confidence
        if ego_score >= tier1_threshold and confidence >= shadow_confidence_threshold:
            return 1
        
        # Shadow tier: high ego but lower confidence (needs user confirmation)
        elif ego_score >= tier1_threshold and confidence < shadow_confidence_threshold:
            return 0.5
        
        # Long-term memory: medium ego
        elif ego_score >= tier2_threshold:
            return 2
        
        # Short-term memory: low ego
        elif ego_score >= tier3_threshold:
            return 3
        
        # Too low value - forget
        else:
            return None
    
    async def _consolidate_single(self, data: Dict[str, Any]):
        """
        Consolidate single memory (alias for _consolidate_item for backwards compatibility)
        """
        await self._consolidate_item(data)
    
    async def _consolidate_item(self, item_data: Dict[str, Any]):
        """Consolidate single item (HIGH priority)"""
        
        user_id = item_data.get('user_id')
        tier4_key = item_data.get('tier4_key')
        
        # Fetch full memory from Tier 4
        memory_data = await self.redis.get(tier4_key)
        
        if not memory_data:
            logger.warning(f"Memory not found in Tier 4: {tier4_key}")
            return
        
        # Parse memory
        try:
            memory = eval(memory_data) if isinstance(memory_data, str) else memory_data
        except:
            logger.error(f"Failed to parse memory: {tier4_key}")
            return
        
        # Calculate ego score
        ego_result = self.ego_scorer.calculate(memory)
        
        # Determine tier
        confidence = memory.get('llm_confidence', 0.85)
        tier = self._determine_tier(ego_result.ego_score, confidence)
        
        # If tier is None, memory should be forgotten (too low value)
        if tier is None:
            logger.info(f"Memory below threshold, forgetting: {tier4_key}")
            return
        
        # Check if spaced repetition consolidation is needed
        spacing_trigger = await self._check_spacing_trigger(memory)
        
        # Create memory node in ArangoDB and publish event
        await self._create_memory_node(user_id, memory, ego_result, tier, spacing_trigger)
    
    async def _consolidate_batch(self, batch_data: List[Dict[str, Any]]):
        """Consolidate batch of items (MED/LOW priority)"""
        
        for item in batch_data:
            await self._consolidate_item(item)
    
    async def _check_spacing_trigger(self, memory: Dict[str, Any]) -> Optional[str]:
        """
        Check if memory should be consolidated based on spaced repetition
        
        Windows: 1 day, 3 days, 7 days, 30 days
        """
        
        observed_at = memory.get('observed_at')
        if not observed_at:
            return None
        
        if isinstance(observed_at, str):
            observed_at = datetime.fromisoformat(observed_at)
        
        age_hours = (datetime.now(timezone.utc) - observed_at).total_seconds() / 3600
        
        # Check each spacing window
        for window in self.spacing_windows:
            # Handle both int format ([24, 72]) and dict format ([{after_hours: 24}])
            if isinstance(window, int):
                window_hours = window
                tolerance_hours = 12
                window_name = f"{window}h"
            else:
                window_hours = window['after_hours']
                tolerance_hours = window.get('tolerance_hours', 12)
                window_name = window.get('name', f"{window_hours}h")
            
            # Check if memory falls within this window
            if (window_hours - tolerance_hours) <= age_hours <= (window_hours + tolerance_hours):
                logger.info(
                    f"Spacing trigger: {window_name}",
                    extra={"memory_id": memory.get('memory_id'), "age_hours": age_hours}
                )
                return window_name
        
        return None
    
    async def _publish_memory_upsert_event(
        self,
        node_id: str,
        user_id: str,
        tier: int,
        ego_score: float,
        content: str,
        metadata: Dict[str, Any]
    ):
        """
        Publish memory.upsert event
        
        The event will be consumed by:
        - ArangoDB consumer (writes to graph)
        - Qdrant consumer (indexes vector)
        """
        
        node_payload = {
            "node_id": node_id,
            "user_id": user_id,
            "tier": tier,
            "ego_score": ego_score,
            "content": content,
            "summary": content[:500],  # First 500 chars
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": 1
        }
        
        # Publish memory.upsert event
        event = Event(
            topic="memory.upsert",
            event_type="consolidation.create",
            payload=node_payload
        )
        
        await self.event_bus.publish(event)
        
        logger.info(
            f"Published memory.upsert event",
            extra={
                "memory_id": node_id,
                "tier": tier,
                "ego_score": ego_score
            }
        )
    
    async def _create_memory_node(
        self,
        user_id: str,
        memory: Dict[str, Any],
        ego_result: Any,
        tier: int,
        spacing_trigger: Optional[str]
    ):
        """
        Create memory node and publish memory.upsert event
        
        The event will be consumed by:
        - ArangoDB consumer (writes to graph)
        - Qdrant consumer (indexes vector)
        """
        
        memory_id = memory.get('memory_id', f"mem_{datetime.now(timezone.utc).timestamp()}")
        content = memory.get('content', '')
        
        metadata = {
            **memory.get('metadata', {}),
            "spacing_trigger": spacing_trigger,
            "consolidation_timestamp": datetime.now(timezone.utc).isoformat(),
            "observed_at": memory.get('observed_at', datetime.now(timezone.utc)).isoformat() if isinstance(memory.get('observed_at'), datetime) else memory.get('observed_at'),
            "sequence": memory.get('sequence', 0),
            "ego_components": ego_result.components.to_dict()
        }
        
        # Publish memory.upsert event
        await self._publish_memory_upsert_event(
            node_id=memory_id,
            user_id=user_id,
            tier=tier,
            ego_score=ego_result.ego_score,
            content=content,
            metadata=metadata
        )
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get consolidation queue statistics"""
        
        try:
            high_info = await self.redis.xinfo_stream(self.high_queue)
            med_info = await self.redis.xinfo_stream(self.med_queue)
            low_info = await self.redis.xinfo_stream(self.low_queue)
            
            return {
                "high_queue": {
                    "length": high_info.get('length', 0),
                    "groups": high_info.get('groups', 0)
                },
                "med_queue": {
                    "length": med_info.get('length', 0),
                    "groups": med_info.get('groups', 0)
                },
                "low_queue": {
                    "length": low_info.get('length', 0),
                    "groups": low_info.get('groups', 0)
                },
                "total_backlog": (
                    high_info.get('length', 0) +
                    med_info.get('length', 0) +
                    low_info.get('length', 0)
                )
            }
        
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"error": str(e)}

