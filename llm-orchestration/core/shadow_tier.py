"""
DAPPY - Shadow Tier (Tier 0.5)
Safety layer for core memory (Tier 1) with user confirmation
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import json

from core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


@dataclass
class ShadowMemory:
    """
    Memory pending promotion to Tier 1 (core memory)
    
    Requires user confirmation before becoming permanent
    """
    shadow_id: str
    node_id: str
    user_id: str
    content: str
    summary: str
    ego_score: float
    confidence: float
    sources: List[str]  # Source message IDs
    model_version: str
    created_at: str
    expires_at: str
    status: str  # 'pending' | 'approved' | 'rejected' | 'auto_promoted'
    auto_promote_after_days: int = 7
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShadowMemory':
        return cls(**data)


class ShadowTier:
    """
    Shadow Tier (Tier 0.5) Manager
    
    Responsibilities:
    - Hold high-ego memories pending user confirmation
    - Auto-promote after N days if no response
    - Provide provenance and revertability
    - Track approval history
    """
    
    def __init__(self, redis_client, config: Dict[str, Any], event_bus: EventBus):
        self.redis = redis_client
        self.config = config
        self.event_bus = event_bus
        
        # Shadow tier config
        shadow_config = config.get('shadow_tier', {})
        self.enabled = shadow_config.get('enabled', True)
        self.auto_promote_after_days = shadow_config.get('auto_promote_after_days', 7)
        self.require_confirmation_threshold = shadow_config.get('require_confirmation_threshold', 0.75)
        self.confidence_threshold = shadow_config.get('confidence_threshold', 0.85)
        
        # Redis key prefix
        self.shadow_prefix = "shadow_tier:"
        self.pending_prefix = f"{self.shadow_prefix}pending:"
        self.approved_prefix = f"{self.shadow_prefix}approved:"
        self.rejected_prefix = f"{self.shadow_prefix}rejected:"
        
        # Alias for backward compatibility with tests
        self.auto_promote_days = self.auto_promote_after_days
        
        logger.info(
            "Shadow Tier initialized",
            extra={
                "enabled": self.enabled,
                "auto_promote_after_days": self.auto_promote_after_days,
                "confirmation_threshold": self.require_confirmation_threshold
            }
        )
    
    async def should_use_shadow_tier(
        self,
        ego_score: float,
        confidence: float
    ) -> bool:
        """
        Determine if memory should go through shadow tier.
        
        IMPORTANT: Shadow Tier is ONLY for Tier 1 memories (ego_score >= 0.8)
        
        Rules:
        - If ego_score >= 0.8 (Tier 1) → ALWAYS route to Shadow Tier for user confirmation
        - If ego_score < 0.8 (Tier 2/3/4) → Skip Shadow Tier, store directly
        
        Note: We removed "auto-promote" logic. ALL Tier 1 memories require user confirmation.
        High confidence just means we're more certain, but user still needs to approve.
        """
        
        if not self.enabled:
            logger.debug("   Shadow Tier disabled")
            return False
        
        # Only Tier 1 memories (ego_score >= 0.8) go to Shadow Tier
        # This threshold MUST match MLScorer._determine_tier()
        TIER_1_THRESHOLD = 0.8
        
        if ego_score >= TIER_1_THRESHOLD:
            logger.info(
                f"   ✅ Tier 1 memory detected (ego_score={ego_score:.4f} >= {TIER_1_THRESHOLD}) "
                f"→ Routing to Shadow Tier for user confirmation"
            )
            return True  # Always use shadow tier for Tier 1
        else:
            logger.debug(
                f"   ⏭️  Tier 2/3/4 memory (ego_score={ego_score:.4f} < {TIER_1_THRESHOLD}) "
                f"→ Skipping Shadow Tier"
            )
            return False
    
    async def propose_core_memory(
        self,
        memory: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Propose a memory for core tier with clarification question
        
        Returns:
            (clarification_id, clarification_question)
        """
        # Extract memory details
        node_id = memory.get('node_id', f"mem_{datetime.utcnow().timestamp()}")
        user_id = memory.get('user_id', 'unknown')
        summary = memory.get('summary', '')
        ego_score = memory.get('ego_score', 0.0)
        confidence = memory.get('confidence', 0.0)
        sources = memory.get('sources', [])
        
        # Add to shadow tier
        shadow_memory = await self.add_to_shadow_tier(
            node_id=node_id,
            user_id=user_id,
            content=memory.get('content', summary),
            summary=summary,
            ego_score=ego_score,
            confidence=confidence,
            sources=[str(s) for s in sources],  # Convert to strings
            model_version=memory.get('model_version', 'v1')
        )
        
        # Generate clarification question (if LLM available)
        question = f"I noticed you mentioned: '{summary}'. Should I remember this as a core preference?"
        
        # Try to generate a better question with LLM if available
        if hasattr(self, 'llm') and self.llm:
            try:
                from openai import AsyncOpenAI
                response = await self.llm.chat.completions.create(
                    model='gpt-4-turbo-preview',
                    messages=[
                        {"role": "system", "content": "Generate a short, natural clarification question for a memory."},
                        {"role": "user", "content": f"Memory: {summary}"}
                    ],
                    max_tokens=50
                )
                question = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Failed to generate LLM clarification: {e}")
        
        return shadow_memory.shadow_id, question
    
    async def add_to_shadow_tier(
        self,
        node_id: str,
        user_id: str,
        content: str,
        summary: str,
        ego_score: float,
        confidence: float,
        sources: List[str],
        model_version: str
    ) -> ShadowMemory:
        """
        Add memory to shadow tier pending confirmation
        """
        
        shadow_id = f"shadow_{node_id}"
        
        # Calculate expiration
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(days=self.auto_promote_after_days)
        
        # Create shadow memory
        shadow_memory = ShadowMemory(
            shadow_id=shadow_id,
            node_id=node_id,
            user_id=user_id,
            content=content,
            summary=summary,
            ego_score=ego_score,
            confidence=confidence,
            sources=sources,
            model_version=model_version,
            created_at=created_at.isoformat(),
            expires_at=expires_at.isoformat(),
            status='pending',
            auto_promote_after_days=self.auto_promote_after_days
        )
        
        # Store in ArangoDB if available (for persistence)
        if hasattr(self, 'arango') and self.arango:
            try:
                shadow_collection = self.arango.collection('shadow_memories')
                shadow_collection.insert(shadow_memory.to_dict())
            except Exception as e:
                logger.warning(f"Failed to store in ArangoDB: {e}")
        
        # Store in Redis (for fast access)
        key = f"{self.pending_prefix}{user_id}:{shadow_id}"
        await self.redis.setex(
            key,
            self.auto_promote_after_days * 24 * 3600,
            json.dumps(shadow_memory.to_dict())
        )
        
        # Add to user's pending list
        await self.redis.sadd(f"{self.pending_prefix}{user_id}:list", shadow_id)
        
        # Publish shadow tier event
        await self._publish_shadow_event(shadow_memory, "created")
        
        logger.info(
            f"Added memory to shadow tier: {shadow_id}",
            extra={
                "shadow_id": shadow_id,
                "user_id": user_id,
                "ego_score": ego_score,
                "confidence": confidence,
                "expires_at": expires_at.isoformat()
            }
        )
        
        return shadow_memory
    
    async def handle_user_confirmation(
        self,
        clarification_id: str,
        confirmed: bool
    ) -> None:
        """
        Handle user confirmation/rejection of shadow memory
        
        Args:
            clarification_id: Shadow memory ID
            confirmed: True if approved, False if rejected
        """
        # Fetch clarification data from Redis
        clarif_data = await self.redis.get(f"clarification:{clarification_id}")
        if not clarif_data:
            logger.warning(f"Clarification not found: {clarification_id}")
            return
        
        data = json.loads(clarif_data)
        node_id = data.get('node_id')
        
        if confirmed:
            # Promote to Tier 1 and publish event
            await self.event_bus.publish({
                'topic': 'tier.promoted',
                'payload': {
                    'node_id': node_id,
                    'from_tier': 'shadow',
                    'to_tier': 1,
                    'actor': 'user',
                    'clarification_id': clarification_id
                }
            })
        else:
            # Demote to Tier 2 and publish event
            await self.event_bus.publish({
                'topic': 'tier.demoted',
                'payload': {
                    'node_id': node_id,
                    'from_tier': 'shadow',
                    'to_tier': 2,
                    'actor': 'user',
                    'clarification_id': clarification_id
                }
            })
        
        # Cleanup Redis
        await self.redis.delete(f"clarification:{clarification_id}")
    
    async def approve_shadow_memory(
        self,
        user_id: str,
        shadow_id: str
    ) -> Optional[ShadowMemory]:
        """
        User approves shadow memory → promote to Tier 1
        """
        
        # Get shadow memory
        shadow_memory = await self.get_shadow_memory(user_id, shadow_id)
        
        if not shadow_memory:
            logger.warning(f"Shadow memory not found: {shadow_id}")
            return None
        
        if shadow_memory.status != 'pending':
            logger.warning(f"Shadow memory not pending: {shadow_id} (status: {shadow_memory.status})")
            return None
        
        # Update status
        shadow_memory.status = 'approved'
        
        # Move from pending to approved
        pending_key = f"{self.pending_prefix}{user_id}:{shadow_id}"
        approved_key = f"{self.approved_prefix}{user_id}:{shadow_id}"
        
        await self.redis.delete(pending_key)
        await self.redis.setex(
            approved_key,
            30 * 24 * 3600,  # Keep for 30 days
            json.dumps(shadow_memory.to_dict())
        )
        
        await self.redis.srem(f"{self.pending_prefix}{user_id}:list", shadow_id)
        await self.redis.sadd(f"{self.approved_prefix}{user_id}:list", shadow_id)
        
        # Publish tier promotion event
        await self._publish_tier_promotion(shadow_memory)
        
        logger.info(
            f"Shadow memory approved: {shadow_id}",
            extra={"shadow_id": shadow_id, "user_id": user_id}
        )
        
        return shadow_memory
    
    async def reject_shadow_memory(
        self,
        user_id: str,
        shadow_id: str,
        reason: Optional[str] = None
    ) -> Optional[ShadowMemory]:
        """
        User rejects shadow memory → discard
        """
        
        # Get shadow memory
        shadow_memory = await self.get_shadow_memory(user_id, shadow_id)
        
        if not shadow_memory:
            logger.warning(f"Shadow memory not found: {shadow_id}")
            return None
        
        if shadow_memory.status != 'pending':
            logger.warning(f"Shadow memory not pending: {shadow_id}")
            return None
        
        # Update status
        shadow_memory.status = 'rejected'
        
        # Move from pending to rejected
        pending_key = f"{self.pending_prefix}{user_id}:{shadow_id}"
        rejected_key = f"{self.rejected_prefix}{user_id}:{shadow_id}"
        
        await self.redis.delete(pending_key)
        await self.redis.setex(
            rejected_key,
            30 * 24 * 3600,  # Keep for 30 days for audit
            json.dumps({**shadow_memory.to_dict(), "rejection_reason": reason})
        )
        
        await self.redis.srem(f"{self.pending_prefix}{user_id}:list", shadow_id)
        await self.redis.sadd(f"{self.rejected_prefix}{user_id}:list", shadow_id)
        
        # Publish rejection event
        await self._publish_shadow_event(shadow_memory, "rejected", reason)
        
        logger.info(
            f"Shadow memory rejected: {shadow_id}",
            extra={"shadow_id": shadow_id, "user_id": user_id, "reason": reason}
        )
        
        return shadow_memory
    
    async def auto_promote_expired(self):
        """
        Background job: Auto-promote expired shadow memories
        
        Should run periodically (e.g., hourly)
        """
        
        logger.info("Running auto-promotion for expired shadow memories")
        
        promoted_count = 0
        
        # Get all user lists
        pattern = f"{self.pending_prefix}*:list"
        async for key in self.redis.scan_iter(match=pattern):
            user_id = key.split(':')[1]
            
            # Get all pending shadow IDs for this user
            shadow_ids = await self.redis.smembers(key)
            
            for shadow_id in shadow_ids:
                shadow_memory = await self.get_shadow_memory(user_id, shadow_id)
                
                if not shadow_memory:
                    continue
                
                # Check if expired
                expires_at_str = shadow_memory.expires_at.replace('Z', '+00:00')
                expires_at = datetime.fromisoformat(expires_at_str)
                
                # Make comparison timezone-aware
                now = datetime.utcnow()
                if expires_at.tzinfo is not None:
                    from datetime import timezone
                    now = datetime.now(timezone.utc)
                
                if now >= expires_at:
                    # Auto-promote
                    shadow_memory.status = 'auto_promoted'
                    
                    # Move to approved
                    pending_key = f"{self.pending_prefix}{user_id}:{shadow_id}"
                    approved_key = f"{self.approved_prefix}{user_id}:{shadow_id}"
                    
                    await self.redis.delete(pending_key)
                    await self.redis.setex(
                        approved_key,
                        30 * 24 * 3600,
                        json.dumps(shadow_memory.to_dict())
                    )
                    
                    await self.redis.srem(f"{self.pending_prefix}{user_id}:list", shadow_id)
                    await self.redis.sadd(f"{self.approved_prefix}{user_id}:list", shadow_id)
                    
                    # Publish tier promotion event
                    await self._publish_tier_promotion(shadow_memory)
                    
                    promoted_count += 1
                    
                    logger.info(
                        f"Auto-promoted expired shadow memory: {shadow_id}",
                        extra={"shadow_id": shadow_id, "user_id": user_id}
                    )
        
        logger.info(f"Auto-promotion complete: {promoted_count} memories promoted")
        
        return promoted_count
    
    async def get_shadow_memory(
        self,
        user_id: str,
        shadow_id: str
    ) -> Optional[ShadowMemory]:
        """Get shadow memory by ID"""
        
        # Try pending first
        key = f"{self.pending_prefix}{user_id}:{shadow_id}"
        data = await self.redis.get(key)
        
        if data:
            return ShadowMemory.from_dict(json.loads(data))
        
        # Try approved
        key = f"{self.approved_prefix}{user_id}:{shadow_id}"
        data = await self.redis.get(key)
        
        if data:
            return ShadowMemory.from_dict(json.loads(data))
        
        # Try rejected
        key = f"{self.rejected_prefix}{user_id}:{shadow_id}"
        data = await self.redis.get(key)
        
        if data:
            return ShadowMemory.from_dict(json.loads(data))
        
        return None
    
    async def get_pending_for_user(self, user_id: str) -> List[ShadowMemory]:
        """Get all pending shadow memories for user"""
        
        # Try ArangoDB first if available (for persistence)
        if hasattr(self, 'arango') and self.arango:
            try:
                query = """
                FOR mem IN shadow_memories
                    FILTER mem.user_id == @user_id
                    FILTER mem.status == 'pending'
                    SORT mem.proposed_at DESC
                    RETURN mem
                """
                result = await self.arango.aql.execute(query, bind_vars={'user_id': user_id})
                # Convert ArangoDB docs to ShadowMemory objects
                memories = []
                for m in result:
                    if isinstance(m, dict):
                        # Remove ArangoDB-specific fields
                        m.pop('_key', None)
                        m.pop('_id', None)
                        m.pop('_rev', None)
                        # Map fields if needed
                        if 'proposed_at' in m and 'created_at' not in m:
                            m['created_at'] = m.pop('proposed_at')
                        memories.append(m)
                    else:
                        memories.append(m)
                return memories
            except Exception as e:
                logger.warning(f"Failed to fetch from ArangoDB: {e}")
        
        # Fallback to Redis
        shadow_ids = await self.redis.smembers(f"{self.pending_prefix}{user_id}:list")
        
        memories = []
        for shadow_id in shadow_ids:
            memory = await self.get_shadow_memory(user_id, shadow_id)
            if memory and memory.status == 'pending':
                memories.append(memory)
        
        # Sort by created_at (newest first)
        memories.sort(key=lambda m: m.created_at, reverse=True)
        
        return memories
    
    async def _publish_shadow_event(
        self,
        shadow_memory: ShadowMemory,
        action: str,
        reason: Optional[str] = None
    ):
        """Publish shadow tier event"""
        
        try:
            event = Event(
                topic="shadow_tier.action",
                event_type=f"shadow.{action}",
                payload={
                    "shadow_id": shadow_memory.shadow_id,
                    "node_id": shadow_memory.node_id,
                    "user_id": shadow_memory.user_id,
                    "action": action,
                    "status": shadow_memory.status,
                    "ego_score": shadow_memory.ego_score,
                    "confidence": shadow_memory.confidence,
                    "reason": reason
                }
            )
            
            await self.event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Failed to publish shadow event: {e}", exc_info=True)
    
    async def _promote_to_tier1(self, node_id: str, actor: str = 'user') -> None:
        """Promote memory to Tier 1 (core memory)"""
        logger.info(f"Promoting {node_id} to Tier 1 (actor: {actor})")
        
        # Publish tier promotion event
        await self.event_bus.publish({
            'event_type': 'tier.promoted',
            'payload': {
                'node_id': node_id,
                'from_tier': 0.5,  # Shadow tier
                'to_tier': 1,
                'actor': actor
            }
        })
    
    async def _demote_to_tier2(self, node_id: str, actor: str = 'user') -> None:
        """Demote memory to Tier 2 (long-term memory)"""
        logger.info(f"Demoting {node_id} to Tier 2 (actor: {actor})")
        
        # Publish tier demotion event
        await self.event_bus.publish({
            'event_type': 'tier.demoted',
            'payload': {
                'node_id': node_id,
                'from_tier': 0.5,  # Shadow tier
                'to_tier': 2,
                'actor': actor
            }
        })
    
    async def _generate_clarification(self, memory: Dict[str, Any]) -> str:
        """Generate clarification question for a memory"""
        summary = memory.get('summary', '')
        question = f"I noticed you mentioned: '{summary}'. Should I remember this as a core preference?"
        
        # Try to generate with LLM if available
        if hasattr(self, 'llm') and self.llm:
            try:
                response = await self.llm.chat.completions.create(
                    model='gpt-4-turbo-preview',
                    messages=[
                        {"role": "system", "content": "Generate a short, natural clarification question."},
                        {"role": "user", "content": f"Memory: {summary}"}
                    ],
                    max_tokens=50
                )
                question = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Failed to generate LLM clarification: {e}")
        
        return question
    
    async def _process_auto_promote_batch(self) -> int:
        """Process batch of auto-promotions for expired shadow memories"""
        promoted_count = await self.auto_promote_expired()
        return promoted_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get shadow tier statistics"""
        
        # Try ArangoDB first if available
        if hasattr(self, 'arango') and self.arango:
            try:
                query = """
                FOR mem IN shadow_memories
                    COLLECT status = mem.status WITH COUNT INTO count
                    RETURN {status: status, count: count}
                """
                result = await self.arango.aql.execute(query)
                
                # Convert to dict
                stats = {}
                for item in result:
                    stats[item['status']] = item['count']
                
                return stats
            except Exception as e:
                logger.warning(f"Failed to fetch stats from ArangoDB: {e}")
        
        # Fallback to Redis counts
        return {
            'pending': 0,
            'approved': 0,
            'rejected': 0,
            'auto_promoted': 0
        }
    
    async def _publish_tier_promotion(self, shadow_memory: ShadowMemory):
        """Publish tier promotion event to create Tier 1 memory"""
        
        try:
            event = Event(
                topic="tier.promoted",
                event_type="tier.shadow_to_tier1",
                payload={
                    "node_id": shadow_memory.node_id,
                    "user_id": shadow_memory.user_id,
                    "content": shadow_memory.content,
                    "summary": shadow_memory.summary,
                    "tier": "tier1",
                    "ego_score": shadow_memory.ego_score,
                    "confidence": shadow_memory.confidence,
                    "sources": shadow_memory.sources,
                    "model_version": shadow_memory.model_version,
                    "observed_at": shadow_memory.created_at,
                    "promoted_from_shadow": True,
                    "shadow_status": shadow_memory.status,
                    "version": 1
                }
            )
            
            await self.event_bus.publish(event)
            
            logger.info(
                f"Published tier promotion event for {shadow_memory.shadow_id}",
                extra={"shadow_id": shadow_memory.shadow_id, "status": shadow_memory.status}
            )
            
        except Exception as e:
            logger.error(f"Failed to publish tier promotion event: {e}", exc_info=True)
    

