"""
DAPPY - Contradiction Signal Detector (Lightweight)

Stripped of LLM reasoning. Produces signals only:
- Similar memories (vector search)
- Temporal gap
- Similarity score

These signals are fed to the KG Maintenance Agent which does the LLM reasoning.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

logger = logging.getLogger(__name__)


@dataclass
class ContradictionSignal:
    """Lightweight signal for a potential contradiction between two memories"""
    new_memory_id: str
    new_memory_content: str
    similar_memory_id: str
    similar_memory_content: str
    similarity_score: float
    temporal_gap_days: float
    is_temporal_change: bool


class ContradictionSignalDetector:
    """
    Produces contradiction signals without LLM reasoning.

    Kept:
    - Vector search for similar memories (Qdrant)
    - Temporal gap calculation
    - Similarity scoring

    Removed:
    - LLM calls (_is_contradiction, _generate_clarification_question)
    - Edge creation in old graph
    - Event publishing (KG Maintenance has its own events)
    - OpenAI client (no LLM needed)
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        config: Dict[str, Any],
    ):
        self.qdrant = qdrant_client
        self.config = config

        contradiction_config = config.get('contradiction_detection', {})
        self.enabled = contradiction_config.get('enabled', True)
        self.temporal_gap_threshold_days = contradiction_config.get('temporal_gap_threshold_days', 365)
        self.similarity_threshold = contradiction_config.get('similarity_threshold', 0.30)

        logger.info(
            f"ContradictionSignalDetector initialized "
            f"(threshold={self.similarity_threshold}, temporal_gap={self.temporal_gap_threshold_days}d)"
        )

    async def get_contradiction_signals(
        self,
        user_id: str,
        memory_content: str,
        memory_id: str,
        embedding: List[float],
        observed_at: Optional[str] = None,
    ) -> List[ContradictionSignal]:
        """
        Find similar memories and produce contradiction signals.

        Args:
            user_id: User ID
            memory_content: Content of the new memory
            memory_id: ID of the new memory
            embedding: Pre-computed embedding (from EmbeddingService)
            observed_at: When the new memory was observed

        Returns:
            List of ContradictionSignal (potential contradictions with metadata)
        """
        if not self.enabled:
            return []

        similar_memories = self._find_similar_memories(
            user_id=user_id,
            embedding=embedding,
            exclude_id=memory_id,
            limit=5
        )

        if not similar_memories:
            return []

        signals = []
        new_observed = self._parse_datetime(observed_at)

        for mem in similar_memories:
            similar_observed = self._parse_datetime(mem.get('observed_at'))
            temporal_gap_days = abs((new_observed - similar_observed).days) if new_observed and similar_observed else 0
            is_temporal_change = temporal_gap_days > self.temporal_gap_threshold_days

            signals.append(ContradictionSignal(
                new_memory_id=memory_id,
                new_memory_content=memory_content,
                similar_memory_id=mem.get('id', ''),
                similar_memory_content=mem.get('content', ''),
                similarity_score=mem.get('score', 0.0),
                temporal_gap_days=temporal_gap_days,
                is_temporal_change=is_temporal_change,
            ))

        logger.info(f"   Contradiction signals: {len(signals)} similar memories found for KG Maintenance")
        return signals

    def _find_similar_memories(
        self,
        user_id: str,
        embedding: List[float],
        exclude_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar memories via Qdrant vector search (synchronous, no LLM)."""
        try:
            filter_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            collection_name = self.config.get('qdrant', {}).get('collection_name', 'memories')

            search_results = self.qdrant.search(
                collection_name=collection_name,
                query_vector=embedding,
                query_filter=Filter(must=filter_conditions),
                limit=limit,
                score_threshold=self.similarity_threshold
            )

            results = []
            for result in search_results:
                if exclude_id and str(result.id) == str(exclude_id):
                    continue
                results.append({
                    'id': result.id,
                    'content': result.payload.get('content', ''),
                    'score': result.score,
                    'observed_at': result.payload.get('observed_at') or result.payload.get('created_at'),
                    'ego_score': result.payload.get('ego_score', 0),
                    'tier': result.payload.get('tier', 3),
                })

            return results

        except Exception as e:
            logger.error(f"Vector search for contradiction signals failed: {e}")
            return []

    @staticmethod
    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string safely."""
        if not dt_str:
            return datetime.now(timezone.utc)
        try:
            if isinstance(dt_str, datetime):
                return dt_str
            if dt_str.endswith('Z'):
                dt_str = dt_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return datetime.now(timezone.utc)

    def format_signals_for_prompt(self, signals: List[ContradictionSignal]) -> str:
        """
        Format contradiction signals as text for the KG Maintenance Agent's LLM prompt.

        Returns a string block that can be injected directly into the prompt.
        """
        if not signals:
            return "No similar memories found (no contradiction signals)."

        lines = [f"Found {len(signals)} similar memories that may indicate contradictions:\n"]
        for i, sig in enumerate(signals, 1):
            lines.append(
                f"  {i}. Similar memory: \"{sig.similar_memory_content[:150]}...\"\n"
                f"     Similarity: {sig.similarity_score:.2f}, "
                f"Temporal gap: {sig.temporal_gap_days:.0f} days, "
                f"Temporal change: {sig.is_temporal_change}"
            )
        return "\n".join(lines)
