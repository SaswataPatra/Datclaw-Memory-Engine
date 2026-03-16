"""
Background Consolidation Service

Runs in background after entity linking. Extracts relations via LLM,
creates candidate edges, and triggers activation scoring → promotion.

This is the "brain consolidation" layer:
  entity_memories (fast, for RAG) runs synchronously
  this service runs async: relations → candidate_edges → activation → thought_graph
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

RELATION_EXTRACTION_PROMPT = """Extract relationships between entities in this memory.
Return a JSON array of objects: [{{"subject": "entity1", "predicate": "relation", "object": "entity2", "confidence": 0.0-1.0}}]

Use lowercase entity names. Use predicates like: sister_of, brother_of, parent_of, spouse_of, works_at, colleague_of, likes, dislikes, prefers, interested_in, located_at, owns, member_of, knows, friend_of.
If no clear relationships, return [].

Memory: "{text}"

JSON array:"""


class BackgroundConsolidationService:
    """
    Extracts relations via LLM and feeds them into the existing
    candidate_edge → activation_scorer → thought_graph pipeline.
    """

    def __init__(
        self,
        api_key: str,
        candidate_edge_store,
        activation_scorer,
        thought_edge_store,
        entity_resolver=None,
        model: str = "gpt-4o-mini"
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.candidate_edge_store = candidate_edge_store
        self.activation_scorer = activation_scorer
        self.thought_edge_store = thought_edge_store
        self.entity_resolver = entity_resolver

    async def process_memory_background(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        ego_score: float,
        tier: int,
        entity_names: List[str],
        session_id: str = None
    ):
        """
        Background consolidation for a memory.
        Called via asyncio.create_task() after entity linking.

        1. Extract relations (LLM)
        2. Resolve entity IDs
        3. Create/update candidate edges with supporting mentions
        4. Run activation scoring
        5. Promote if threshold met
        """
        try:
            relations = await self._extract_relations(content)
            if not relations:
                return

            logger.debug(f"Background consolidation: {len(relations)} relations from {memory_id}")

            candidate_ids = []
            for rel in relations:
                subject = rel.get("subject", "").lower().strip()
                predicate = rel.get("predicate", "").lower().strip()
                obj = rel.get("object", "").lower().strip()
                confidence = float(rel.get("confidence", 0.5))

                if not subject or not predicate or not obj:
                    continue

                # Resolve entities to IDs if resolver available
                subject_id = None
                object_id = None
                if self.entity_resolver:
                    try:
                        subject_resolved = await self.entity_resolver.resolve(
                            text=subject, user_id=user_id, create_if_missing=True
                        )
                        object_resolved = await self.entity_resolver.resolve(
                            text=obj, user_id=user_id, create_if_missing=True
                        )
                        subject_id = subject_resolved.entity_id if subject_resolved else None
                        object_id = object_resolved.entity_id if object_resolved else None
                    except Exception as e:
                        logger.debug(f"Entity resolution skipped: {e}")

                # Build candidate edge using existing schema
                from core.graph.schemas import CandidateEdge, SupportingMention
                candidate = CandidateEdge(
                    user_id=user_id,
                    subject_entity_id=subject_id or f"unresolved_{subject}",
                    object_entity_id=object_id or f"unresolved_{obj}",
                    predicate=predicate,
                    subject_span={"text": subject},
                    object_span={"text": obj},
                    supporting_mentions=[
                        SupportingMention(
                            mem_id=memory_id,
                            srl_conf=confidence,
                            ego=ego_score,
                        )
                    ],
                    first_seen=datetime.utcnow(),
                    last_fired_at=datetime.utcnow(),
                )

                try:
                    stored = await self.candidate_edge_store.create_or_update(candidate)
                    candidate_ids.append(stored.candidate_id)
                except Exception as e:
                    logger.debug(f"Candidate edge creation failed: {e}")

            # Activation scoring (only for tier 1-2 memories)
            if candidate_ids and tier <= 2:
                await self._score_and_maybe_promote(user_id, candidate_ids)

        except Exception as e:
            logger.warning(f"Background consolidation failed for {memory_id}: {e}", exc_info=True)

    async def _extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relations from text using LLM."""
        if not text or not text.strip():
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract entity relationships from text. Return only a JSON array."},
                    {"role": "user", "content": RELATION_EXTRACTION_PROMPT.format(text=text[:1000])}
                ],
                temperature=0.1,
                max_tokens=400
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            items = json.loads(content)
            if not isinstance(items, list):
                return []
            return [r for r in items if isinstance(r, dict) and r.get("subject") and r.get("predicate") and r.get("object")]
        except json.JSONDecodeError:
            return []
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")
            return []

    async def _score_and_maybe_promote(self, user_id: str, candidate_ids: List[str]):
        """Score candidates and promote if above threshold."""
        for candidate_id in candidate_ids:
            try:
                candidate = await self.candidate_edge_store.get(candidate_id)
                if not candidate:
                    continue

                result = await self.activation_scorer.score(candidate)

                await self.candidate_edge_store.update_activation(
                    candidate_id=candidate_id,
                    activation=result.activation_score,
                    status=result.decision
                )

                if result.decision == "promote":
                    await self._promote_candidate(candidate, result)
                    logger.info(f"🧠 Promoted edge: {candidate.predicate} (score={result.activation_score:.3f})")
                else:
                    logger.debug(f"Edge {candidate_id}: {result.decision} (score={result.activation_score:.3f})")

            except Exception as e:
                logger.warning(f"Activation scoring failed for {candidate_id}: {e}")

    async def _promote_candidate(self, candidate, activation_result):
        """Promote a candidate edge to thought_graph."""
        try:
            from core.graph.schemas import ThoughtEdge
            thought_edge = ThoughtEdge(
                relation=candidate.predicate,
                relation_category=self._infer_category(candidate.predicate),
                strength=activation_result.activation_score,
                effective_from=candidate.first_seen,
                is_bidirectional=self._is_symmetric(candidate.predicate),
                supporting_mentions=[m.mem_id for m in candidate.supporting_mentions]
            )

            await self.thought_edge_store.create(
                thought_edge,
                from_entity_id=candidate.subject_entity_id,
                to_entity_id=candidate.object_entity_id
            )

            await self.candidate_edge_store.update_status(candidate.candidate_id, "promoted")

        except Exception as e:
            logger.error(f"Failed to promote candidate: {e}")

    @staticmethod
    def _infer_category(predicate: str) -> str:
        family = {"sister_of", "brother_of", "parent_of", "child_of", "spouse_of", "family_of", "daughter_of", "son_of", "sibling_of"}
        professional = {"works_at", "works_with", "colleague_of", "manages", "employed_by", "reports_to"}
        temporal = {"contradicts", "supersedes", "evolves_to", "replaces"}
        if predicate in family:
            return "family"
        if predicate in professional:
            return "professional"
        if predicate in temporal:
            return "temporal"
        return "general"

    @staticmethod
    def _is_symmetric(predicate: str) -> bool:
        symmetric = {"friend_of", "knows", "colleague_of", "works_with", "sibling_of", "married_to", "partner_of"}
        return predicate in symmetric
