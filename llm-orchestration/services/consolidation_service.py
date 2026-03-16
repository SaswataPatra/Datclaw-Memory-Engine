"""
Memory Consolidation Service
Single LLM call that extracts entities and relations with confidence scoring.
Replaces separate entity_extraction + relation_extraction calls.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """You are a Memory Consolidation Agent. Extract all meaningful knowledge from this memory.

INPUTS:
- Memory text
- Document date: when the user said/wrote this (ISO format)
- Ego score (0-1): importance/stability indicator
- Tier (1-3): 1=core knowledge, 2=long-term, 3=transient

YOUR TASK:

1. Extract entities: people, places, projects, topics, organizations, activities, events
2. Extract relations between entities as (subject, predicate, object) triples
3. Extract temporal information: resolve dates and time references
4. Assign confidence (0-1) to each relation

RULES:

- Extract entities from ALL meaningful content, including casual conversation
- Casual mentions of people, places, activities, and events ARE worth extracting
- Use "user" for the person speaking
- Normalize entities: prefer existing names, avoid duplicates
- Confidence reflects certainty and importance:
  - 0.9-1.0: Explicit, stated facts (names, relationships, core identities)
  - 0.7-0.9: Clear activities or interests mentioned in context
  - 0.5-0.7: Casual mentions, one-off references
  - <0.5: Vague or ambiguous references
- Higher ego score + lower tier → higher confidence

PREDICATE GUIDELINES:

IDENTITY & CORE KNOWLEDGE (high confidence — these define who the user is):
- user --identifies_as--> [gender identity]  (e.g. transgender woman, non-binary)
- user --gender--> [gender]
- user --pronouns--> [pronouns]
- user --sexual_orientation--> [orientation]
- user --nationality--> [country/ethnicity]
- user --religion--> [belief]
- user --age--> [age]
- user --has_name--> [name/nickname]
- user --life_goal--> [goal]  (e.g. adopt children, become a counselor)
- user --values--> [value]  (e.g. inclusivity, acceptance)
- user --struggles_with--> [challenge]

For family relations, use directional predicates:
- user --has_father--> [name]
- user --has_mother--> [name]
- user --has_sister--> [name]
- user --has_brother--> [name]
- user --has_daughter--> [name]
- user --has_son--> [name]
- user --has_partner--> [name]
- user --has_friend--> [name]
- [name] --father_of--> user

For education/career:
- user --studies--> [field]
- user --pursuing_degree_in--> [field]
- user --works_on--> [project]
- user --works_at--> [company]
- user --career_goal--> [role/field]
- user --researching--> [topic]

For interests/skills:
- user --interested_in--> [topic]
- user --skilled_in--> [skill]
- user --hobby--> [activity]

For locations:
- user --lives_in--> [place]
- user --visited--> [place]

For activities/events:
- user --attended--> [event]
- user --participated_in--> [activity]

For general relations:
- [entity1] --is_a--> [type]
- [entity1] --part_of--> [entity2]
- [entity1] --related_to--> [entity2]

IMPORTANT: Identity-related facts (gender, orientation, ethnicity, core beliefs, life goals) are ALWAYS high confidence (0.9+) when explicitly stated. These are core to the user's self-definition.

TEMPORAL EXTRACTION:

Extract and resolve ALL time references:
- Explicit dates: "May 7, 2023", "in 2022", "March 15th" → extract as-is
- Relative time: "yesterday", "last week", "two months ago" → resolve against document_date
- Future references: "next Monday", "in 3 days" → resolve against document_date

Examples:
- Document date: 2023-05-08, Text: "I went to the vet yesterday" → event_dates: ["2023-05-07"]
- Document date: 2023-03-15, Text: "I painted a sunrise in 2022" → event_dates: ["2022"]
- Document date: 2024-01-10, Text: "Last week I joined a gym" → event_dates: ["2024-01-03"] (approximate start of last week)

Return event_dates as ISO strings (YYYY-MM-DD or YYYY). If no temporal info, return empty array.

OUTPUT FORMAT (strict JSON):

{{
  "entities": [
    {{"name": "entity_name", "type": "person|place|org|project|topic|event|activity|other"}}
  ],
  "relations": [
    {{"subject": "entity1", "predicate": "relation", "object": "entity2", "confidence": 0.0}}
  ],
  "temporal": {{
    "event_dates": ["2023-05-07"],
    "time_expressions": ["yesterday"]
  }}
}}

Return empty arrays/objects ONLY if the text contains no identifiable entities, relations, or temporal information.

---

MEMORY: {text}
DOCUMENT DATE: {document_date}
EGO SCORE: {ego_score}
TIER: {tier}

JSON output:"""


BATCHED_CONSOLIDATION_PROMPT = """You are a Memory Consolidation Agent processing a batch of memories. For each memory, extract entities, relations, temporal information, and summarize the assistant's factual contributions.

BATCH INPUT:
Each memory has:
- Content: What the user said
- Context: What the assistant replied (may contain facts about the assistant or additional information)
- Document Date: When the conversation happened

YOUR TASK (per memory):
1. Extract entities from BOTH content and context
2. Extract relations from BOTH content and context
3. Extract and resolve temporal references from BOTH content and context
4. Generate a context_summary: Extract ONLY factual statements from the assistant's response. Skip pleasantries, questions, and filler. If the assistant shares facts (e.g., "I painted X in 2022", "My daughter is 5"), extract those. If it's just validation/questions, return empty string.

RULES:
- Use same entity/relation/temporal extraction rules as single-memory consolidation
- For context_summary: 1-2 sentences max, facts only, or empty string if no facts
- Map each output back to the correct memory index

OUTPUT FORMAT:
{{
  "memories": [
    {{
      "index": 0,
      "entities": [{{"name": "entity", "type": "person|place|..."}}],
      "relations": [{{"subject": "e1", "predicate": "rel", "object": "e2", "confidence": 0.9}}],
      "temporal": {{"event_dates": ["2023-05-07"], "time_expressions": ["yesterday"]}},
      "context_summary": "Factual summary of assistant's response, or empty string."
    }}
  ]
}}

---

{batch_input}

JSON output:"""


class ConsolidationService:
    """
    Consolidates memory into structured knowledge (entities + relations).
    Supports both single-memory and batched processing.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def consolidate(
        self,
        text: str,
        ego_score: float = 0.5,
        tier: int = 2,
        document_date: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract entities, relations, and temporal information from memory text.

        Args:
            text: Memory content
            ego_score: Importance score (0-1)
            tier: Memory tier (1-3)
            document_date: ISO date when the memory was created/observed (for resolving relative time)

        Returns:
            {
                "entities": [{"name": str, "type": str}],
                "relations": [{"subject": str, "predicate": str, "object": str, "confidence": float}],
                "temporal": {"event_dates": [str], "time_expressions": [str]}
            }
        """
        if not text or not text.strip():
            return {
                "entities": [],
                "relations": [],
                "temporal": {"event_dates": [], "time_expressions": []},
                "context_summary": ""
            }
        
        # If context is provided, use batch-style prompt for consistency
        if context and context.strip():
            batch_result = await self.consolidate_batch(
                memories=[{
                    "content": text,
                    "context": context,
                    "ego_score": ego_score,
                    "tier": tier,
                    "document_date": document_date or datetime.utcnow().isoformat()
                }],
                max_batch_size=1
            )
            return batch_result[0] if batch_result else {
                "entities": [],
                "relations": [],
                "temporal": {"event_dates": [], "time_expressions": []},
                "context_summary": ""
            }

        try:
            from datetime import datetime
            doc_date_str = document_date or datetime.utcnow().isoformat()
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Extract all meaningful entities, relations, and temporal information as structured JSON."},
                    {"role": "user", "content": CONSOLIDATION_PROMPT.format(
                        text=text[:1500],
                        document_date=doc_date_str,
                        ego_score=ego_score,
                        tier=tier
                    )}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            result = json.loads(content)
            
            if not isinstance(result, dict):
                return {"entities": [], "relations": [], "temporal": {"event_dates": [], "time_expressions": []}}
            
            # Validate and normalize entities
            entities = []
            for ent in result.get("entities", []):
                if isinstance(ent, dict) and "name" in ent:
                    name = str(ent["name"]).lower().strip()
                    ent_type = str(ent.get("type", "other")).lower()
                    if name and len(name) > 1:
                        entities.append({"name": name, "type": ent_type})
            
            # Validate and normalize relations
            relations = []
            for rel in result.get("relations", []):
                if isinstance(rel, dict):
                    subject = str(rel.get("subject", "")).lower().strip()
                    predicate = str(rel.get("predicate", "")).lower().strip()
                    obj = str(rel.get("object", "")).lower().strip()
                    confidence = float(rel.get("confidence", 0.5))
                    
                    if subject and predicate and obj and len(subject) > 1 and len(obj) > 1:
                        relations.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "confidence": min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
                        })
            
            # Validate and normalize temporal data
            temporal = result.get("temporal", {})
            if not isinstance(temporal, dict):
                temporal = {"event_dates": [], "time_expressions": []}
            
            event_dates = []
            for date in temporal.get("event_dates", []):
                if isinstance(date, str) and date.strip():
                    event_dates.append(date.strip())
            
            time_expressions = []
            for expr in temporal.get("time_expressions", []):
                if isinstance(expr, str) and expr.strip():
                    time_expressions.append(expr.strip())
            
            temporal_normalized = {
                "event_dates": event_dates[:10],
                "time_expressions": time_expressions[:10]
            }
            
            logger.debug(f"Consolidation: {len(entities)} entities, {len(relations)} relations, {len(event_dates)} event dates")
            return {
                "entities": entities[:20],
                "relations": relations[:20],
                "temporal": temporal_normalized
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Consolidation JSON parse failed: {e}")
            return {"entities": [], "relations": [], "temporal": {"event_dates": [], "time_expressions": []}}
        except Exception as e:
            logger.warning(f"Consolidation failed: {e}", exc_info=True)
            return {"entities": [], "relations": [], "temporal": {"event_dates": [], "time_expressions": []}}
    
    async def consolidate_batch(
        self,
        memories: List[Dict[str, Any]],
        max_batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Batch consolidation: process multiple memories in one LLM call.
        
        Args:
            memories: List of dicts with keys: content, context (optional), ego_score, tier, document_date
            max_batch_size: Max memories per batch (default 10)
        
        Returns:
            List of consolidation results (same order as input), each with:
            {
                "entities": [...],
                "relations": [...],
                "temporal": {...},
                "context_summary": str
            }
        """
        if not memories:
            return []
        
        import asyncio
        
        total_batches = (len(memories) + max_batch_size - 1) // max_batch_size
        max_parallel = 5
        logger.info(f"🔄 Starting batch consolidation: {len(memories)} memories → {total_batches} batches ({max_parallel} parallel)")
        
        # Build all batches
        all_batches = []
        for batch_start in range(0, len(memories), max_batch_size):
            all_batches.append(memories[batch_start:batch_start + max_batch_size])
        
        # Process in parallel waves of max_parallel
        results = []
        for wave_idx, wave_start in enumerate(range(0, len(all_batches), max_parallel)):
            wave = all_batches[wave_start:wave_start + max_parallel]
            wave_num = wave_idx + 1
            total_waves = (len(all_batches) + max_parallel - 1) // max_parallel
            
            logger.info(f"   🚀 Wave {wave_num}/{total_waves}: launching {len(wave)} batches in parallel...")
            
            tasks = [self._consolidate_single_batch(batch) for batch in wave]
            wave_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, wr in enumerate(wave_results):
                batch_idx = wave_start + i
                if isinstance(wr, Exception):
                    logger.warning(f"   ❌ Batch {batch_idx + 1} failed: {wr}")
                    results.extend([self._empty_result() for _ in wave[i]])
                else:
                    results.extend(wr)
            
            logger.info(f"   ✅ Wave {wave_num}/{total_waves} complete ({len(results)}/{len(memories)} memories done)")
        
        logger.info(f"✅ Batch consolidation complete: {len(results)} results")
        return results
    
    async def _consolidate_single_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of memories."""
        from datetime import datetime
        
        # Build batch input
        batch_lines = []
        for i, mem in enumerate(batch):
            content = mem.get('content', '').strip()
            context = mem.get('context', '').strip()
            doc_date = mem.get('document_date') or datetime.utcnow().isoformat()
            ego = mem.get('ego_score', 0.5)
            tier = mem.get('tier', 2)
            
            batch_lines.append(f"M{i}:")
            batch_lines.append(f"  Content: {content[:800]}")
            if context:
                batch_lines.append(f"  Context: {context[:800]}")
            batch_lines.append(f"  Document Date: {doc_date}")
            batch_lines.append(f"  Ego Score: {ego:.2f}, Tier: {tier}")
            batch_lines.append("")
        
        batch_input = "\n".join(batch_lines)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Process batches of memories and extract structured knowledge as JSON."},
                    {"role": "user", "content": BATCHED_CONSOLIDATION_PROMPT.format(batch_input=batch_input)}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            result = json.loads(content)
            
            if not isinstance(result, dict) or "memories" not in result:
                logger.warning(f"Batch consolidation returned invalid format")
                return [self._empty_result() for _ in batch]
            
            # Validate and normalize each memory's result
            normalized = []
            for mem_result in result.get("memories", []):
                normalized.append(self._normalize_batch_result(mem_result))
            
            # Ensure we have results for all input memories
            while len(normalized) < len(batch):
                normalized.append(self._empty_result())
            
            logger.debug(f"Batch consolidation: {len(batch)} memories processed")
            return normalized[:len(batch)]
            
        except json.JSONDecodeError as e:
            logger.warning(f"Batch consolidation JSON parse failed: {e}")
            return [self._empty_result() for _ in batch]
        except Exception as e:
            logger.warning(f"Batch consolidation failed: {e}", exc_info=True)
            return [self._empty_result() for _ in batch]
    
    def _normalize_batch_result(self, mem_result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single memory's result from batch output."""
        # Validate entities
        entities = []
        for ent in mem_result.get("entities", []):
            if isinstance(ent, dict) and "name" in ent:
                name = str(ent["name"]).lower().strip()
                ent_type = str(ent.get("type", "other")).lower()
                if name and len(name) > 1:
                    entities.append({"name": name, "type": ent_type})
        
        # Validate relations
        relations = []
        for rel in mem_result.get("relations", []):
            if isinstance(rel, dict):
                subject = str(rel.get("subject", "")).lower().strip()
                predicate = str(rel.get("predicate", "")).lower().strip()
                obj = str(rel.get("object", "")).lower().strip()
                confidence = float(rel.get("confidence", 0.5))
                
                if subject and predicate and obj and len(subject) > 1 and len(obj) > 1:
                    relations.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": min(max(confidence, 0.0), 1.0)
                    })
        
        # Validate temporal
        temporal = mem_result.get("temporal", {})
        if not isinstance(temporal, dict):
            temporal = {"event_dates": [], "time_expressions": []}
        
        event_dates = [str(d).strip() for d in temporal.get("event_dates", []) if d]
        time_expressions = [str(e).strip() for e in temporal.get("time_expressions", []) if e]
        
        # Context summary
        context_summary = str(mem_result.get("context_summary", "")).strip()
        
        return {
            "entities": entities[:20],
            "relations": relations[:20],
            "temporal": {
                "event_dates": event_dates[:10],
                "time_expressions": time_expressions[:10]
            },
            "context_summary": context_summary[:500]  # Cap at 500 chars
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "entities": [],
            "relations": [],
            "temporal": {"event_dates": [], "time_expressions": []},
            "context_summary": ""
        }
