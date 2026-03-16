"""
Knowledge Graph Maintenance Agent
Runs in the background to detect contradictions, update relations, and keep the KG lean.

This agent:
1. Detects contradicting relations (e.g., two different father names)
2. Resolves conflicts by deprecating old relations or merging them
3. Increments supporting mentions for reinforced relations
4. Promotes high-confidence relations
5. Removes low-confidence, superseded relations
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


CONTRADICTION_DETECTION_PROMPT = """You are a Knowledge Graph Maintenance Agent. Your PRIMARY job is to detect contradictions and reinforcements in the user's knowledge graph.

═══════════════════════════════════════════════════════════════════════════════
INPUTS PROVIDED:
═══════════════════════════════════════════════════════════════════════════════
1. NEW MEMORY: The latest statement/question/correction from the user
2. NEW RELATIONS: Relations extracted from this memory (may be empty - still analyze!)
3. EXISTING KG: All current relations with temporal metadata (created_at, last_mentioned, supporting_mentions)
4. CONTRADICTION SIGNALS: Similar past memories that might conflict (from vector search)
5. REINFORCEMENT CANDIDATES: Existing relations that match the new memory content

═══════════════════════════════════════════════════════════════════════════════
YOUR CRITICAL TASKS (IN ORDER OF IMPORTANCE):
═══════════════════════════════════════════════════════════════════════════════

🚨 TASK 1: DETECT CONTRADICTIONS (HIGHEST PRIORITY)
Look for information that CONFLICTS with existing relations:

Types of contradictions:
a) DIRECT: Same (subject, predicate), different object
   Example: KG has "user --has_father--> john" but memory says "my father is mike"
   → This is a CLEAR contradiction - mark it!

b) SEMANTIC: Different wording, same meaning, conflicting values
   Example: KG has "user --works_at--> google" but memory says "I'm employed by microsoft"
   → "works_at" and "employed_by" are semantically equivalent - contradiction!

c) TEMPORAL: Old information superseded by new facts
   Example: KG has "user --lives_in--> boston" (created 6 months ago) but memory says "I moved to NYC"
   → Temporal supersession - mark as contradiction with type "temporal"

d) NAME CORRECTIONS: User explicitly corrects a name/value
   Example: Memory says "No it is Arun Kumar Patra" or "Actually it's X"
   → ALWAYS mark these as contradictions - user is correcting something!

🔄 TASK 2: DETECT REINFORCEMENTS (SECOND PRIORITY)
Look for information that CONFIRMS existing relations:

When to mark reinforcement:
- New memory mentions the SAME fact as an existing relation
- Check the REINFORCEMENT CANDIDATES section - these are pre-filtered matches
- If a relation has low supporting_mentions (<3) and is confirmed again, DEFINITELY reinforce it
- If a relation is old (last_mentioned > 30 days ago) and mentioned again, reinforce it

Example: KG has "user --works_at--> nexqloud" and memory says "I work at Nexqloud"
→ This is a reinforcement - mark it!

➕ TASK 3: ADD IMPLIED RELATIONS (LOWEST PRIORITY)
Only if the memory implies NEW information not in the KG:
- Be conservative - only add if clearly implied
- Don't add relations that are already extracted in "NEW RELATIONS"

═══════════════════════════════════════════════════════════════════════════════
CRITICAL DECISION RULES:
═══════════════════════════════════════════════════════════════════════════════

❌ QUESTIONS ALONE DON'T CREATE CONTRADICTIONS
- "What is my father's name?" → NO contradiction (just asking)
- "Who do I work for?" → NO contradiction (just asking)
- Questions only matter if they're followed by corrections

✅ CORRECTIONS ALWAYS CREATE CONTRADICTIONS
- "No it is X" → CONTRADICTION (user correcting)
- "Actually Y" → CONTRADICTION (user correcting)
- "I meant Z" → CONTRADICTION (user correcting)

⚠️ USE TEMPORAL CONTEXT
- Check created_at: Is this relation old?
- Check last_mentioned: When was it last confirmed?
- Check supporting_mentions: How many times has this been mentioned?
- Old relations (>90 days) with low mentions (<2) are more likely to be outdated

🎯 USE CONTRADICTION SIGNALS
- If similar memories are provided, check if they conflict with current memory
- Temporal gap > 30 days suggests information might have changed

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON):
═══════════════════════════════════════════════════════════════════════════════

{{
  "contradictions": [
    {{
      "old_relation": {{"subject": "user", "predicate": "has_father", "object": "john"}},
      "new_relation": {{"subject": "user", "predicate": "has_father", "object": "mike"}},
      "type": "direct",
      "confidence": 0.95,
      "resolution": "replace",
      "reasoning": "User explicitly stated father's name is Mike, contradicting existing relation with John"
    }}
  ],
  "reinforcements": [
    {{
      "relation": {{"subject": "user", "predicate": "works_at", "object": "nexqloud"}},
      "reasoning": "User mentioned working at Nexqloud again, confirming existing relation"
    }}
  ],
  "new_relations_to_add": [
    {{
      "subject": "user",
      "predicate": "interested_in",
      "object": "ai",
      "confidence": 0.8,
      "reasoning": "User discussed AI projects extensively, implying interest"
    }}
  ]
}}

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES:
═══════════════════════════════════════════════════════════════════════════════

Example 1 - Direct Contradiction:
Memory: "My father's name is Arun Kumar Patra"
KG: "user --has_father--> albida patra" (created: 2024-01-15, mentions: 1)
→ OUTPUT: contradiction with resolution="replace", type="direct"

Example 2 - Correction:
Memory: "No it is Rohit Patra"
KG: "user --has_father--> arun patra" (created: 2024-01-15, mentions: 2)
→ OUTPUT: contradiction with resolution="replace", type="direct"

Example 3 - Reinforcement:
Memory: "I work at Nexqloud as a blockchain developer"
KG: "user --works_at--> nexqloud" (created: 2024-01-10, mentions: 1, last_mentioned: 2024-01-10)
→ OUTPUT: reinforcement (confirming work relationship)

Example 4 - Question Only (NO ACTION):
Memory: "What is my father's name?"
KG: "user --has_father--> arun patra"
→ OUTPUT: {{}} (no contradictions, no reinforcements - just a question)

Example 5 - Multiple Relations:
Memory: "I'm working on DAPPY and a liquidation bot"
KG: "user --works_on--> dappy" (mentions: 3), "user --works_on--> memory engine" (mentions: 1)
→ OUTPUT: reinforcement for "dappy", new_relation for "liquidation bot"

═══════════════════════════════════════════════════════════════════════════════
INPUT DATA:
═══════════════════════════════════════════════════════════════════════════════

NEW MEMORY:
{new_memory}

NEW RELATIONS EXTRACTED:
{new_relations}

EXISTING KNOWLEDGE GRAPH (with temporal metadata):
{existing_relations}

CONTRADICTION SIGNALS (similar past memories):
{contradiction_signals}

REINFORCEMENT CANDIDATES (relations matching new memory):
{reinforcement_candidates}

═══════════════════════════════════════════════════════════════════════════════
YOUR JSON OUTPUT:
═══════════════════════════════════════════════════════════════════════════════
"""


class KGMaintenanceAgent:
    """
    Background agent that maintains the knowledge graph by detecting contradictions,
    updating relations, and keeping the graph lean and accurate.
    
    Uses ContradictionSignalDetector for lightweight signals (vector search + temporal gap)
    and LLM reasoning for actual contradiction resolution.
    """

    def __init__(self, knowledge_graph_store, api_key: str, model: str = "gpt-4o-mini",
                 contradiction_signal_detector=None):
        self.kg_store = knowledge_graph_store
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.signal_detector = contradiction_signal_detector

    async def process_memory(
        self,
        user_id: str,
        memory_id: str,
        memory_content: str,
        new_relations: List[Dict[str, Any]],
        ego_score: float,
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Process a new memory and update the KG accordingly.
        
        This method ALWAYS runs, even if new_relations is empty.
        The LLM analyzes the entire visible KG against the new memory.
        
        Args:
            user_id: User ID
            memory_id: Memory ID
            memory_content: The memory text (statement, question, or correction)
            new_relations: Relations extracted from the memory (may be empty)
            ego_score: Ego score of the memory
            embedding: Pre-computed embedding for contradiction signal detection
        
        Returns:
            Summary of actions taken
        """
        logger.info("="*80)
        logger.info(f"🔧 KG MAINTENANCE: Processing memory {memory_id}")
        logger.info(f"   User: {user_id}, Ego Score: {ego_score:.2f}")
        logger.info(f"   Memory: {memory_content[:100]}{'...' if len(memory_content) > 100 else ''}")
        logger.info(f"   New Relations Extracted: {len(new_relations)}")
        if new_relations:
            logger.info("   Newly Extracted Relations:")
            for rel in new_relations[:10]:
                logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel.get('confidence', 0.5):.2f})")
        
        # Get existing relations for this user
        # Strategy: If KG is small (<100), use full KG. If large (>=100), use localized KG.
        full_kg = self.kg_store.get_user_relations(user_id=user_id, limit=500)
        kg_size = len(full_kg)
        
        KG_LOCALIZATION_THRESHOLD = 100
        
        if kg_size < KG_LOCALIZATION_THRESHOLD:
            # Small KG: use all relations for comprehensive analysis
            existing_relations = full_kg
            logger.info(f"   📊 FULL KG: {kg_size} relations (below threshold, using all)")
        else:
            # Large KG: localize to entities mentioned in the new memory
            # Extract entity names from new relations
            entity_names = set()
            entity_names.add("user")  # Always include user
            for rel in new_relations:
                entity_names.add(rel.get('subject', '').lower())
                entity_names.add(rel.get('object', '').lower())
            
            # Get relations involving these entities
            existing_relations = self.kg_store.get_user_relations(
                user_id=user_id,
                entity_names=list(entity_names),
                limit=200
            )
            logger.info(f"   📊 LOCALIZED KG: {len(existing_relations)} relations (from {kg_size} total, entities: {sorted(entity_names)[:5]}...)")
        
        if existing_relations:
            logger.info("   Current KG Relations (top 20):")
            for rel in existing_relations[:20]:
                logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel.get('confidence', 0.5):.2f})")
            if len(existing_relations) > 20:
                logger.info(f"      ... and {len(existing_relations) - 20} more relations")
        
        if not existing_relations:
            logger.info("   No existing KG to maintain")
            logger.info("="*80)
            return {"contradictions_found": 0, "relations_updated": 0, "relations_removed": 0, "relations_added": 0}
        
        # Get contradiction signals (vector search + temporal gap, no LLM)
        contradiction_signals_text = "No contradiction signals available."
        if self.signal_detector and embedding:
            try:
                signals = await self.signal_detector.get_contradiction_signals(
                    user_id=user_id,
                    memory_content=memory_content,
                    memory_id=memory_id,
                    embedding=embedding,
                )
                contradiction_signals_text = self.signal_detector.format_signals_for_prompt(signals)
                logger.info(f"   📡 Contradiction signals: {len(signals)} similar memories found")
            except Exception as e:
                logger.warning(f"   ⚠️ Contradiction signal detection failed (non-fatal): {e}")
        
        # Get reinforcement candidates (relations that match the new memory)
        reinforcement_candidates_text = self._get_reinforcement_candidates(
            memory_content=memory_content,
            new_relations=new_relations,
            existing_relations=existing_relations
        )
        
        # Analyze memory against entire KG (LLM decides what changes are needed)
        analysis = await self._analyze_relations(
            memory_content=memory_content,
            new_relations=new_relations,
            existing_relations=existing_relations,
            contradiction_signals=contradiction_signals_text,
            reinforcement_candidates=reinforcement_candidates_text
        )
        
        contradictions = analysis.get("contradictions", [])
        reinforcements = analysis.get("reinforcements", [])
        new_relations_to_add = analysis.get("new_relations_to_add", [])
        
        logger.info(f"   📊 Analysis: {len(contradictions)} contradictions, {len(reinforcements)} reinforcements, {len(new_relations_to_add)} new relations to add")
        
        # Process contradictions
        relations_updated = 0
        relations_removed = 0
        
        for contradiction in contradictions:
            old_rel = contradiction["old_relation"]
            new_rel = contradiction["new_relation"]
            resolution = contradiction["resolution"]
            confidence = contradiction.get("confidence", 0.8)
            
            logger.info(f"   ⚠️  CONTRADICTION ({contradiction['type']}):")
            logger.info(f"      Old: {old_rel['subject']} --{old_rel['predicate']}--> {old_rel['object']}")
            logger.info(f"      New: {new_rel['subject']} --{new_rel['predicate']}--> {new_rel['object']}")
            logger.info(f"      Resolution: {resolution} (confidence: {confidence:.2f})")
            logger.info(f"      Reasoning: {contradiction.get('reasoning', 'N/A')}")
            
            if resolution == "replace":
                # Set effective_to on old relation instead of deleting (preserves history)
                superseded = await self._supersede_relation(user_id, old_rel)
                if superseded:
                    relations_updated += 1
                    logger.info(f"      ✅ Superseded old relation (set effective_to)")
            
            elif resolution == "merge":
                # Update old relation's confidence, keep both
                updated = await self._update_relation_confidence(
                    user_id, old_rel, confidence * 0.5  # Lower confidence for merged
                )
                if updated:
                    relations_updated += 1
                    logger.info(f"      ✅ Lowered old relation confidence")
        
        # Process reinforcements (supporting mentions++)
        for reinforcement in reinforcements:
            rel = reinforcement["relation"]
            updated = await self._increment_supporting_mentions(user_id, rel)
            if updated:
                relations_updated += 1
                logger.info(f"   ✅ REINFORCEMENT: {rel['subject']} --{rel['predicate']}--> {rel['object']}")
                logger.info(f"      Reasoning: {reinforcement.get('reasoning', 'N/A')}")
        
        # Process new relations to add (inferred from context)
        relations_added = 0
        for new_rel in new_relations_to_add:
            subject = new_rel.get("subject", "").lower().strip()
            predicate = new_rel.get("predicate", "").lower().strip()
            obj = new_rel.get("object", "").lower().strip()
            confidence = new_rel.get("confidence", 0.7)
            
            if subject and predicate and obj:
                stored = self.kg_store.store_relations(
                    user_id=user_id,
                    memory_id=memory_id,
                    relations=[{
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": confidence
                    }]
                )
                if stored > 0:
                    relations_added += 1
                    logger.info(f"   ✅ ADDED: {subject} --{predicate}--> {obj} (confidence: {confidence:.2f})")
                    logger.info(f"      Reasoning: {new_rel.get('reasoning', 'N/A')}")
        
        logger.info(f"   📈 Summary: {relations_added} added, {relations_updated} updated, {relations_removed} removed")
        logger.info("="*80)
        
        return {
            "contradictions_found": len(contradictions),
            "relations_updated": relations_updated,
            "relations_removed": relations_removed,
            "relations_added": relations_added,
            "reinforcements_found": len(reinforcements)
        }

    def _get_reinforcement_candidates(
        self,
        memory_content: str,
        new_relations: List[Dict[str, Any]],
        existing_relations: List[Dict[str, Any]]
    ) -> str:
        """
        Identify existing relations that might be reinforced by the new memory.
        
        Returns formatted string for LLM prompt.
        """
        candidates = []
        memory_lower = memory_content.lower()
        
        # Extract key terms from new relations
        new_relation_terms = set()
        for rel in new_relations:
            new_relation_terms.add(rel.get("subject", "").lower())
            new_relation_terms.add(rel.get("predicate", "").lower())
            new_relation_terms.add(rel.get("object", "").lower())
        
        # Find existing relations that match
        for rel in existing_relations:
            subject = rel.get("subject", "").lower()
            predicate = rel.get("predicate", "").lower()
            obj = rel.get("object", "").lower()
            
            # Check if any part of the relation appears in the memory or new relations
            if (subject in memory_lower or obj in memory_lower or
                subject in new_relation_terms or obj in new_relation_terms):
                
                mentions = rel.get("supporting_mentions", 0)
                last_mentioned = rel.get("last_mentioned", "N/A")
                created = rel.get("created_at", "N/A")
                
                candidates.append(
                    f"- {subject} --{predicate}--> {obj} "
                    f"(mentions: {mentions}, last: {last_mentioned[:10]}, created: {created[:10]})"
                )
        
        if not candidates:
            return "No reinforcement candidates found."
        
        return "\n".join(candidates[:10])  # Limit to top 10

    async def _analyze_relations(
        self,
        memory_content: str,
        new_relations: List[Dict[str, Any]],
        existing_relations: List[Dict[str, Any]],
        contradiction_signals: str = "No contradiction signals available.",
        reinforcement_candidates: str = "No reinforcement candidates found."
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze relations for contradictions and reinforcements.
        Receives contradiction signals (similar memories + temporal gap) and
        reinforcement candidates (existing relations matching the memory) as additional context.
        """
        try:
            new_rels_str = "\n".join([
                f"- {r['subject']} --{r['predicate']}--> {r['object']} (confidence: {r.get('confidence', 0.5):.2f})"
                for r in new_relations
            ])

            existing_rels_str = "\n".join([
                f"- {r['subject']} --{r['predicate']}--> {r['object']} "
                f"(confidence: {r.get('confidence', 0.5):.2f}, "
                f"mentions: {r.get('supporting_mentions', 0)}, "
                f"created: {r.get('created_at', 'N/A')[:10]}, "
                f"last_mentioned: {r.get('last_mentioned', 'N/A')[:10]})"
                for r in existing_relations[:50]
            ])

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a knowledge graph maintenance agent. Your PRIMARY job is detecting contradictions and reinforcements. Be thorough and decisive."},
                    {"role": "user", "content": CONTRADICTION_DETECTION_PROMPT.format(
                        new_memory=memory_content[:500],
                        new_relations=new_rels_str,
                        existing_relations=existing_rels_str,
                        contradiction_signals=contradiction_signals,
                        reinforcement_candidates=reinforcement_candidates
                    )}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            result = json.loads(content)
            
            if not isinstance(result, dict):
                return {"contradictions": [], "reinforcements": [], "new_relations_to_add": []}
            
            return {
                "contradictions": result.get("contradictions", []),
                "reinforcements": result.get("reinforcements", []),
                "new_relations_to_add": result.get("new_relations_to_add", [])
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"KG maintenance JSON parse failed: {e}")
            return {"contradictions": [], "reinforcements": [], "new_relations_to_add": []}
        except Exception as e:
            logger.warning(f"KG maintenance analysis failed: {e}", exc_info=True)
            return {"contradictions": [], "reinforcements": [], "new_relations_to_add": []}

    async def _supersede_relation(self, user_id: str, relation: Dict[str, str]) -> bool:
        """
        Supersede a relation by setting effective_to (preserves history instead of deleting).
        """
        try:
            from datetime import datetime
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.subject == @subject
                FILTER rel.predicate == @predicate
                FILTER rel.object == @object
                FILTER rel.effective_to == null
                UPDATE rel WITH {
                    effective_to: @effective_to
                } IN entity_relations
                RETURN NEW
            """
            
            cursor = self.kg_store.db.aql.execute(
                query,
                bind_vars={
                    "user_id": user_id,
                    "subject": relation["subject"].lower().strip(),
                    "predicate": relation["predicate"].lower().strip(),
                    "object": relation["object"].lower().strip(),
                    "effective_to": datetime.now().isoformat()
                }
            )
            
            updated = list(cursor)
            return len(updated) > 0
            
        except Exception as e:
            logger.warning(f"Failed to supersede relation: {e}")
            return False

    async def _remove_relation(self, user_id: str, relation: Dict[str, str]) -> bool:
        """
        Remove a relation from the KG (hard delete - use _supersede_relation instead for history preservation).
        TO-DO: Incase of a contradiction, we need to remove the relation from the KG.
        """
        try:
            # Query to find and delete the relation
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.subject == @subject
                FILTER rel.predicate == @predicate
                FILTER rel.object == @object
                REMOVE rel IN entity_relations
                RETURN OLD
            """
            
            cursor = self.kg_store.db.aql.execute(
                query,
                bind_vars={
                    "user_id": user_id,
                    "subject": relation["subject"].lower().strip(),
                    "predicate": relation["predicate"].lower().strip(),
                    "object": relation["object"].lower().strip()
                }
            )
            
            removed = list(cursor)
            return len(removed) > 0
            
        except Exception as e:
            logger.warning(f"Failed to remove relation: {e}")
            return False

    async def _update_relation_confidence(
        self,
        user_id: str,
        relation: Dict[str, str],
        new_confidence: float
    ) -> bool:
        """
        Update the confidence score of a relation.
        """
        try:
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.subject == @subject
                FILTER rel.predicate == @predicate
                FILTER rel.object == @object
                UPDATE rel WITH { confidence: @confidence } IN entity_relations
                RETURN NEW
            """
            
            cursor = self.kg_store.db.aql.execute(
                query,
                bind_vars={
                    "user_id": user_id,
                    "subject": relation["subject"].lower().strip(),
                    "predicate": relation["predicate"].lower().strip(),
                    "object": relation["object"].lower().strip(),
                    "confidence": new_confidence
                }
            )
            
            updated = list(cursor)
            return len(updated) > 0
            
        except Exception as e:
            logger.warning(f"Failed to update relation confidence: {e}")
            return False

    async def _increment_supporting_mentions(
        self,
        user_id: str,
        relation: Dict[str, str]
    ) -> bool:
        """
        Increment supporting mentions for a relation (reinforcement).
        Increases confidence score slightly.
        """
        try:
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.subject == @subject
                FILTER rel.predicate == @predicate
                FILTER rel.object == @object
                LET new_confidence = MIN([rel.confidence + 0.05, 1.0])
                LET new_mentions = (rel.supporting_mentions || 0) + 1
                UPDATE rel WITH { 
                    confidence: new_confidence,
                    supporting_mentions: new_mentions,
                    last_mentioned: @timestamp
                } IN entity_relations
                RETURN NEW
            """
            
            cursor = self.kg_store.db.aql.execute(
                query,
                bind_vars={
                    "user_id": user_id,
                    "subject": relation["subject"].lower().strip(),
                    "predicate": relation["predicate"].lower().strip(),
                    "object": relation["object"].lower().strip(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            updated = list(cursor)
            return len(updated) > 0
            
        except Exception as e:
            logger.warning(f"Failed to increment supporting mentions: {e}")
            return False

    async def cleanup_low_confidence_relations(
        self,
        user_id: str,
        confidence_threshold: float = 0.3,
        age_days: int = 30
    ) -> int:
        """
        Remove low-confidence relations that haven't been reinforced.
        
        Args:
            user_id: User ID
            confidence_threshold: Remove relations below this confidence
            age_days: Only remove if older than this many days
        
        Returns:
            Number of relations removed
        """
        try:
            from datetime import timedelta
            cutoff_date = (datetime.now() - timedelta(days=age_days)).isoformat()
            
            query = """
            FOR rel IN entity_relations
                FILTER rel.user_id == @user_id
                FILTER rel.confidence < @threshold
                FILTER rel.created_at < @cutoff_date
                FILTER (rel.supporting_mentions || 0) == 0
                REMOVE rel IN entity_relations
                RETURN OLD
            """
            
            cursor = self.kg_store.db.aql.execute(
                query,
                bind_vars={
                    "user_id": user_id,
                    "threshold": confidence_threshold,
                    "cutoff_date": cutoff_date
                }
            )
            
            removed = list(cursor)
            count = len(removed)
            
            if count > 0:
                logger.info(f"🧹 Cleaned up {count} low-confidence relations for user {user_id}")
            
            return count
            
        except Exception as e:
            logger.warning(f"Failed to cleanup low-confidence relations: {e}")
            return 0
