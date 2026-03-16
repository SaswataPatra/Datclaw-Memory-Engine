"""
LLM-based Relation Extraction for Lightweight Graph Building
Extracts (subject, predicate, object) triples from memory text.
Simpler alternative to the full dependency-based pipeline.
"""

import json
import logging
from typing import List, Dict, Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

RELATION_EXTRACTION_PROMPT = """Extract relationships between entities in this memory.
Return a JSON array of objects: [{{"subject": "entity1", "predicate": "relation", "object": "entity2"}}]

IMPORTANT: Extract both explicit and implicit relationships. For example:
- "I'm working on the DAPPY project" → [("user", "works_on", "dappy"), ("dappy", "is_a", "project")]
- "building a liquidation BOT" → [("user", "works_on", "liquidation bot"), ("liquidation bot", "is_a", "project")]
- "My sister Sarah" → [("sarah", "sister_of", "user"), ("sarah", "is_a", "person")]

Use lowercase entity names. Use predicates like:
- Family: sister_of, brother_of, parent_of, spouse_of, child_of
- Professional: works_at, colleague_of, manager_of, reports_to
- Projects: works_on, contributes_to, maintains, created, building
- Interests: interested_in, likes, dislikes, prefers
- Location: located_at, lives_in, based_in
- Ownership: owns, member_of, part_of
- Social: knows, friend_of
- Taxonomy: is_a, type_of, instance_of

Use "user" or "i" for the person speaking. Extract taxonomy relations (is_a) for categorization.

Memory: "{text}"

JSON array:"""


class RelationExtractionService:
    """
    LLM-based relation extraction for lightweight graph building.
    Produces (subject, predicate, object) triples without entity resolution complexity.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def extract_relations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relations from memory text.

        Args:
            text: Memory content

        Returns:
            List of {"subject": str, "predicate": str, "object": str}
        """
        if not text or not text.strip():
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract relationships from text. Return only a JSON array of {subject, predicate, object} objects."},
                    {"role": "user", "content": RELATION_EXTRACTION_PROMPT.format(text=text[:1000])}
                ],
                temperature=0.1,
                max_tokens=400
            )
            content = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            items = json.loads(content)
            if not isinstance(items, list):
                return []
            
            result = []
            for item in items:
                if isinstance(item, dict):
                    subject = str(item.get("subject", "")).lower().strip()
                    predicate = str(item.get("predicate", "")).lower().strip()
                    obj = str(item.get("object", "")).lower().strip()
                    
                    if subject and predicate and obj and len(subject) > 1 and len(obj) > 1:
                        result.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj
                        })
            
            return result[:15]  # Max 15 relations per memory
            
        except json.JSONDecodeError as e:
            logger.warning(f"Relation extraction JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}", exc_info=True)
            return []
