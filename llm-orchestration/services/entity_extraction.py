"""
LLM-based Entity Extraction for Ingestion
Extracts entities from memory text for entity-memory bipartite linking.
Replaces spaCy + relation extraction pipeline.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EXTRACT_ENTITIES_PROMPT = """Extract entities mentioned in this memory text.
Focus on: people, places, organizations, projects, topics, and specific things.
Return ONLY a JSON array of objects: [{{"name": "entity_name", "type": "person|place|org|project|topic|other"}}]
Use lowercase for names. Skip pronouns (I, you, my) and generic words. Max 10 entities.

Memory: "{text}"

JSON array:"""


class EntityExtractionService:
    """
    LLM-based entity extraction for memory ingestion.
    Produces entity names for entity-memory bipartite linking.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from memory text.

        Args:
            text: Memory content

        Returns:
            List of {"name": str, "type": str}
        """
        if not text or not text.strip():
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract entity names from text. Return only a JSON array of {name, type} objects."},
                    {"role": "user", "content": EXTRACT_ENTITIES_PROMPT.format(text=text[:1000])}
                ],
                temperature=0.1,
                max_tokens=300
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            items = json.loads(content)
            if not isinstance(items, list):
                return []
            result = []
            for item in items:
                if isinstance(item, dict) and item.get("name"):
                    name = str(item["name"]).lower().strip()
                    etype = str(item.get("type", "other")).lower()
                    if name and len(name) > 1:
                        result.append({"name": name, "type": etype})
            return result[:10]
        except json.JSONDecodeError as e:
            logger.warning(f"Entity extraction JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}", exc_info=True)
            return []
