"""
Query Understanding Service
Uses LLM (gpt-4o-mini) to extract entities and intent from user queries.
Replaces spaCy-based entity extraction for better quality.
"""

import json
import logging
from typing import List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EXTRACT_ENTITIES_PROMPT = """Extract core entities from this query to find relevant memories.

Examples:
- "what projects am I working on" → ["project"]
- "tell me about DAPPY" → ["dappy"]
- "what did my sister say" → ["sister"]
- "where does Sarah live" → ["sarah"]
- "what's my liquidation bot doing" → ["liquidation bot"]

Rules:
- Extract the CORE NOUN, not the full phrase
- Skip: pronouns (I, you, my), question words (what, which, where), articles (the, a)
- For categories/types, use singular form: "projects" → "project"
- Keep multi-word names together: "liquidation bot", "new york"
- Return lowercase JSON array

Query: {{query}}

JSON array:"""


class QueryUnderstandingService:
    """
    LLM-based query understanding for memory retrieval.
    Extracts entities for entity-memory expansion and PPR seeding.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def extract_entities(self, query: str) -> List[str]:
        """
        Extract entity names from a user query using LLM.

        Args:
            query: User's query text

        Returns:
            List of entity names (lowercase) for resolution/lookup
        """
        if not query or not query.strip():
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You extract entity names from queries. Return only a JSON array of strings."},
                    {"role": "user", "content": EXTRACT_ENTITIES_PROMPT.replace("{query}", query[:500])}
                ],
                temperature=0.1,
                max_tokens=150
            )
            content = response.choices[0].message.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            entities = json.loads(content)
            if not isinstance(entities, list):
                return []
            result = [str(e).lower().strip() for e in entities if e][:10]
            logger.debug(f"LLM extracted entities: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Query understanding JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"Query understanding failed: {e}", exc_info=True)
            return []
