"""
Session JSON Parser for MemoryBench / benchmark ingestion.

Parses UnifiedSession format (from MemoryBench) into ConversationChunks.
Each user+assistant message pair becomes a chunk (content=user, context=assistant).
"""

import json
import logging
from typing import List, Any

from .models import ConversationChunk, BaseParser

logger = logging.getLogger(__name__)


class SessionJsonParser(BaseParser):
    """
    Parser for MemoryBench UnifiedSession JSON format.
    
    Expects source to be a JSON string with structure:
    {
        "sessionId": "...",
        "messages": [{"role": "user"|"assistant", "content": "...", ...}],
        "metadata": {...}
    }
    
    Or an array of such sessions.
    """

    @property
    def source_type(self) -> str:
        return "session_json"

    async def parse(self, source: str, **kwargs) -> List[ConversationChunk]:
        """
        Parse JSON session(s) into ConversationChunks.
        
        Each user message + following assistant response becomes one chunk.
        Standalone messages are skipped or paired with empty context.
        """
        try:
            data = json.loads(source)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        sessions = data if isinstance(data, list) else [data]
        chunks: List[ConversationChunk] = []

        for session in sessions:
            session_id = session.get("sessionId", "unknown")
            messages = session.get("messages", [])
            metadata = session.get("metadata", {})
            iso_date = metadata.get("date", "")
            formatted_date = metadata.get("formattedDate", "")

            # Pair user messages with assistant responses
            i = 0
            while i < len(messages):
                msg = messages[i]
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()

                if not content:
                    i += 1
                    continue

                if role == "user":
                    # Look for following assistant response
                    context = None
                    if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                        context = messages[i + 1].get("content", "").strip() or None
                        i += 2  # Consume both
                    else:
                        i += 1

                    chunk_meta = {
                        "session_id": session_id,
                        "session_metadata": metadata,
                        "iso_date": iso_date,
                        "formatted_date": formatted_date,
                    }
                    chunks.append(
                        ConversationChunk(
                            content=content,
                            context=context,
                            timestamp=iso_date,
                            source_type=self.source_type,
                            metadata=chunk_meta,
                        )
                    )
                elif role == "assistant":
                    # Standalone assistant message - use as content with no context
                    chunk_meta = {
                        "session_id": session_id,
                        "session_metadata": metadata,
                        "iso_date": iso_date,
                        "formatted_date": formatted_date,
                    }
                    chunks.append(
                        ConversationChunk(
                            content=content,
                            context=None,
                            timestamp=iso_date,
                            source_type=self.source_type,
                            metadata=chunk_meta,
                        )
                    )
                    i += 1
                else:
                    i += 1

        logger.info(f"Parsed {len(sessions)} session(s) into {len(chunks)} chunks")
        return chunks
