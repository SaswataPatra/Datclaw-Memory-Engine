"""
Raw text pasted from the browser (no server-side file path).
"""

from __future__ import annotations

from typing import Any, List

from .base import SourceAdapter

from services.ingestion.models import ConversationChunk


class TextPasteAdapter(SourceAdapter):
    """``source`` is the full text body (UTF-8)."""

    category = "file"
    subtype = "paste"

    @property
    def source_type(self) -> str:
        return "text_paste"

    async def extract(self, source: str, **kwargs: Any) -> List[ConversationChunk]:
        text = (source or "").strip()
        if not text:
            raise ValueError("Empty text content")
        return [
            ConversationChunk(
                content=text,
                context=None,
                timestamp=None,
                source_type=self.source_type,
                metadata={"kind": "paste"},
            )
        ]
