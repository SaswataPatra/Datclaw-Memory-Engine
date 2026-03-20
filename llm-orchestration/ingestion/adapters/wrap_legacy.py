"""
Wrap existing ``services.ingestion`` BaseParser implementations as SourceAdapters.
"""

from __future__ import annotations

from typing import Any, List

from .base import SourceAdapter

from services.ingestion.models import BaseParser, ConversationChunk


class LegacyParserAdapter(SourceAdapter):
    """Wraps a BaseParser instance."""

    # Legacy parsers don't have category/subtype metadata; leave None
    category = None
    subtype = None

    def __init__(self, parser: BaseParser):
        self._parser = parser

    @property
    def source_type(self) -> str:
        return self._parser.source_type

    async def extract(self, source: str, **kwargs: Any) -> List[ConversationChunk]:
        return await self._parser.parse(source, **kwargs)
