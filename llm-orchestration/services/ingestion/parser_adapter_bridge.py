"""
Bridge ``ingestion.adapters.SourceAdapter`` to legacy ``BaseParser`` for IngestionService.
"""

from __future__ import annotations

from typing import Any, List

from ingestion.adapters.base import SourceAdapter

from .models import BaseParser, ConversationChunk


class SourceAdapterParser(BaseParser):
    """Wraps a :class:`SourceAdapter` so it can be registered on ``IngestionService``."""

    def __init__(self, adapter: SourceAdapter) -> None:
        self._adapter = adapter

    @property
    def source_type(self) -> str:
        return self._adapter.source_type

    async def parse(self, source: str, **kwargs: Any) -> List[ConversationChunk]:
        return await self._adapter.extract(source, **kwargs)
