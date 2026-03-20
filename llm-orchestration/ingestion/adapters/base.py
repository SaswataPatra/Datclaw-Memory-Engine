"""
Abstract adapter: extract content into ConversationChunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, List, Optional

if TYPE_CHECKING:
    from services.ingestion.models import ConversationChunk


class SourceAdapter(ABC):
    """
    One adapter per source (or family).

    Subclasses should set class attributes for auto-discovery:
    - ``category``: SourceCategory value (e.g. "llm_chat", "file")
    - ``subtype``: SourceSubtype value (e.g. "chatgpt", "pdf")
    - ``enabled``: bool (default True; set False to disable auto-registration)
    """

    category: ClassVar[Optional[str]] = None
    subtype: ClassVar[Optional[str]] = None
    enabled: ClassVar[bool] = True

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Registry key (e.g. ``chatgpt``, ``text_file``, ``pdf_file``)."""

    @abstractmethod
    async def extract(self, source: str, **kwargs: Any) -> List["ConversationChunk"]:
        """Fetch/read ``source`` and return conversation chunks."""
