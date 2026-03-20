"""
Plain text and Markdown files from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from .base import SourceAdapter

from services.ingestion.models import ConversationChunk

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".readme"}


class PlainTextFileAdapter(SourceAdapter):
    """
    Reads a UTF-8 text file; emits a single ConversationChunk.

    ``source`` must be a filesystem path.
    """

    category = "file"
    subtype = "text_plain"

    @property
    def source_type(self) -> str:
        return "text_file"

    async def extract(self, source: str, **kwargs: Any) -> List[ConversationChunk]:
        path = Path(source).expanduser()
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        suffix = path.suffix.lower()
        if suffix not in _TEXT_EXTENSIONS and not path.name.lower().endswith("readme"):
            logger.warning(
                "Reading %s as plain text despite extension %s", path, suffix or "(none)"
            )
        text = path.read_text(encoding=kwargs.get("encoding", "utf-8"), errors="replace")
        text = text.strip()
        if not text:
            raise ValueError(f"Empty file: {path}")
        meta = {
            "path": str(path.resolve()),
            "filename": path.name,
            "kind": "markdown" if suffix in (".md", ".markdown") else "text",
        }
        return [
            ConversationChunk(
                content=text,
                context=None,
                timestamp=None,
                source_type=self.source_type,
                metadata=meta,
            )
        ]
