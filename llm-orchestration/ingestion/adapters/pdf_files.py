"""
PDF extraction (optional dependency: pypdf).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from .base import SourceAdapter

from services.ingestion.models import ConversationChunk

logger = logging.getLogger(__name__)


class PdfFileAdapter(SourceAdapter):
    """
    Extracts text from a PDF using pypdf (PyPDF2).

    Install: ``pip install pypdf`` (or add to requirements-optional).
    """

    category = "file"
    subtype = "pdf"

    @property
    def source_type(self) -> str:
        return "pdf_file"

    async def extract(self, source: str, **kwargs: Any) -> List[ConversationChunk]:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "PDF ingestion requires the 'pypdf' package. "
                "Install with: pip install pypdf"
            ) from e

        path = Path(source).expanduser()
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {path}")

        reader = PdfReader(str(path))
        parts: List[str] = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception as ex:
                logger.warning("Page %s extract failed: %s", i, ex)
                t = ""
            if t.strip():
                parts.append(t.strip())

        text = "\n\n".join(parts).strip()
        if not text:
            raise ValueError(f"No extractable text in PDF: {path}")

        meta = {
            "path": str(path.resolve()),
            "filename": path.name,
            "page_count": len(reader.pages),
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
