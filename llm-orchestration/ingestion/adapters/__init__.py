"""Source adapters for the ingestion pipeline."""

from .base import SourceAdapter
from .pdf_files import PdfFileAdapter
from .plain_files import PlainTextFileAdapter
from .text_paste import TextPasteAdapter
from .wrap_legacy import LegacyParserAdapter

__all__ = [
    "LegacyParserAdapter",
    "PdfFileAdapter",
    "PlainTextFileAdapter",
    "SourceAdapter",
    "TextPasteAdapter",
]
