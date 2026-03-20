"""
Memory Ingestion System

Pluggable architecture for importing memories from various sources.
"""

from .models import ConversationChunk, BaseParser
from .ingestion_service import IngestionService
from .parser_adapter_bridge import SourceAdapterParser

__all__ = [
    "BaseParser",
    "ConversationChunk",
    "IngestionService",
    "SourceAdapterParser",
]
