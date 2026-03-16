"""
Memory Ingestion System

Pluggable architecture for importing memories from various sources.
"""

from .models import ConversationChunk, BaseParser
from .ingestion_service import IngestionService

__all__ = [
    'ConversationChunk',
    'BaseParser',
    'IngestionService',
]
