"""
Data models for memory ingestion system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ConversationChunk:
    """
    Standardized intermediate format for ingested memories.
    
    All parsers convert their source format into a list of ConversationChunks.
    This allows the ingestion service to be format-agnostic.
    """
    content: str
    context: Optional[str] = None
    timestamp: Optional[str] = None
    source_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.content or not self.content.strip():
            raise ValueError("ConversationChunk content cannot be empty")


class BaseParser(ABC):
    """
    Abstract base class for all ingestion parsers.
    
    Each parser is responsible for:
    1. Fetching/reading data from its source format
    2. Converting it into a list of ConversationChunks
    """
    
    @abstractmethod
    async def parse(self, source: str, **kwargs) -> List[ConversationChunk]:
        """
        Parse the source and return a list of conversation chunks.
        
        Args:
            source: Source identifier (URL, file path, etc.)
            **kwargs: Parser-specific options
            
        Returns:
            List of ConversationChunk objects
            
        Raises:
            ValueError: If source is invalid or cannot be parsed
            Exception: For network errors, file errors, etc.
        """
        pass
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier for this parser."""
        pass
