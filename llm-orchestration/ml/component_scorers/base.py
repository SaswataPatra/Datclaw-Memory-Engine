from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ScorerResult:
    """Result of a single component scorer"""
    score: float
    metadata: Optional[Dict[str, Any]] = None


class ComponentScorer(ABC):
    """Abstract base class for all ego scoring components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def score(self, memory: Dict[str, Any], **kwargs) -> ScorerResult:
        """
        Calculate a score for a given memory.
        Returns a score between 0.0 and 1.0.
        """
        pass
