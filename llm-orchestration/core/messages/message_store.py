"""
DAPPY LLM Orchestration Service - Message Storage Interface
Pluggable raw conversation storage with hot/long-term/cold tiers
"""

from typing import Protocol, Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class Message:
    """Raw conversation message"""
    message_id: str
    user_id: str
    session_id: str
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    observed_at: datetime
    sequence: int
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        data = asdict(self)
        # Convert datetime to ISO format
        data['observed_at'] = self.observed_at.isoformat()
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Deserialize from dictionary"""
        # Convert ISO strings back to datetime
        if isinstance(data.get('observed_at'), str):
            data['observed_at'] = datetime.fromisoformat(data['observed_at'])
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class MessageStore(Protocol):
    """
    Abstract interface for message storage backends
    
    Implementations:
    - RedisMessageBus: Hot tier (14 days)
    - ArangoMessageStore: Long-term (Phase 1)
    - MongoMessageStore: Long-term (Phase 1.5, cost-optimized)
    - S3MessageArchiver: Cold tier (6+ months)
    """
    
    async def append(
        self,
        message_id: str,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        observed_at: datetime,
        sequence: int,
        metadata: Optional[Dict] = None
    ) -> None:
        """Append a message to the store"""
        ...
    
    async def get_recent(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50
    ) -> List[Message]:
        """Get recent messages (hot tier)"""
        ...
    
    async def get_range(
        self,
        user_id: str,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Message]:
        """Get messages in time range (may hit cold tier)"""
        ...
    
    async def link_to_memory(
        self,
        message_id: str,
        memory_node_id: str
    ) -> None:
        """Link message to memory node (provenance)"""
        ...

