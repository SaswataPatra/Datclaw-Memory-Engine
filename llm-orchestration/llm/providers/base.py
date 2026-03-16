"""
LLM Provider Base Classes and Protocols
Abstract interface for pluggable LLM providers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMMessage:
    """Represents a message in a conversation"""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider"""
    content: str
    model: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers
    
    All LLM providers (OpenAI, Anthropic, Ollama, etc.) must implement this interface.
    This enables pluggable architecture where providers can be swapped via configuration.
    """
    
    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response (not used in this method)
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse with content and metadata
            
        Raises:
            LLMProviderError: If the API call fails
        """
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate streaming chat completion
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
            
        Yields:
            str: Chunks of the response as they arrive
            
        Raises:
            LLMProviderError: If the API call fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is available and healthy
        
        Returns:
            bool: True if provider is healthy, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provider name (e.g., "openai-gpt-4", "anthropic-claude-3")
        
        Returns:
            str: Human-readable provider name
        """
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Whether this provider supports streaming responses
        
        Returns:
            bool: True if streaming is supported
        """
        pass


class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass


class LLMProviderConnectionError(LLMProviderError):
    """Raised when connection to provider fails"""
    pass


class LLMProviderTimeoutError(LLMProviderError):
    """Raised when provider request times out"""
    pass


class LLMProviderRateLimitError(LLMProviderError):
    """Raised when provider rate limit is exceeded"""
    pass


class LLMProviderAuthenticationError(LLMProviderError):
    """Raised when authentication fails"""
    pass

