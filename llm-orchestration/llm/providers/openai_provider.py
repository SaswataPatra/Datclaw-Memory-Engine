"""
OpenAI LLM Provider
Implements LLMProvider interface for OpenAI's GPT models
"""

from typing import List, AsyncIterator, Optional
import logging

try:
    from openai import AsyncOpenAI, OpenAIError, APITimeoutError, RateLimitError, AuthenticationError
except ImportError:
    raise ImportError(
        "OpenAI package not installed. Install with: pip install openai"
    )

from .base import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    LLMProviderError,
    LLMProviderConnectionError,
    LLMProviderTimeoutError,
    LLMProviderRateLimitError,
    LLMProviderAuthenticationError
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM Provider
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        organization: Optional[str] = None,
        timeout: float = 60.0
    ):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4-turbo-preview", "gpt-3.5-turbo")
            organization: Optional organization ID
            timeout: Request timeout in seconds
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            timeout=timeout
        )
        self.model = model
        self._name = f"openai-{model}"
        
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion using OpenAI API
        
        Args:
            messages: Conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Ignored (use chat_stream for streaming)
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLMResponse with completion and metadata
        """
        try:
            # Convert to OpenAI message format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            logger.debug(f"Calling OpenAI API with {len(messages)} messages")
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            choice = response.choices[0]
            
            logger.debug(
                f"OpenAI response received: {response.usage.total_tokens} tokens, "
                f"finish_reason: {choice.finish_reason}"
            )
            
            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=choice.finish_reason,
                metadata={
                    "response_id": response.id,
                    "created": response.created
                }
            )
            
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise LLMProviderAuthenticationError(f"OpenAI authentication failed: {e}")
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit error: {e}")
            raise LLMProviderRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except APITimeoutError as e:
            logger.error(f"OpenAI timeout error: {e}")
            raise LLMProviderTimeoutError(f"OpenAI request timed out: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}", exc_info=True)
            raise LLMProviderConnectionError(f"Failed to connect to OpenAI: {e}")
    
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
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
            
        Yields:
            str: Chunks of the response
        """
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            logger.debug(f"Starting OpenAI streaming with {len(messages)} messages")
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise LLMProviderAuthenticationError(f"OpenAI authentication failed: {e}")
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit error: {e}")
            raise LLMProviderRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except APITimeoutError as e:
            logger.error(f"OpenAI timeout error: {e}")
            raise LLMProviderTimeoutError(f"OpenAI request timed out: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI streaming: {e}", exc_info=True)
            raise LLMProviderConnectionError(f"Failed to stream from OpenAI: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Try to retrieve model info
            await self.client.models.retrieve(self.model)
            logger.debug(f"OpenAI health check passed for model: {self.model}")
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    @property
    def name(self) -> str:
        """Provider name"""
        return self._name
    
    @property
    def supports_streaming(self) -> bool:
        """OpenAI supports streaming"""
        return True

