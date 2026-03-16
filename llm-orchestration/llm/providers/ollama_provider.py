"""
Ollama LLM Provider
Implements LLMProvider interface for local Ollama models
"""

from typing import List, AsyncIterator, Optional
import logging
import json

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx package not installed. Install with: pip install httpx"
    )

from .base import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    LLMProviderError,
    LLMProviderConnectionError,
    LLMProviderTimeoutError
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama LLM Provider (Local Models)
    
    Supports running models locally via Ollama:
    - llama2, llama3
    - mistral, mixtral
    - codellama
    - And many more!
    
    Free to use, no API keys required.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: float = 120.0
    ):
        """
        Initialize Ollama provider
        
        Args:
            base_url: Ollama server URL
            model: Model name (e.g., "llama2", "mistral", "codellama")
            timeout: Request timeout in seconds (longer for local models)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._name = f"ollama-{model}"
        self.client = httpx.AsyncClient(timeout=timeout)
        
        logger.info(f"Initialized Ollama provider with model: {model} at {base_url}")
    
    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion using Ollama
        
        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Ignored (use chat_stream for streaming)
            **kwargs: Additional Ollama parameters
            
        Returns:
            LLMResponse with completion and metadata
        """
        try:
            # Convert to Ollama message format
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            logger.debug(f"Calling Ollama API with {len(messages)} messages")
            
            # Build request payload
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            # Merge additional options
            if kwargs:
                payload["options"].update(kwargs)
            
            # Call Ollama API
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            logger.debug(
                f"Ollama response received: "
                f"eval_count={data.get('eval_count', 0)}, "
                f"done={data.get('done', False)}"
            )
            
            return LLMResponse(
                content=data["message"]["content"],
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                },
                finish_reason="stop" if data.get("done") else "length",
                metadata={
                    "created_at": data.get("created_at"),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "eval_duration": data.get("eval_duration")
                }
            )
            
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout error: {e}")
            raise LLMProviderTimeoutError(f"Ollama request timed out: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise LLMProviderError(f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}", exc_info=True)
            raise LLMProviderConnectionError(f"Failed to connect to Ollama: {e}")
    
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
            **kwargs: Additional Ollama parameters
            
        Yields:
            str: Chunks of the response
        """
        try:
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            logger.debug(f"Starting Ollama streaming with {len(messages)} messages")
            
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            if kwargs:
                payload["options"].update(kwargs)
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse Ollama stream line: {line}")
                            continue
                            
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout error: {e}")
            raise LLMProviderTimeoutError(f"Ollama streaming timed out: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise LLMProviderError(f"Ollama streaming error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama streaming: {e}", exc_info=True)
            raise LLMProviderConnectionError(f"Failed to stream from Ollama: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if Ollama server is accessible
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            is_healthy = response.status_code == 200
            
            if is_healthy:
                logger.debug(f"Ollama health check passed at {self.base_url}")
            else:
                logger.warning(f"Ollama health check failed: status {response.status_code}")
                
            return is_healthy
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    @property
    def name(self) -> str:
        """Provider name"""
        return self._name
    
    @property
    def supports_streaming(self) -> bool:
        """Ollama supports streaming"""
        return True
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

