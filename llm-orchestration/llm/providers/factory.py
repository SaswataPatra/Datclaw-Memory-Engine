"""
LLM Provider Factory
Creates LLM provider instances based on configuration
"""

from typing import Literal, Optional, Dict, Any
import logging

from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances
    
    Supports pluggable provider selection via configuration.
    """
    
    @staticmethod
    def create(
        provider_type: Literal["openai", "anthropic", "ollama", "azure"],
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider instance
        
        Args:
            provider_type: Type of provider ("openai", "anthropic", "ollama", "azure")
            **kwargs: Provider-specific configuration
            
        Returns:
            LLMProvider: Configured provider instance
            
        Raises:
            ValueError: If provider_type is unknown
            ImportError: If required dependencies are missing
            
        Examples:
            # OpenAI
            provider = LLMProviderFactory.create(
                "openai",
                api_key="sk-...",
                model="gpt-4-turbo-preview"
            )
            
            # Ollama (local)
            provider = LLMProviderFactory.create(
                "ollama",
                base_url="http://localhost:11434",
                model="llama2"
            )
        """
        logger.info(f"Creating LLM provider: {provider_type}")
        
        if provider_type == "openai":
            return LLMProviderFactory._create_openai(**kwargs)
        
        elif provider_type == "anthropic":
            return LLMProviderFactory._create_anthropic(**kwargs)
        
        elif provider_type == "ollama":
            return LLMProviderFactory._create_ollama(**kwargs)
        
        elif provider_type == "azure":
            return LLMProviderFactory._create_azure(**kwargs)
        
        else:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Supported: openai, anthropic, ollama, azure"
            )
    
    @staticmethod
    def _create_openai(**kwargs) -> OpenAIProvider:
        """Create OpenAI provider"""
        required_keys = ["api_key"]
        missing = [k for k in required_keys if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required OpenAI config: {missing}")
        
        return OpenAIProvider(
            api_key=kwargs["api_key"],
            model=kwargs.get("model", "gpt-4-turbo-preview"),
            organization=kwargs.get("organization"),
            timeout=kwargs.get("timeout", 60.0)
        )
    
    @staticmethod
    def _create_anthropic(**kwargs) -> LLMProvider:
        """Create Anthropic provider"""
        try:
            from .anthropic_provider import AnthropicProvider
        except ImportError:
            raise ImportError(
                "Anthropic provider not implemented yet. "
                "Use 'openai' or 'ollama' for now."
            )
        
        required_keys = ["api_key"]
        missing = [k for k in required_keys if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required Anthropic config: {missing}")
        
        return AnthropicProvider(
            api_key=kwargs["api_key"],
            model=kwargs.get("model", "claude-3-opus-20240229"),
            timeout=kwargs.get("timeout", 60.0)
        )
    
    @staticmethod
    def _create_ollama(**kwargs) -> OllamaProvider:
        """Create Ollama provider"""
        return OllamaProvider(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "llama2"),
            timeout=kwargs.get("timeout", 120.0)
        )
    
    @staticmethod
    def _create_azure(**kwargs) -> LLMProvider:
        """Create Azure OpenAI provider"""
        try:
            from .azure_provider import AzureOpenAIProvider
        except ImportError:
            raise ImportError(
                "Azure OpenAI provider not implemented yet. "
                "Use 'openai' or 'ollama' for now."
            )
        
        required_keys = ["api_key", "endpoint", "deployment_name"]
        missing = [k for k in required_keys if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required Azure config: {missing}")
        
        return AzureOpenAIProvider(
            api_key=kwargs["api_key"],
            endpoint=kwargs["endpoint"],
            deployment_name=kwargs["deployment_name"],
            api_version=kwargs.get("api_version", "2024-02-15-preview"),
            timeout=kwargs.get("timeout", 60.0)
        )
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> LLMProvider:
        """
        Create provider from configuration dictionary
        
        Args:
            config: Configuration dict with 'provider' key and provider-specific config
            
        Returns:
            LLMProvider: Configured provider
            
        Example:
            config = {
                "provider": "openai",
                "openai": {
                    "api_key": "sk-...",
                    "model": "gpt-4-turbo-preview"
                }
            }
            provider = LLMProviderFactory.from_config(config)
        """
        provider_type = config.get("provider")
        if not provider_type:
            raise ValueError("Missing 'provider' key in config")
        
        provider_config = config.get(provider_type, {})
        
        return LLMProviderFactory.create(
            provider_type=provider_type,
            **provider_config
        )

