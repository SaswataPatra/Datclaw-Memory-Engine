"""
DAPPY LLM Orchestration Service - Configuration Module
Loads YAML config with environment variable substitution
"""

import os
import re
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration manager with env var substitution"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to base.yaml in same directory
            config_path = Path(__file__).parent / "base.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config and substitute environment variables"""
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Recursively substitute ${VAR} with environment variables
        return self._substitute_env_vars(raw_config)
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in config"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Match ${VAR_NAME} or ${VAR_NAME:default_value}
            pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'
            
            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2)
                return os.getenv(var_name, default_value or '')
            
            return re.sub(pattern, replace_var, obj)
        else:
            return obj
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path
        
        Example:
            config.get('redis.host')  # Returns 'localhost'
            config.get('llm.api_key')  # Returns value from OPENAI_API_KEY env
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section"""
        return self.get(section, {})
    
    @property
    def all(self) -> Dict[str, Any]:
        """Return entire config"""
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: config['redis']"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists: 'redis' in config"""
        return key in self._config


# Global config instance
_config: Config = None


def load_config(config_path: str = None) -> Config:
    """Load configuration (singleton pattern)"""
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config


def get_config() -> Config:
    """Get current config instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config

