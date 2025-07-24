"""Configuration settings for the Memory MCP Server."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    For MVP, we focus on ChromaDB + Ollama configuration only.
    """
    
    # Storage Backend Configuration (MVP: ChromaDB only)
    storage_backend: Literal["chroma"] = Field(
        default="chroma",
        description="Storage backend to use (MVP supports ChromaDB only)"
    )
    
    # ChromaDB Settings
    chroma_path: str = Field(
        default="./data/chroma_db",
        description="Path to ChromaDB storage directory"
    )
    
    chroma_collection_name: str = Field(
        default="memories",
        description="Name of the ChromaDB collection for storing memories"
    )
    
    # Embedding Provider Configuration (MVP: Ollama only)
    embedding_provider: Literal["ollama"] = Field(
        default="ollama",
        description="Embedding provider to use (MVP supports Ollama only)"
    )
    
    # Ollama Settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    
    ollama_model: str = Field(
        default="mxbai-embed-large",
        description="Ollama embedding model to use"
    )
    
    # General Configuration
    max_memories_per_request: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of memories that can be stored in a single request"
    )
    
    default_search_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default limit for search results"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default similarity threshold for semantic search"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Field Validators (Pydantic V2)
    @field_validator("chroma_path")
    @classmethod
    def validate_chroma_path(cls, v: str) -> str:
        """Ensure ChromaDB path is valid and create directory if needed."""
        path = Path(v)
        
        # Create the directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create ChromaDB directory '{v}': {e}")
        
        # Check if the path is writable
        if not os.access(path, os.W_OK):
            raise ValueError(f"ChromaDB directory '{v}' is not writable")
        
        return str(path.resolve())
    
    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Validate Ollama base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must start with http:// or https://")
        
        # Remove trailing slash for consistency
        return v.rstrip("/")
    
    @field_validator("chroma_collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """Validate ChromaDB collection name."""
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Collection name must be non-empty and contain only alphanumeric characters, hyphens, and underscores"
            )
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v_upper
    
    # Pydantic V2 configuration
    model_config = ConfigDict(
        # Load from .env file if it exists
        env_file=".env",
        env_file_encoding="utf-8",
        # Make environment variable names case-insensitive
        case_sensitive=False,
        # Example configuration for documentation
        json_schema_extra={
            "example": {
                "storage_backend": "chroma",
                "chroma_path": "./data/chroma_db",
                "chroma_collection_name": "memories",
                "embedding_provider": "ollama",
                "ollama_base_url": "http://localhost:11434",
                "ollama_model": "mxbai-embed-large",
                "max_memories_per_request": 100,
                "default_search_limit": 10,
                "similarity_threshold": 0.7,
                "log_level": "INFO"
            }
        }
    )
    
    def get_chroma_path(self) -> Path:
        """Get ChromaDB path as a Path object."""
        return Path(self.chroma_path)
    
    def get_ollama_embed_url(self) -> str:
        """Get the complete Ollama embeddings API URL."""
        return f"{self.ollama_base_url}/api/embeddings"
    
    def __str__(self) -> str:
        """String representation of settings (excluding sensitive data)."""
        return (
            f"Settings("
            f"storage_backend={self.storage_backend}, "
            f"chroma_path={self.chroma_path}, "
            f"embedding_provider={self.embedding_provider}, "
            f"ollama_model={self.ollama_model}"
            f")"
        )


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    This implements a simple singleton pattern to ensure consistent
    configuration across the application.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload of settings from environment variables.
    
    Useful for testing or when configuration has changed.
    """
    global _settings
    _settings = Settings()
    return _settings
