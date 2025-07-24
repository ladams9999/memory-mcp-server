"""Embedding providers for the Memory MCP Server."""

from .embedding_provider_interface import EmbeddingProvider
from .ollama import OllamaEmbeddingProvider

__all__ = ["EmbeddingProvider", "OllamaEmbeddingProvider"]
