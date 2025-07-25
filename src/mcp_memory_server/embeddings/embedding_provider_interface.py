"""Base interface for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    This interface defines the contract that all embedding providers
    must implement to be compatible with the Memory MCP Server.
    """

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a single text.

        Args:
            text: The text to embed

        Returns:
            List[float]: The embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Returns:
            int: The embedding dimension

        Raises:
            EmbeddingError: If dimension cannot be determined
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the embedding provider is healthy and available.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass


class EmbeddingError(Exception):
    """Exception raised when embedding operations fail."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize EmbeddingError.

        Args:
            message: Error message
            provider: Name of the embedding provider
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.provider:
            return f"[{self.provider}] {super().__str__()}"
        return super().__str__()
