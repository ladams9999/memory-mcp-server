"""Base interface for storage providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.memory import Memory


class StorageError(Exception):
    """Exception raised when storage operations fail."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize StorageError.

        Args:
            message: Error message
            provider: Name of the storage provider
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


class StorageProvider(ABC):
    """
    Abstract base class for storage providers.

    This interface defines the contract that all storage providers
    must implement to be compatible with the Memory MCP Server.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage provider.

        This method should create any necessary resources like
        collections, indexes, or connections.

        Raises:
            StorageError: If initialization fails
        """
        pass

    @abstractmethod
    async def store_memory(self, memory: Memory) -> str:
        """
        Store a single memory.

        Args:
            memory: The memory to store

        Returns:
            str: The ID of the stored memory

        Raises:
            StorageError: If storage fails
        """
        pass

    @abstractmethod
    async def store_memories(self, memories: List[Memory]) -> List[str]:
        """
        Store multiple memories in batch.

        Args:
            memories: List of memories to store

        Returns:
            List[str]: List of IDs of the stored memories

        Raises:
            StorageError: If storage fails
        """
        pass

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            Optional[Memory]: The memory if found, None otherwise

        Raises:
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_memories(self, memory_ids: List[str]) -> List[Optional[Memory]]:
        """
        Retrieve multiple memories by IDs.

        Args:
            memory_ids: List of memory IDs to retrieve

        Returns:
            List[Optional[Memory]]: List of memories (None for not found)

        Raises:
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.0,
        context_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for memories using semantic similarity.

        Args:
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score to include
            context_filter: Optional context to filter by
            metadata_filter: Optional metadata filters

        Returns:
            List[Dict[str, Any]]: List of search results with memories and scores
            Each result should have keys: 'memory', 'score', 'distance'

        Raises:
            StorageError: If search fails
        """
        pass

    @abstractmethod
    async def get_memories_by_context(
        self,
        context: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> List[Memory]:
        """
        Retrieve memories by context.

        Args:
            context: The context to filter by
            limit: Maximum number of memories to return
            since: Only return memories created after this timestamp

        Returns:
            List[Memory]: List of memories matching the context

        Raises:
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            bool: True if deleted, False if not found

        Raises:
            StorageError: If deletion fails
        """
        pass

    @abstractmethod
    async def count_memories(self, context_filter: Optional[str] = None) -> int:
        """
        Count the total number of memories.

        Args:
            context_filter: Optional context to filter by

        Returns:
            int: Number of memories

        Raises:
            StorageError: If counting fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the storage provider is healthy and available.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the storage provider and clean up resources.
        """
        pass
