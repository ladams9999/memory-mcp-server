"""Memory service for handling memory operations with embeddings and storage."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

import numpy as np

from ..models.memory import Memory
from ..embeddings.embedding_provider_interface import EmbeddingProvider, EmbeddingError
from ..storage.storage_interface import StorageProvider, StorageError


logger = logging.getLogger(__name__)


class MemoryServiceError(Exception):
    """Base exception for memory service operations."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class MemoryService:
    """
    Memory service that integrates embedding generation with storage operations.

    This service provides high-level operations for storing, retrieving, and searching
    memories while handling embedding generation and similarity calculations.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        storage_provider: StorageProvider,
        batch_size: int = 10,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the memory service.

        Args:
            embedding_provider: Provider for generating embeddings
            storage_provider: Provider for storing and retrieving memories
            batch_size: Maximum batch size for embedding generation
            similarity_threshold: Default threshold for similarity searches
        """
        self.embedding_provider = embedding_provider
        self.storage_provider = storage_provider
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize both embedding and storage providers."""
        try:
            await self.storage_provider.initialize()
            logger.info("Memory service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory service: {e}")
            raise MemoryServiceError(f"Initialization failed: {e}", cause=e)

    async def store_memories(
        self,
        memories_data: List[Dict[str, Any]],
        context: str,
        generate_embeddings: bool = True,
    ) -> List[Memory]:
        """
        Store multiple memories with optional embedding generation.

        Args:
            memories_data: List of memory data dictionaries
            context: Context identifier for all memories
            generate_embeddings: Whether to generate embeddings for the memories

        Returns:
            List[Memory]: The stored memory objects with generated IDs and embeddings

        Raises:
            MemoryServiceError: If storage or embedding generation fails
        """
        if not memories_data:
            return []

        try:
            # Create Memory objects from the input data
            memories = []
            for data in memories_data:
                memory_dict = {
                    "content": data.get("content", ""),
                    "context": context,
                    "metadata": data.get("metadata", {}),
                    "timestamp": data.get("timestamp", datetime.now(timezone.utc)),
                }
                memories.append(Memory(**memory_dict))

            # Generate embeddings in batches if requested
            if generate_embeddings:
                await self._generate_embeddings_batch(memories)

            # Store all memories
            async with self._lock:
                memory_ids = await self.storage_provider.store_memories(memories)

            logger.info(
                f"Successfully stored {len(memory_ids)} memories in context '{context}'"
            )
            return memories

        except EmbeddingError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise MemoryServiceError(f"Failed to generate embeddings: {e}", cause=e)
        except StorageError as e:
            logger.error(f"Storage operation failed: {e}")
            raise MemoryServiceError(f"Failed to store memories: {e}", cause=e)
        except Exception as e:
            logger.error(f"Unexpected error storing memories: {e}")
            raise MemoryServiceError(f"Unexpected error: {e}", cause=e)

    async def retrieve_memories(
        self,
        context: str,
        limit: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Memory]:
        """
        Retrieve all memories for a given context.

        Args:
            context: Context identifier to retrieve memories for
            limit: Maximum number of memories to retrieve
            metadata_filters: Optional metadata filters to apply (currently not supported by storage interface)

        Returns:
            List[Memory]: The retrieved memories

        Raises:
            MemoryServiceError: If retrieval fails
        """
        try:
            # Note: metadata_filters are not yet supported by the storage interface
            # They would need to be implemented in a future version
            if metadata_filters:
                logger.warning(
                    "Metadata filters are not yet supported by the storage interface"
                )

            memories = await self.storage_provider.get_memories_by_context(
                context=context, limit=limit
            )

            logger.info(f"Retrieved {len(memories)} memories for context '{context}'")
            return memories

        except StorageError as e:
            logger.error(f"Memory retrieval failed: {e}")
            raise MemoryServiceError(f"Failed to retrieve memories: {e}", cause=e)
        except Exception as e:
            logger.error(f"Unexpected error retrieving memories: {e}")
            raise MemoryServiceError(f"Unexpected error: {e}", cause=e)

    async def search_memories(
        self,
        query: str,
        context: str,
        limit: int = 10,
        threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Memory, float]]:
        """
        Perform semantic search for memories within a context.

        Args:
            query: Search query text
            context: Context identifier to search within
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (uses default if None)
            metadata_filters: Optional metadata filters to apply

        Returns:
            List[Tuple[Memory, float]]: List of (memory, similarity_score) tuples

        Raises:
            MemoryServiceError: If search fails
        """
        if not query.strip():
            return []

        threshold = threshold or self.similarity_threshold

        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_provider.get_embedding(query.strip())

            # Retrieve all memories from the context
            memories = await self.retrieve_memories(
                context, metadata_filters=metadata_filters
            )

            if not memories:
                return []

            # Filter memories that have embeddings
            memories_with_embeddings = [m for m in memories if m.embedding is not None]

            if not memories_with_embeddings:
                logger.warning(
                    f"No memories with embeddings found in context '{context}'"
                )
                return []

            # Calculate similarities
            similarities = []
            for memory in memories_with_embeddings:
                if memory.embedding is not None:  # Type guard for mypy
                    similarity = self._calculate_cosine_similarity(
                        query_embedding, memory.embedding
                    )
                    if similarity >= threshold:
                        similarities.append((memory, similarity))

            # Sort by similarity (highest first) and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = similarities[:limit]

            logger.info(
                f"Found {len(results)} memories above threshold {threshold:.2f} "
                f"for query in context '{context}'"
            )

            return results

        except EmbeddingError as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise MemoryServiceError(
                f"Failed to generate query embedding: {e}", cause=e
            )
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise MemoryServiceError(f"Search failed: {e}", cause=e)

    async def _generate_embeddings_batch(self, memories: List[Memory]) -> None:
        """
        Generate embeddings for a list of memories in batches.

        Args:
            memories: List of Memory objects to generate embeddings for

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not memories:
            return

        logger.info(
            f"Generating embeddings for {len(memories)} memories in batches of {self.batch_size}"
        )

        # Process memories in batches
        for i in range(0, len(memories), self.batch_size):
            batch = memories[i : i + self.batch_size]

            # Generate embeddings for this batch
            tasks = []
            for memory in batch:
                if memory.content.strip():  # Only generate for non-empty content
                    task = self.embedding_provider.get_embedding(memory.content)
                    tasks.append((memory, task))

            # Wait for all embeddings in this batch to complete
            if tasks:
                try:
                    # Execute embedding tasks concurrently within the batch
                    results = await asyncio.gather(
                        *[task for _, task in tasks], return_exceptions=True
                    )

                    # Assign embeddings to memories
                    for (memory, _), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Failed to generate embedding for memory {memory.id}: {result}"
                            )
                            # Continue with other memories rather than failing the entire batch
                            memory.embedding = None
                        else:
                            # Type cast since mypy can't infer that result is List[float] here
                            memory.embedding = result  # type: ignore[assignment]

                except Exception as e:
                    logger.error(f"Batch embedding generation failed: {e}")
                    raise EmbeddingError(
                        f"Batch embedding failed: {e}", provider="batch"
                    )

        successful_embeddings = sum(1 for m in memories if m.embedding is not None)
        logger.info(
            f"Successfully generated {successful_embeddings}/{len(memories)} embeddings"
        )

    def _calculate_cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure result is between 0 and 1 (cosine similarity can be negative)
            # Convert from [-1, 1] to [0, 1] range
            return float((similarity + 1) / 2)

        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0

    async def get_memory_stats(self, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Args:
            context: Optional context to get stats for (all contexts if None)

        Returns:
            Dict[str, Any]: Dictionary containing memory statistics
        """
        try:
            if context:
                memories = await self.retrieve_memories(context)
                total_memories = len(memories)
                memories_with_embeddings = sum(
                    1 for m in memories if m.embedding is not None
                )
                contexts = [context] if memories else []
            else:
                # This would require additional storage provider methods to get global stats
                # For now, return basic info
                total_memories = 0
                memories_with_embeddings = 0
                contexts = []

            return {
                "total_memories": total_memories,
                "memories_with_embeddings": memories_with_embeddings,
                "embedding_coverage": memories_with_embeddings / total_memories
                if total_memories > 0
                else 0.0,
                "contexts": contexts,
                "context_count": len(contexts),
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "total_memories": 0,
                "memories_with_embeddings": 0,
                "embedding_coverage": 0.0,
                "contexts": [],
                "context_count": 0,
                "error": str(e),
            }

    async def close(self) -> None:
        """Close the memory service and clean up resources."""
        try:
            # Check if the embedding provider has a close method (like OllamaEmbeddingProvider)
            if hasattr(self.embedding_provider, "close") and callable(
                getattr(self.embedding_provider, "close")
            ):
                close_method = getattr(self.embedding_provider, "close")
                await close_method()

            # Close storage provider
            await self.storage_provider.close()

            logger.info("Memory service closed successfully")
        except Exception as e:
            logger.error(f"Error closing memory service: {e}")
