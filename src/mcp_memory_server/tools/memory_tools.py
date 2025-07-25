"""
MCP tools for memory operations: store, retrieve, and search.
"""

import logging
from typing import List, Dict, Any, Optional

# Import the FastMCP app instance, Settings, and embedding provider
# Shared FastMCP app instance
from mcp_memory_server.app_instance import app
from ..services.memory_service import MemoryService, MemoryServiceError
from ..embeddings.ollama import OllamaEmbeddingProvider
from ..storage.chroma import ChromaStorageProvider
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Initialize providers and service
settings = get_settings()
storage_provider = ChromaStorageProvider(
    persist_directory=settings.get_chroma_path(),
    collection_name=settings.chroma_collection_name,
)
embedding_provider = OllamaEmbeddingProvider(
    base_url=settings.ollama_base_url, model=settings.ollama_model
)
service = MemoryService(
    embedding_provider=embedding_provider,
    storage_provider=storage_provider,
    batch_size=settings.max_memories_per_request,
    similarity_threshold=settings.similarity_threshold,
)


@app.tool("store_memories")
async def store_memories(memories: List[Dict[str, Any]], context: str) -> List[str]:
    """
    Store a batch of memories for a given context.

    Returns list of memory IDs.
    """
    # Input validation for context and memory content
    if not isinstance(context, str) or not context.strip():
        logger.error("Invalid context provided to store_memories")
        return []
    if not isinstance(memories, list):
        logger.error("Invalid memories provided to store_memories; expected a list of memory dicts")
        return []
    # Validate each memory dict has non-empty 'content'
    for idx, m in enumerate(memories):
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, str) or not content.strip():
            logger.error(f"Invalid memory at index {idx}: 'content' must be a non-empty string")
            return []
    try:
        await service.initialize()
        stored = await service.store_memories(memories, context)
        return [m.id for m in stored]
    except MemoryServiceError as e:
        logger.error(f"Error in store_memories tool: {e}")
        return []


@app.tool("retrieve_memories")
async def retrieve_memories(
    context: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve memories by context.
    Returns list of memory dicts.
    """
    try:
        await service.initialize()
        results = await service.retrieve_memories(context, limit)
        return [m.model_dump() for m in results]
    except MemoryServiceError as e:
        logger.error(f"Error in retrieve_memories tool: {e}")
        return []


@app.tool("search_memories")
async def search_memories(
    query: str, context: str, limit: int = 10, threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search for memories semantically.
    Returns list of dicts with memory and score.
    """
    try:
        await service.initialize()
        results = await service.search_memories(query, context, limit, threshold)
        # Format output
        return [{"memory": m[0].model_dump(), "score": m[1]} for m in results]
    except MemoryServiceError as e:
        logger.error(f"Error in search_memories tool: {e}")
        return []
