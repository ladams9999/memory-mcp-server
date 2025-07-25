"""Integration tests for MemoryService end-to-end operations."""

import pytest
import shutil

from mcp_memory_server.services.memory_service import MemoryService
from mcp_memory_server.storage.chroma import ChromaStorageProvider
from mcp_memory_server.embeddings.embedding_provider_interface import EmbeddingProvider
from mcp_memory_server.models.memory import Memory


def setup_module(module):
    # Ensure data directory exists
    pass

class DummyEmbeddingProvider(EmbeddingProvider):
    async def get_embedding(self, text: str):
        # Return a constant embedding vector for testing
        return [0.1, 0.1, 0.1]

    async def get_embeddings(self, texts):
        # Return constant embeddings for each text in list
        embeddings = []
        for text in texts:
            embeddings.append(await self.get_embedding(text))
        return embeddings

    async def get_embedding_dimension(self):
        # Return embedding dimension length
        sample = await self.get_embedding("")
        return len(sample)

    async def health_check(self):
        # Always healthy for testing
        return True

import pytest_asyncio

@pytest_asyncio.fixture
async def memory_service(tmp_path):
    # Setup temporary directory for ChromaDB storage
    storage_dir = str(tmp_path / "chromadb")
    provider = ChromaStorageProvider(persist_directory=storage_dir, collection_name="test_memories")
    embedding = DummyEmbeddingProvider()
    service = MemoryService(
        embedding_provider=embedding,
        storage_provider=provider,
        batch_size=5,
        similarity_threshold=0.0,
    )
    await service.initialize()
    yield service
    # Cleanup storage directory
    shutil.rmtree(storage_dir, ignore_errors=True)

@pytest.mark.asyncio
async def test_store_and_retrieve(memory_service):
    # Store memories
    memories = [
        {"content": "First memory test"},
        {"content": "Second memory test"},
    ]
    ids = await memory_service.store_memories(memories, context="test_context")
    assert len(ids) == 2

    # Retrieve memories
    results = await memory_service.retrieve_memories("test_context")
    assert len(results) == 2
    contents = {m.content for m in results}
    assert contents == {"First memory test", "Second memory test"}

@pytest.mark.asyncio
async def test_search_semantic(memory_service):
    # Ensure memories exist
    await memory_service.store_memories([
        {"content": "Alpha test"},
        {"content": "Beta test"},
    ], context="search_context")
    # Perform semantic search
    results = await memory_service.search_memories("Alpha", context="search_context", limit=2, threshold=0.0)
    assert isinstance(results, list)
    assert results
    # Each result is a (Memory, score) tuple
    memory_item, score = results[0]
    assert isinstance(memory_item, Memory)
    assert isinstance(score, float)
