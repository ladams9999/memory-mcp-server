import pytest
import pytest_asyncio
from datetime import datetime

from mcp_memory_server.services.memory_service import MemoryService, MemoryServiceError
from mcp_memory_server.models.memory import Memory
from mcp_memory_server.embeddings.embedding_provider_interface import EmbeddingError
from mcp_memory_server.storage.storage_interface import StorageError


class DummyEmbeddingProvider:
    async def get_embedding(self, text: str):
        return [1.0, 2.0]

    async def get_embeddings(self, texts):
        return [[1.0, 2.0] for _ in texts]
    async def get_embedding_dimension(self) -> int:
        return 2
    async def health_check(self) -> bool:
        return True


class DummyStorageProvider:
    def __init__(self):
        self.stored = []

    async def initialize(self):
        return None

    async def store_memories(self, memories):
        self.stored = memories
        return [m.id for m in memories]

    async def get_memories_by_context(self, context: str, limit=None, since=None):
        return self.stored
    # Other abstract methods are not used by MemoryService and can be omitted


@pytest_asyncio.fixture
async def service():
    embed = DummyEmbeddingProvider()
    storage = DummyStorageProvider()
    # type: ignore[arg-type]
    service = MemoryService(
        embedding_provider=embed,  # type: ignore[arg-type]
        storage_provider=storage,  # type: ignore[arg-type]
        batch_size=10,
        similarity_threshold=0.5
    )
    await service.initialize()
    return service


@pytest.mark.asyncio
async def test_store_memories_sets_embeddings(service):
    data = [{'content': 'hello', 'metadata': {}}, {'content': 'world', 'metadata': {}}]
    memories = await service.store_memories(data, context='ctx')
    # Embeddings should be set
    for mem in memories:
        assert mem.embedding == [1.0, 2.0]
    # Storage should have been called
    stored = service.storage_provider.stored
    assert len(stored) == 2
    assert all(isinstance(m, Memory) for m in stored)


@pytest.mark.asyncio
async def test_retrieve_memories(service):
    # Pre-store a memory
    mem = Memory(content='test', context='ctx', metadata={}, timestamp=datetime.utcnow())
    service.storage_provider.stored = [mem]
    result = await service.retrieve_memories('ctx')
    assert result == [mem]


@pytest.mark.asyncio
async def test_store_memories_error(service):
    # Override embedding provider to throw on single embedding
    async def dummy_embed(text: str):
        raise EmbeddingError('fail')
    service.embedding_provider.get_embedding = dummy_embed
    # Should handle embedding errors gracefully and set embeddings to None
    memories = await service.store_memories([{'content': 'hey', 'metadata': {}}], 'ctx')
    assert len(memories) == 1
    assert memories[0].embedding is None


@pytest.mark.asyncio
async def test_retrieve_memories_error(service):
    service.storage_provider.get_memories_by_context = lambda *args, **kwargs: (_ for _ in ()).throw(StorageError('fail'))
    with pytest.raises(MemoryServiceError):
        await service.retrieve_memories('ctx')
