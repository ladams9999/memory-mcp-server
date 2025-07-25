import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch

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
    mem = Memory(content='test', context='ctx', metadata={}, timestamp=datetime.now(timezone.utc))
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


@pytest.mark.asyncio
async def test_search_memories_success(service):
    """Test successful semantic search."""
    # Pre-store memories with embeddings
    mem1 = Memory(content='hello world', context='ctx', embedding=[1.0, 0.0])
    mem2 = Memory(content='goodbye world', context='ctx', embedding=[0.0, 1.0])
    service.storage_provider.stored = [mem1, mem2]
    
    # Mock embedding provider to return query embedding
    async def mock_get_embedding(text):
        return [1.0, 0.0]  # Similar to mem1
    service.embedding_provider.get_embedding = mock_get_embedding
    
    results = await service.search_memories('hello', 'ctx', limit=5, threshold=0.5)
    
    # Should return both memories, but mem1 with higher similarity
    assert len(results) == 2
    # Results should be sorted by similarity (highest first)
    assert results[0][0] == mem1  # mem1 should be first (highest similarity)
    assert results[0][1] >= results[1][1]  # First result should have higher or equal similarity
    assert results[0][1] >= 0.5  # Should have high similarity


@pytest.mark.asyncio
async def test_search_memories_empty_query(service):
    """Test search with empty query returns empty results."""
    results = await service.search_memories('', 'ctx')
    assert results == []
    
    results = await service.search_memories('   ', 'ctx')
    assert results == []


@pytest.mark.asyncio
async def test_search_memories_no_memories(service):
    """Test search when no memories exist in context."""
    service.storage_provider.stored = []
    
    async def mock_get_embedding(text):
        return [1.0, 0.0]
    service.embedding_provider.get_embedding = mock_get_embedding
    
    results = await service.search_memories('hello', 'ctx')
    assert results == []


@pytest.mark.asyncio
async def test_search_memories_no_embeddings(service):
    """Test search when memories have no embeddings."""
    # Store memories without embeddings
    mem1 = Memory(content='hello world', context='ctx', embedding=None)
    mem2 = Memory(content='goodbye world', context='ctx', embedding=None)
    service.storage_provider.stored = [mem1, mem2]
    
    async def mock_get_embedding(text):
        return [1.0, 0.0]
    service.embedding_provider.get_embedding = mock_get_embedding
    
    results = await service.search_memories('hello', 'ctx')
    assert results == []


@pytest.mark.asyncio
async def test_search_memories_embedding_error(service):
    """Test search when query embedding generation fails."""
    async def mock_get_embedding(text):
        raise EmbeddingError('Embedding failed')
    service.embedding_provider.get_embedding = mock_get_embedding
    
    with pytest.raises(MemoryServiceError) as exc_info:
        await service.search_memories('hello', 'ctx')
    
    assert 'Failed to generate query embedding' in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_memory_stats_with_context(service):
    """Test get_memory_stats for specific context."""
    # Pre-store memories
    mem1 = Memory(content='test1', context='ctx', embedding=[1.0, 0.0])
    mem2 = Memory(content='test2', context='ctx', embedding=None)
    service.storage_provider.stored = [mem1, mem2]
    
    stats = await service.get_memory_stats('ctx')
    
    assert stats['total_memories'] == 2
    assert stats['memories_with_embeddings'] == 1
    assert stats['embedding_coverage'] == 0.5
    assert stats['contexts'] == ['ctx']
    assert stats['context_count'] == 1


@pytest.mark.asyncio
async def test_get_memory_stats_global(service):
    """Test get_memory_stats for all contexts."""
    stats = await service.get_memory_stats()
    
    # Should return basic structure for global stats
    assert 'total_memories' in stats
    assert 'memories_with_embeddings' in stats
    assert 'embedding_coverage' in stats
    assert 'contexts' in stats
    assert 'context_count' in stats


@pytest.mark.asyncio
async def test_close_service(service):
    """Test closing the memory service."""
    # Mock the close methods
    service.storage_provider.close = Mock(return_value=None)
    
    # Add close method to embedding provider
    async def mock_close():
        pass
    service.embedding_provider.close = mock_close
    
    await service.close()
    
    # Verify close was called on storage provider
    service.storage_provider.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_service_error_handling(service):
    """Test close service with error handling."""
    # Make storage provider close raise an error
    async def failing_close():
        raise Exception('Close failed')
    service.storage_provider.close = failing_close
    
    # Should not raise an exception, just log the error
    await service.close()


@pytest.mark.asyncio
async def test_store_memories_without_embeddings(service):
    """Test storing memories without generating embeddings."""
    data = [{'content': 'hello', 'metadata': {}}, {'content': 'world', 'metadata': {}}]
    memories = await service.store_memories(data, context='ctx', generate_embeddings=False)
    
    # Embeddings should be None
    for mem in memories:
        assert mem.embedding is None
    
    # Storage should have been called
    stored = service.storage_provider.stored
    assert len(stored) == 2


@pytest.mark.asyncio
async def test_retrieve_memories_with_metadata_filters_warning(service):
    """Test that metadata filters generate a warning."""
    import logging
    
    # Capture log messages
    with patch('mcp_memory_server.services.memory_service.logger') as mock_logger:
        await service.retrieve_memories('ctx', metadata_filters={'key': 'value'})
        
        # Should log a warning about unsupported metadata filters
        mock_logger.warning.assert_called_once_with(
            'Metadata filters are not yet supported by the storage interface'
        )


@pytest.mark.asyncio
async def test_cosine_similarity_calculation(service):
    """Test cosine similarity calculation."""
    # Test identical vectors
    sim = service._calculate_cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 0.001  # Should be 1.0 (perfect similarity)
    
    # Test orthogonal vectors
    sim = service._calculate_cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(sim - 0.5) < 0.001  # Should be 0.5 (no similarity in cosine space)
    
    # Test opposite vectors
    sim = service._calculate_cosine_similarity([1.0, 0.0], [-1.0, 0.0])
    assert abs(sim - 0.0) < 0.001  # Should be 0.0 (opposite)


@pytest.mark.asyncio
async def test_cosine_similarity_zero_vectors(service):
    """Test cosine similarity with zero vectors."""
    # Test with zero vector
    sim = service._calculate_cosine_similarity([0.0, 0.0], [1.0, 0.0])
    assert sim == 0.0  # Should handle zero division gracefully
    
    sim = service._calculate_cosine_similarity([1.0, 0.0], [0.0, 0.0])
    assert sim == 0.0  # Should handle zero division gracefully


@pytest.mark.asyncio
async def test_batch_embedding_generation(service):
    """Test batch embedding generation."""
    # Create memories with content
    mem1 = Memory(content='hello', context='ctx')
    mem2 = Memory(content='world', context='ctx')
    mem3 = Memory(content=' ', context='ctx')  # Whitespace-only content
    memories = [mem1, mem2, mem3]
    
    # Mock embedding provider to track calls
    call_count = 0
    async def mock_get_embedding(text):
        nonlocal call_count
        call_count += 1
        return [float(call_count), 0.0]
    
    service.embedding_provider.get_embedding = mock_get_embedding
    
    await service._generate_embeddings_batch(memories)
    
    # Should generate embeddings for non-empty content
    assert mem1.embedding == [1.0, 0.0]
    assert mem2.embedding == [2.0, 0.0]
    assert mem3.embedding is None  # Whitespace-only content should not get embedding
    
    # Should have called embedding provider twice (not for whitespace-only content)
    assert call_count == 2


@pytest.mark.asyncio
async def test_batch_embedding_error_handling(service):
    """Test batch embedding with some failures."""
    mem1 = Memory(content='hello', context='ctx')
    mem2 = Memory(content='world', context='ctx')
    memories = [mem1, mem2]
    
    # Mock embedding provider to fail on second call
    call_count = 0
    async def mock_get_embedding(text):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception('Embedding failed')
        return [1.0, 0.0]
    
    service.embedding_provider.get_embedding = mock_get_embedding
    
    await service._generate_embeddings_batch(memories)
    
    # First memory should have embedding, second should be None
    assert mem1.embedding == [1.0, 0.0]
    assert mem2.embedding is None


class TestMemoryServiceError:
    """Test MemoryServiceError exception class."""
    
    def test_memory_service_error_basic(self):
        """Test basic MemoryServiceError creation."""
        error = MemoryServiceError("Test error")
        assert str(error) == "Test error"
        assert error.cause is None
    
    def test_memory_service_error_with_cause(self):
        """Test MemoryServiceError with cause."""
        original_error = ValueError("Original error")
        error = MemoryServiceError("Test error", cause=original_error)
        
        assert str(error) == "Test error"
        assert error.cause == original_error
