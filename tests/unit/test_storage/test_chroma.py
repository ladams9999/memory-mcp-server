"""Tests for the ChromaDB storage provider."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from mcp_memory_server.storage.chroma import ChromaStorageProvider
from mcp_memory_server.storage.storage_interface import StorageError
from mcp_memory_server.models.memory import Memory


class TestChromaStorageProviderInit:
    """Test ChromaStorageProvider initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        provider = ChromaStorageProvider(
            persist_directory="/tmp/test_chroma",
            collection_name="test_memories"
        )
        
        assert provider.persist_directory == Path("/tmp/test_chroma")
        assert provider.collection_name == "test_memories"
        assert provider.embedding_dimension is None
        assert provider._client is None
        assert provider._collection is None
    
    def test_init_with_embedding_dimension(self):
        """Test initialization with embedding dimension."""
        provider = ChromaStorageProvider(
            persist_directory="/tmp/test_chroma",
            collection_name="test_memories",
            embedding_dimension=768
        )
        
        assert provider.embedding_dimension == 768
    
    def test_init_path_as_string(self):
        """Test initialization with string path."""
        provider = ChromaStorageProvider(
            persist_directory="/tmp/test_chroma"
        )
        
        assert provider.persist_directory == Path("/tmp/test_chroma")
        assert provider.collection_name == "memories"  # default


class TestChromaStorageProviderLifecycle:
    """Test ChromaDB provider lifecycle methods."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def provider(self, temp_dir):
        """Create ChromaStorageProvider instance."""
        return ChromaStorageProvider(
            persist_directory=temp_dir,
            collection_name="test_memories"
        )
    
    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, provider, temp_dir):
        """Test that initialize creates the persist directory."""
        # Directory shouldn't exist initially
        test_path = Path(temp_dir) / "subdir"
        provider.persist_directory = test_path
        
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            
            mock_client_class.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Collection not found")
            mock_client.create_collection.return_value = mock_collection
            
            await provider.initialize()
            
            assert test_path.exists()
            assert test_path.is_dir()
    
    @pytest.mark.asyncio
    async def test_initialize_creates_new_collection(self, provider):
        """Test initialization with new collection."""
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            
            mock_client_class.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Collection not found")
            mock_client.create_collection.return_value = mock_collection
            
            await provider.initialize()
            
            assert provider._client is mock_client
            assert provider._collection is mock_collection
            mock_client.create_collection.assert_called_once_with(
                name="test_memories",
                metadata={"description": "Memory MCP Server storage"}
            )
    
    @pytest.mark.asyncio
    async def test_initialize_retrieves_existing_collection(self, provider):
        """Test initialization with existing collection."""
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            
            mock_client_class.return_value = mock_client
            mock_client.get_collection.return_value = mock_collection
            
            await provider.initialize()
            
            assert provider._client is mock_client
            assert provider._collection is mock_collection
            mock_client.get_collection.assert_called_once_with(name="test_memories")
            mock_client.create_collection.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_error_handling(self, provider):
        """Test error handling during initialization."""
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client_class.side_effect = Exception("ChromaDB error")
            
            with pytest.raises(StorageError) as exc_info:
                await provider.initialize()
            
            assert "Failed to initialize ChromaDB" in str(exc_info.value)
            assert exc_info.value.provider == "chromadb"
    
    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Test close method."""
        provider._client = Mock()
        provider._collection = Mock()
        
        await provider.close()
        
        assert provider._client is None
        assert provider._collection is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self, provider):
        """Test async context manager functionality."""
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            
            mock_client_class.return_value = mock_client
            mock_client.get_collection.return_value = mock_collection
            
            async with provider as p:
                assert p is provider
                assert provider._client is mock_client
                assert provider._collection is mock_collection
            
            # After context exit, should be closed
            assert provider._client is None
            assert provider._collection is None


class TestChromaStorageProviderMemoryOperations:
    """Test memory storage and retrieval operations."""
    
    @pytest.fixture
    def provider(self):
        """Create initialized provider mock."""
        provider = ChromaStorageProvider("/tmp/test")
        provider._client = Mock()
        provider._collection = Mock()
        return provider
    
    @pytest.fixture
    def sample_memory(self):
        """Create sample memory for testing."""
        return Memory(
            id="test-memory-123",
            content="Test memory content",
            context="test_context",
            metadata={"source": "test", "confidence": 0.95},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
    
    @pytest.mark.asyncio
    async def test_store_memory_success(self, provider, sample_memory):
        """Test successful memory storage."""
        provider._collection.add = Mock()
        
        result = await provider.store_memory(sample_memory)
        
        assert result == sample_memory.id
        provider._collection.add.assert_called_once()
        
        # Check the call arguments
        call_args = provider._collection.add.call_args
        assert call_args[1]["ids"] == [sample_memory.id]
        assert call_args[1]["embeddings"] == [sample_memory.embedding]
        assert call_args[1]["documents"] == [sample_memory.content]
        
        metadata = call_args[1]["metadatas"][0]
        assert metadata["context"] == "test_context"
        assert metadata["source"] == "test"
        assert metadata["confidence"] == 0.95
        assert "timestamp" in metadata
    
    @pytest.mark.asyncio
    async def test_store_memory_without_embedding(self, provider, sample_memory):
        """Test storing memory without embedding fails."""
        sample_memory.embedding = None
        
        with pytest.raises(StorageError) as exc_info:
            await provider.store_memory(sample_memory)
        
        assert "must have embedding to store" in str(exc_info.value)
        assert exc_info.value.provider == "chromadb"
    
    @pytest.mark.asyncio
    async def test_store_memory_not_initialized(self, sample_memory):
        """Test storing memory when not initialized."""
        provider = ChromaStorageProvider("/tmp/test")
        
        with pytest.raises(StorageError) as exc_info:
            await provider.store_memory(sample_memory)
        
        assert "Storage not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_store_memory_chromadb_error(self, provider, sample_memory):
        """Test ChromaDB error handling during storage."""
        provider._collection.add.side_effect = Exception("ChromaDB error")
        
        with pytest.raises(StorageError) as exc_info:
            await provider.store_memory(sample_memory)
        
        assert "Failed to store" in str(exc_info.value)
        assert exc_info.value.provider == "chromadb"
    
    @pytest.mark.asyncio
    async def test_store_memories_batch(self, provider):
        """Test batch memory storage."""
        memories = [
            Memory(
                id=f"test-{i}",
                content=f"Content {i}",
                context="test",
                embedding=[0.1, 0.2, 0.3],
                metadata={}
            )
            for i in range(3)
        ]
        
        provider._collection.add = Mock()
        
        result = await provider.store_memories(memories)
        
        assert result == ["test-0", "test-1", "test-2"]
        provider._collection.add.assert_called_once()
        
        call_args = provider._collection.add.call_args
        assert len(call_args[1]["ids"]) == 3
        assert len(call_args[1]["embeddings"]) == 3
        assert len(call_args[1]["documents"]) == 3
        assert len(call_args[1]["metadatas"]) == 3
    
    @pytest.mark.asyncio
    async def test_store_memories_empty_list(self, provider):
        """Test storing empty list of memories."""
        result = await provider.store_memories([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_memory_success(self, provider, sample_memory):
        """Test successful memory retrieval."""
        # Mock ChromaDB response
        mock_response = {
            "ids": [sample_memory.id],
            "documents": [sample_memory.content],
            "metadatas": [{
                "context": sample_memory.context,
                "timestamp": sample_memory.timestamp.isoformat(),
                "source": "test",
                "confidence": 0.95
            }],
            "embeddings": [sample_memory.embedding]
        }
        provider._collection.get.return_value = mock_response
        
        result = await provider.get_memory(sample_memory.id)
        
        assert result is not None
        assert result.id == sample_memory.id
        assert result.content == sample_memory.content
        assert result.context == sample_memory.context
        assert result.embedding == sample_memory.embedding
        assert result.metadata["source"] == "test"
        assert result.metadata["confidence"] == 0.95
    
    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, provider):
        """Test retrieving non-existent memory."""
        provider._collection.get.return_value = {"ids": []}
        
        result = await provider.get_memory("non-existent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_memory_chromadb_error(self, provider):
        """Test ChromaDB error handling during retrieval."""
        provider._collection.get.side_effect = Exception("ChromaDB error")
        
        with pytest.raises(StorageError) as exc_info:
            await provider.get_memory("test-id")
        
        assert "Failed to retrieve memory" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_memories_multiple(self, provider):
        """Test retrieving multiple memories."""
        # Mock the get_memory method to return different memories
        memories = [
            Memory(id="test-1", content="Content 1", context="test", embedding=[0.1]),
            None,  # Not found
            Memory(id="test-3", content="Content 3", context="test", embedding=[0.3])
        ]
        
        with patch.object(provider, 'get_memory') as mock_get:
            mock_get.side_effect = memories
            
            result = await provider.get_memories(["test-1", "test-2", "test-3"])
            
            assert len(result) == 3
            assert result[0].id == "test-1"
            assert result[1] is None
            assert result[2].id == "test-3"
    
    @pytest.mark.asyncio
    async def test_delete_memory_success(self, provider):
        """Test successful memory deletion."""
        provider._collection.get.return_value = {"ids": ["test-id"]}
        provider._collection.delete = Mock()
        
        result = await provider.delete_memory("test-id")
        
        assert result is True
        provider._collection.delete.assert_called_once_with(ids=["test-id"])
    
    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, provider):
        """Test deleting non-existent memory."""
        provider._collection.get.return_value = {"ids": []}
        
        result = await provider.delete_memory("non-existent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_count_memories(self, provider):
        """Test counting memories."""
        provider._collection.get.return_value = {"ids": ["1", "2", "3"]}
        
        result = await provider.count_memories()
        
        assert result == 3
        provider._collection.get.assert_called_once_with(where=None, include=[])
    
    @pytest.mark.asyncio
    async def test_count_memories_with_context_filter(self, provider):
        """Test counting memories with context filter."""
        provider._collection.get.return_value = {"ids": ["1", "2"]}
        
        result = await provider.count_memories(context_filter="test_context")
        
        assert result == 2
        provider._collection.get.assert_called_once_with(
            where={"context": "test_context"}, 
            include=[]
        )


class TestChromaStorageProviderSearch:
    """Test search and context-based retrieval."""
    
    @pytest.fixture
    def provider(self):
        """Create initialized provider mock."""
        provider = ChromaStorageProvider("/tmp/test")
        provider._client = Mock()
        provider._collection = Mock()
        return provider
    
    @pytest.mark.asyncio
    async def test_search_memories_success(self, provider):
        """Test successful memory search."""
        # Mock ChromaDB query response
        mock_response = {
            "ids": [["mem-1", "mem-2"]],
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[
                {"context": "ctx1", "timestamp": "2025-01-01T12:00:00+00:00"},
                {"context": "ctx2", "timestamp": "2025-01-01T13:00:00+00:00"}
            ]],
            "embeddings": [[[0.1, 0.2], [0.3, 0.4]]],
            "distances": [[0.5, 1.0]]
        }
        provider._collection.query.return_value = mock_response
        
        query_embedding = [0.1, 0.2, 0.3]
        result = await provider.search_memories(query_embedding, limit=5)
        
        assert len(result) == 2
        
        # Check first result
        first_result = result[0]
        assert first_result["memory"].id == "mem-1"
        assert first_result["memory"].content == "Doc 1"
        assert first_result["distance"] == 0.5
        assert first_result["score"] == 1.0 / (1.0 + 0.5)  # similarity calculation
        
        # Verify ChromaDB was called correctly
        provider._collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=5,
            where=None,
            include=["documents", "metadatas", "embeddings", "distances"]
        )
    
    @pytest.mark.asyncio
    async def test_search_memories_with_filters(self, provider):
        """Test search with context and metadata filters."""
        provider._collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "embeddings": [[]],
            "distances": [[]]
        }
        
        query_embedding = [0.1, 0.2, 0.3]
        context_filter = "test_context"
        metadata_filter = {"source": "test", "confidence": 0.9}
        
        await provider.search_memories(
            query_embedding,
            context_filter=context_filter,
            metadata_filter=metadata_filter
        )
        
        # Check that where clause was built correctly
        call_args = provider._collection.query.call_args
        where_clause = call_args[1]["where"]
        assert where_clause["context"] == "test_context"
        assert where_clause["source"] == "test"
        assert where_clause["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_search_memories_similarity_threshold(self, provider):
        """Test search with similarity threshold filtering."""
        mock_response = {
            "ids": [["mem-1", "mem-2", "mem-3"]],
            "documents": [["Doc 1", "Doc 2", "Doc 3"]],
            "metadatas": [[
                {"context": "ctx", "timestamp": "2025-01-01T12:00:00+00:00"},
                {"context": "ctx", "timestamp": "2025-01-01T12:00:00+00:00"},
                {"context": "ctx", "timestamp": "2025-01-01T12:00:00+00:00"}
            ]],
            "embeddings": [[[0.1], [0.2], [0.3]]],
            "distances": [[0.1, 2.0, 5.0]]  # Different similarity scores
        }
        provider._collection.query.return_value = mock_response
        
        result = await provider.search_memories(
            [0.1], 
            similarity_threshold=0.3  # Should filter out lower similarities
        )
        
        # Only first result should pass threshold
        # similarity = 1/(1+0.1) = 0.909 > 0.3 ✓
        # similarity = 1/(1+2.0) = 0.333 > 0.3 ✓  
        # similarity = 1/(1+5.0) = 0.167 < 0.3 ✗
        assert len(result) == 2
        assert result[0]["memory"].id == "mem-1"
        assert result[1]["memory"].id == "mem-2"
    
    @pytest.mark.asyncio
    async def test_get_memories_by_context(self, provider):
        """Test retrieving memories by context."""
        mock_response = {
            "ids": ["mem-1", "mem-2"],
            "documents": ["Doc 1", "Doc 2"],
            "metadatas": [
                {"context": "test_ctx", "timestamp": "2025-01-01T12:00:00+00:00"},
                {"context": "test_ctx", "timestamp": "2025-01-01T11:00:00+00:00"}
            ],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        provider._collection.get.return_value = mock_response
        
        result = await provider.get_memories_by_context("test_ctx", limit=10)
        
        assert len(result) == 2
        # Should be sorted by timestamp (newest first)
        assert result[0].id == "mem-1"  # 12:00 > 11:00
        assert result[1].id == "mem-2"
        
        provider._collection.get.assert_called_once_with(
            where={"context": "test_ctx"},
            limit=10,
            include=["documents", "metadatas", "embeddings"]
        )
    
    @pytest.mark.asyncio
    async def test_get_memories_by_context_with_since_filter(self, provider):
        """Test retrieving memories by context with since filter."""
        since_time = datetime(2025, 1, 1, 11, 30, 0, tzinfo=timezone.utc)
        
        mock_response = {
            "ids": ["mem-1", "mem-2"],
            "documents": ["Doc 1", "Doc 2"],
            "metadatas": [
                {"context": "test_ctx", "timestamp": "2025-01-01T12:00:00+00:00"},  # After since
                {"context": "test_ctx", "timestamp": "2025-01-01T11:00:00+00:00"}   # Before since
            ],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        provider._collection.get.return_value = mock_response
        
        result = await provider.get_memories_by_context("test_ctx", since=since_time)
        
        # Should only return memories after since time
        assert len(result) == 1
        assert result[0].id == "mem-1"


class TestChromaStorageProviderHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        provider = ChromaStorageProvider("/tmp/test")
        provider._client = Mock()
        provider._collection = Mock()
        
        with patch.object(provider, 'count_memories', return_value=5):
            result = await provider.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        provider = ChromaStorageProvider("/tmp/test")
        
        result = await provider.health_check()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_operation_fails(self):
        """Test health check when operation fails."""
        provider = ChromaStorageProvider("/tmp/test")
        provider._client = Mock()
        provider._collection = Mock()
        
        with patch.object(provider, 'count_memories', side_effect=Exception("Error")):
            result = await provider.health_check()
            assert result is False
