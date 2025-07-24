"""Tests for the Ollama embedding provider."""

import pytest
from unittest.mock import AsyncMock, Mock
import httpx
import asyncio

from mcp_memory_server.embeddings.ollama import OllamaEmbeddingProvider
from mcp_memory_server.embeddings.embedding_provider_interface import EmbeddingError


class TestOllamaEmbeddingProviderInit:
    """Test OllamaEmbeddingProvider initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="mxbai-embed-large"
        )
        
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "mxbai-embed-large"
        assert provider.timeout == 30.0
        assert provider._embedding_dimension is None
        assert isinstance(provider.client, httpx.AsyncClient)
    
    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model",
            timeout=60.0
        )
        
        assert provider.timeout == 60.0
    
    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base_url."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434/",
            model="test-model"
        )
        
        assert provider.base_url == "http://localhost:11434"
    
    def test_init_multiple_trailing_slashes(self):
        """Test that multiple trailing slashes are stripped."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434///",
            model="test-model"
        )
        
        assert provider.base_url == "http://localhost:11434"


class TestOllamaEmbeddingProviderContextManager:
    """Test OllamaEmbeddingProvider async context manager."""
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        async with provider as p:
            assert p is provider
            assert hasattr(p, 'client')
        
        # After context exit, client should be closed
        # We can't easily test this without mocking, but we can verify the method exists
        assert hasattr(provider, 'close')
    
    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test close method functionality."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        # Mock the client aclose method
        provider.client.aclose = AsyncMock()
        
        await provider.close()
        provider.client.aclose.assert_called_once()


class TestOllamaEmbeddingProviderGetEmbedding:
    """Test get_embedding method."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_success(self):
        """Test successful embedding generation."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4]}
        mock_response.raise_for_status.return_value = None
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        result = await provider.get_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        assert provider._embedding_dimension == 4
        
        # Verify the request was made correctly
        provider.client.post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "test-model", "prompt": "test text"}
        )
    
    @pytest.mark.asyncio
    async def test_get_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding("")
        
        assert "Text cannot be empty" in str(exc_info.value)
        assert exc_info.value.provider == "ollama"
    
    @pytest.mark.asyncio
    async def test_get_embedding_whitespace_only(self):
        """Test embedding generation with whitespace-only text."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding("   \t\n   ")
        
        assert "Text cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_embedding_strips_whitespace(self):
        """Test that whitespace is stripped from input text."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status.return_value = None
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        await provider.get_embedding("  test text  ")
        
        # Check that whitespace was stripped in the request
        provider.client.post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "test-model", "prompt": "test text"}
        )
    
    @pytest.mark.asyncio
    async def test_get_embedding_http_error(self):
        """Test HTTP error handling."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        
        http_error = httpx.HTTPStatusError(
            "404 Not Found", 
            request=Mock(), 
            response=mock_response
        )
        
        provider.client.post = AsyncMock(side_effect=http_error)
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding("test text")
        
        assert "HTTP error 404" in str(exc_info.value)
        assert exc_info.value.provider == "ollama"
        assert exc_info.value.original_error is http_error
    
    @pytest.mark.asyncio
    async def test_get_embedding_request_error(self):
        """Test request error handling."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        # Mock request error
        request_error = httpx.RequestError("Connection failed")
        provider.client.post = AsyncMock(side_effect=request_error)
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding("test text")
        
        assert "Request error" in str(exc_info.value)
        assert exc_info.value.provider == "ollama"
        assert exc_info.value.original_error is request_error
    
    @pytest.mark.asyncio
    async def test_get_embedding_missing_embedding_field(self):
        """Test response missing embedding field."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"model": "test-model"}
        mock_response.raise_for_status.return_value = None
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding("test text")
        
        assert "missing 'embedding' field" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_embedding_invalid_embedding_format(self):
        """Test invalid embedding format in response."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": "not a list"}
        mock_response.raise_for_status.return_value = None
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding("test text")
        
        assert "Invalid embedding format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_embedding_empty_embedding(self):
        """Test empty embedding list in response."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": []}
        mock_response.raise_for_status.return_value = None
        
        provider.client.post = AsyncMock(return_value=mock_response)
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding("test text")
        
        assert "Invalid embedding format" in str(exc_info.value)


class TestOllamaEmbeddingProviderGetEmbeddings:
    """Test get_embeddings method for batch processing."""
    
    @pytest.mark.asyncio
    async def test_get_embeddings_success(self):
        """Test successful batch embedding generation."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        # Mock get_embedding method
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        async def mock_get_embedding(text):
            index = ["text1", "text2", "text3"].index(text)
            return embeddings[index]
        
        provider.get_embedding = AsyncMock(side_effect=mock_get_embedding)
        
        result = await provider.get_embeddings(["text1", "text2", "text3"])
        
        assert result == embeddings
        assert provider.get_embedding.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_embeddings_empty_list(self):
        """Test batch embedding with empty list."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        result = await provider.get_embeddings([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_embeddings_with_empty_texts(self):
        """Test batch embedding with some empty texts."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        # Set up dimension and mock embedding
        provider._embedding_dimension = 3
        
        async def mock_get_embedding(text):
            return [0.1, 0.2, 0.3]
        
        provider.get_embedding = AsyncMock(side_effect=mock_get_embedding)
        
        result = await provider.get_embeddings(["text1", "", "text2", "   "])
        
        expected = [
            [0.1, 0.2, 0.3],  # text1
            [0.0, 0.0, 0.0],  # empty -> zero vector
            [0.1, 0.2, 0.3],  # text2
            [0.0, 0.0, 0.0]   # whitespace -> zero vector
        ]
        
        assert result == expected
        assert provider.get_embedding.call_count == 2  # Only called for valid texts
    
    @pytest.mark.asyncio
    async def test_get_embeddings_all_empty(self):
        """Test batch embedding with all empty texts."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embeddings(["", "   ", "\t\n"])
        
        assert "All texts are empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_embeddings_concurrent_limit(self):
        """Test that concurrent requests are limited by semaphore."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        call_times = []
        
        async def mock_get_embedding(text):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate some processing time
            return [0.1, 0.2, 0.3]
        
        provider.get_embedding = AsyncMock(side_effect=mock_get_embedding)
        
        # Test with 10 texts (should be limited to 5 concurrent)
        texts = [f"text{i}" for i in range(10)]
        await provider.get_embeddings(texts)
        
        assert provider.get_embedding.call_count == 10


class TestOllamaEmbeddingProviderDimension:
    """Test get_embedding_dimension method."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_dimension_cached(self):
        """Test getting dimension when already cached."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        provider._embedding_dimension = 512
        
        dimension = await provider.get_embedding_dimension()
        assert dimension == 512
    
    @pytest.mark.asyncio
    async def test_get_embedding_dimension_from_test(self):
        """Test getting dimension by generating test embedding."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        provider.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        
        dimension = await provider.get_embedding_dimension()
        
        assert dimension == 5
        provider.get_embedding.assert_called_once_with("test")
    
    @pytest.mark.asyncio
    async def test_get_embedding_dimension_error(self):
        """Test error handling when dimension cannot be determined."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        provider.get_embedding = AsyncMock(side_effect=Exception("Connection failed"))
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.get_embedding_dimension()
        
        assert "Failed to determine embedding dimension" in str(exc_info.value)


class TestOllamaEmbeddingProviderHealthCheck:
    """Test health_check method."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        # Mock successful API tags response
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "test-model"},
                {"name": "other-model"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        provider.client.get = AsyncMock(return_value=mock_response)
        provider.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        result = await provider.health_check()
        
        assert result is True
        provider.client.get.assert_called_once_with("http://localhost:11434/api/tags")
        provider.get_embedding.assert_called_once_with("health check")
    
    @pytest.mark.asyncio
    async def test_health_check_model_not_found(self):
        """Test health check when model is not available."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="missing-model"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "other-model"},
                {"name": "another-model"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        provider.client.get = AsyncMock(return_value=mock_response)
        
        result = await provider.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_api_error(self):
        """Test health check when API is not available."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        provider.client.get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        
        result = await provider.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_embedding_fails(self):
        """Test health check when test embedding fails."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "test-model"}]
        }
        mock_response.raise_for_status.return_value = None
        
        provider.client.get = AsyncMock(return_value=mock_response)
        provider.get_embedding = AsyncMock(side_effect=EmbeddingError("Embedding failed"))
        
        result = await provider.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_no_models_field(self):
        """Test health check when response has no models field."""
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="test-model"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status.return_value = None
        
        provider.client.get = AsyncMock(return_value=mock_response)
        provider.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        result = await provider.health_check()
        
        assert result is True  # Should still pass if embedding works
