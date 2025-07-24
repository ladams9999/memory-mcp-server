"""Tests for the embedding provider interface."""

import pytest
from abc import ABC

from mcp_memory_server.embeddings.embedding_provider_interface import (
    EmbeddingProvider,
    EmbeddingError
)


class TestEmbeddingError:
    """Test EmbeddingError exception class."""
    
    def test_embedding_error_basic(self):
        """Test basic EmbeddingError creation."""
        error = EmbeddingError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.provider is None
        assert error.original_error is None
        
    def test_embedding_error_with_provider(self):
        """Test EmbeddingError with provider information."""
        error = EmbeddingError("Test error", provider="test_provider")
        
        assert str(error) == "[test_provider] Test error"
        assert error.provider == "test_provider"
        assert error.original_error is None
        
    def test_embedding_error_with_original_error(self):
        """Test EmbeddingError with original exception."""
        original = ValueError("Original error")
        error = EmbeddingError("Test error", original_error=original)
        
        assert str(error) == "Test error"
        assert error.provider is None
        assert error.original_error is original
        
    def test_embedding_error_full(self):
        """Test EmbeddingError with all parameters."""
        original = RuntimeError("Runtime error")
        error = EmbeddingError(
            "Full error message", 
            provider="full_provider", 
            original_error=original
        )
        
        assert str(error) == "[full_provider] Full error message"
        assert error.provider == "full_provider"
        assert error.original_error is original


class TestEmbeddingProvider:
    """Test EmbeddingProvider abstract base class."""
    
    def test_embedding_provider_is_abstract(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()
    
    def test_embedding_provider_inheritance(self):
        """Test that EmbeddingProvider is properly defined as ABC."""
        assert issubclass(EmbeddingProvider, ABC)
        
        # Check that all required methods are abstract
        abstract_methods = EmbeddingProvider.__abstractmethods__
        expected_methods = {
            'get_embedding',
            'get_embeddings', 
            'get_embedding_dimension',
            'health_check'
        }
        
        assert abstract_methods == expected_methods
    
    def test_concrete_implementation_requirements(self):
        """Test what methods a concrete implementation must define."""
        
        class ConcreteProvider(EmbeddingProvider):
            async def get_embedding(self, text: str):
                return [0.1, 0.2, 0.3]
            
            async def get_embeddings(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            async def get_embedding_dimension(self):
                return 3
            
            async def health_check(self):
                return True
        
        # Should be able to instantiate now
        provider = ConcreteProvider()
        assert isinstance(provider, EmbeddingProvider)
    
    def test_missing_methods_fails(self):
        """Test that missing abstract methods prevents instantiation."""
        
        class IncompleteProvider(EmbeddingProvider):
            async def get_embedding(self, text: str):
                return [0.1, 0.2, 0.3]
            # Missing other required methods
        
        with pytest.raises(TypeError):
            IncompleteProvider()
