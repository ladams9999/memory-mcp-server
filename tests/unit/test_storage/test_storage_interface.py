"""Tests for the storage provider interface."""

import pytest
from abc import ABC

from mcp_memory_server.storage.storage_interface import (
    StorageProvider,
    StorageError
)


class TestStorageError:
    """Test StorageError exception class."""
    
    def test_storage_error_basic(self):
        """Test basic StorageError creation."""
        error = StorageError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.provider is None
        assert error.original_error is None
        
    def test_storage_error_with_provider(self):
        """Test StorageError with provider information."""
        error = StorageError("Test error", provider="test_provider")
        
        assert str(error) == "[test_provider] Test error"
        assert error.provider == "test_provider"
        assert error.original_error is None
        
    def test_storage_error_with_original_error(self):
        """Test StorageError with original exception."""
        original = ValueError("Original error")
        error = StorageError("Test error", original_error=original)
        
        assert str(error) == "Test error"
        assert error.provider is None
        assert error.original_error is original
        
    def test_storage_error_full(self):
        """Test StorageError with all parameters."""
        original = RuntimeError("Runtime error")
        error = StorageError(
            "Full error message", 
            provider="full_provider", 
            original_error=original
        )
        
        assert str(error) == "[full_provider] Full error message"
        assert error.provider == "full_provider"
        assert error.original_error is original


class TestStorageProvider:
    """Test StorageProvider abstract base class."""
    
    def test_storage_provider_is_abstract(self):
        """Test that StorageProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StorageProvider()
    
    def test_storage_provider_inheritance(self):
        """Test that StorageProvider is properly defined as ABC."""
        assert issubclass(StorageProvider, ABC)
        
        # Check that all required methods are abstract
        abstract_methods = StorageProvider.__abstractmethods__
        expected_methods = {
            'initialize',
            'store_memory',
            'store_memories',
            'get_memory',
            'get_memories',
            'search_memories',
            'get_memories_by_context',
            'delete_memory',
            'count_memories',
            'health_check',
            'close'
        }
        
        assert abstract_methods == expected_methods
    
    def test_concrete_implementation_requirements(self):
        """Test what methods a concrete implementation must define."""
        
        class ConcreteProvider(StorageProvider):
            async def initialize(self):
                pass
            
            async def store_memory(self, memory):
                return "test-id"
            
            async def store_memories(self, memories):
                return ["test-id"]
            
            async def get_memory(self, memory_id):
                return None
            
            async def get_memories(self, memory_ids):
                return []
            
            async def search_memories(self, query_embedding, limit=10, 
                                    similarity_threshold=0.0, context_filter=None, 
                                    metadata_filter=None):
                return []
            
            async def get_memories_by_context(self, context, limit=None, since=None):
                return []
            
            async def delete_memory(self, memory_id):
                return True
            
            async def count_memories(self, context_filter=None):
                return 0
            
            async def health_check(self):
                return True
            
            async def close(self):
                pass
        
        # Should be able to instantiate now
        provider = ConcreteProvider()
        assert isinstance(provider, StorageProvider)
    
    def test_missing_methods_fails(self):
        """Test that missing abstract methods prevents instantiation."""
        
        class IncompleteProvider(StorageProvider):
            async def initialize(self):
                pass
            # Missing other required methods
        
        with pytest.raises(TypeError):
            IncompleteProvider()
