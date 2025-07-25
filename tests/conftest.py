"""Pytest configuration and fixtures for the Memory MCP Server tests."""

import sys
import os

# Add project src directory to PYTHONPATH for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import pytest
import asyncio
from datetime import datetime, timezone

@pytest.fixture
def event_loop():
    """Override pytest-asyncio event_loop fixture to use a fresh loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_memory_data():
    """Sample data for creating Memory instances in tests."""
    return {
        "id": "test-memory-123",
        "content": "This is a sample memory for testing purposes",
        "context": "test_context",
        "metadata": {
            "source": "pytest",
            "confidence": 0.95,
            "tags": ["test", "sample"],
        },
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        "timestamp": datetime(2025, 7, 23, 23, 30, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def minimal_memory_data():
    """Minimal required data for creating Memory instances."""
    return {"content": "Minimal test memory", "context": "minimal_test"}


@pytest.fixture
def large_embedding():
    """Large embedding vector for testing."""
    return [0.1] * 1536  # Typical OpenAI embedding size


@pytest.fixture
def complex_metadata():
    """Complex metadata structure for testing."""
    return {
        "source": "conversation",
        "confidence": 0.95,
        "tags": ["important", "user_preference"],
        "nested": {"level1": {"level2": "deep_value", "numbers": [1, 2, 3]}},
        "boolean_flag": True,
        "numeric_value": 42.5,
    }


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


@pytest.fixture
def ollama_base_url():
    """Base URL for Ollama testing."""
    return "http://localhost:11434"


@pytest.fixture
def ollama_model():
    """Model name for Ollama testing."""
    return "mxbai-embed-large"


@pytest.fixture
def sample_texts():
    """Sample texts for batch embedding testing."""
    return [
        "This is the first sample text",
        "Here is another piece of text",
        "And this is the third sample",
    ]


@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}


@pytest.fixture
def memory_with_embedding(sample_memory_data):
    """Memory instance with embedding for storage testing."""
    from mcp_memory_server.models.memory import Memory

    return Memory(**sample_memory_data)


@pytest.fixture
def memories_batch():
    """Batch of memory instances for bulk storage testing."""
    from mcp_memory_server.models.memory import Memory

    memories = []
    for i in range(5):
        memories.append(
            Memory(
                id=f"batch-memory-{i}",
                content=f"Batch memory content {i}",
                context=f"batch_context_{i % 2}",  # Two different contexts
                metadata={"batch_index": i, "source": "batch_test"},
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],
                timestamp=datetime(2025, 1, 1, 12, i, 0, tzinfo=timezone.utc),
            )
        )
    return memories


@pytest.fixture
def search_results_mock():
    """Mock ChromaDB search results for testing."""
    return {
        "ids": [["result-1", "result-2", "result-3"]],
        "documents": [["First result", "Second result", "Third result"]],
        "metadatas": [
            [
                {
                    "context": "search_test",
                    "timestamp": "2025-01-01T12:00:00+00:00",
                    "source": "test",
                },
                {
                    "context": "search_test",
                    "timestamp": "2025-01-01T11:00:00+00:00",
                    "source": "test",
                },
                {
                    "context": "other_test",
                    "timestamp": "2025-01-01T10:00:00+00:00",
                    "source": "test",
                },
            ]
        ],
        "embeddings": [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
        "distances": [[0.2, 0.5, 0.8]],
    }
