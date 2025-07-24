"""Pytest configuration and fixtures for the Memory MCP Server tests."""

import pytest
from datetime import datetime, timezone


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
            "tags": ["test", "sample"]
        },
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        "timestamp": datetime(2025, 7, 23, 23, 30, 0, tzinfo=timezone.utc)
    }


@pytest.fixture
def minimal_memory_data():
    """Minimal required data for creating Memory instances."""
    return {
        "content": "Minimal test memory",
        "context": "minimal_test"
    }


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
        "nested": {
            "level1": {
                "level2": "deep_value",
                "numbers": [1, 2, 3]
            }
        },
        "boolean_flag": True,
        "numeric_value": 42.5
    }
