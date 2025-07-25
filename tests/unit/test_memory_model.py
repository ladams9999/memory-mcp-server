from datetime import datetime, timezone
import pytest

from mcp_memory_server.models.memory import Memory


def test_memory_to_dict_and_from_dict_roundtrip():
    # Create a Memory instance
    ts = datetime.now(timezone.utc)
    mem = Memory(
        id="1234",
        content="Test content",
        context="test_ctx",
        metadata={"key": "value"},
        embedding=[0.1, 0.2, 0.3],
        timestamp=ts
    )
    # Convert to dict and back
    data = mem.to_dict()
    assert data["id"] == "1234"
    assert data["content"] == "Test content"
    assert data["context"] == "test_ctx"
    assert data["metadata"] == {"key": "value"}
    assert isinstance(data["timestamp"], str)

    mem2 = Memory.from_dict(data)
    assert mem2.id == mem.id
    assert mem2.content == mem.content
    assert mem2.context == mem.context
    assert mem2.metadata == mem.metadata
    # Compare timezone-aware timestamps directly
    assert mem2.timestamp == mem.timestamp


def test_has_embedding():
    mem = Memory(content="c", context="ctx", embedding=[1.0, 2.0])
    assert mem.has_embedding()

    mem2 = Memory(content="c", context="ctx")
    assert not mem2.has_embedding()


def test_str_and_repr_contains_content():
    mem = Memory(content="abcdef", context="ctx")
    s = str(mem)
    assert 'Memory(' in repr(mem) or 'Memory(id=' in s
    # __str__ shows first 50 chars of content
    assert 'abcdef' in s
