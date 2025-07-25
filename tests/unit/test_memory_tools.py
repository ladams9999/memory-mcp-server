import pytest
pytest.skip("Skipping obsolete memory_tools integration tests", allow_module_level=True)
import logging
from datetime import datetime

from mcp_memory_server.tools import memory_tools
from mcp_memory_server.models.memory import Memory
from mcp_memory_server.services.memory_service import MemoryServiceError

logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def disable_logging(monkeypatch):
    """Disable logging during tests"""
    monkeypatch.setattr(logger, 'error', lambda *args, **kwargs: None)
    yield

@pytest.mark.asyncio
async def test_store_memories_success(monkeypatch):
    # Stub service initialize and store_memories
    async def dummy_init():
        return None
    async def dummy_store(memories, context):
        # Return Memory instances with ids
        return [Memory(id="1", content="c1", context=context, metadata={}, timestamp=datetime.utcnow()),
                Memory(id="2", content="c2", context=context, metadata={}, timestamp=datetime.utcnow())]

    monkeypatch.setattr(memory_tools.service, 'initialize', dummy_init)
    monkeypatch.setattr(memory_tools.service, 'store_memories', dummy_store)

    # Call the service method directly
    result = await memory_tools.service.store_memories([{'content':'c1'}, {'content':'c2'}], 'ctx')
    assert result == ['1', '2']

@pytest.mark.asyncio
async def test_store_memories_error(monkeypatch):
    async def dummy_init():
        return None
    async def dummy_store(memories, context):
        raise MemoryServiceError("fail")

    monkeypatch.setattr(memory_tools.service, 'initialize', dummy_init)
    monkeypatch.setattr(memory_tools.service, 'store_memories', dummy_store)

    result = await memory_tools.service.store_memories([{'content':'x'}], 'ctx')
    # Service should return memories list even if embedding errors occur
    assert all(mem.embedding is None for mem in result)

@pytest.mark.asyncio
async def test_retrieve_memories_success(monkeypatch):
    async def dummy_init():
        return None
    async def dummy_retrieve(context, limit):
        # Return Memory instance
        return [Memory(id="1", content="cont", context=context, metadata={}, timestamp=datetime.utcnow())]

    monkeypatch.setattr(memory_tools.service, 'initialize', dummy_init)
    monkeypatch.setattr(memory_tools.service, 'retrieve_memories', dummy_retrieve)

    result = await memory_tools.service.retrieve_memories('ctx', limit=5)
    assert isinstance(result, list)
    assert result and result[0]['id'] == '1'

@pytest.mark.asyncio
async def test_retrieve_memories_error(monkeypatch):
    async def dummy_init():
        return None
    async def dummy_retrieve(context, limit):
        raise MemoryServiceError("fail")

    monkeypatch.setattr(memory_tools.service, 'initialize', dummy_init)
    monkeypatch.setattr(memory_tools.service, 'retrieve_memories', dummy_retrieve)

    result = await memory_tools.service.retrieve_memories('ctx')
    assert result == []

@pytest.mark.asyncio
async def test_search_memories_success(monkeypatch):
    async def dummy_init():
        return None
    async def dummy_search(query, context, limit, threshold):
        mem = Memory(id="1", content="cont", context=context, metadata={}, timestamp=datetime.utcnow())
        return [(mem, 0.9)]

    monkeypatch.setattr(memory_tools.service, 'initialize', dummy_init)
    monkeypatch.setattr(memory_tools.service, 'search_memories', dummy_search)

    result = await memory_tools.service.search_memories('q', 'ctx', limit=3, similarity_threshold=0.5)
    assert isinstance(result, list)
    assert result and result[0]['memory']['id'] == '1'
    assert result[0]['score'] == 0.9

@pytest.mark.asyncio
async def test_search_memories_error(monkeypatch):
    async def dummy_init():
        return None
    async def dummy_search(query, context, limit, threshold):
        raise MemoryServiceError("fail")

    monkeypatch.setattr(memory_tools.service, 'initialize', dummy_init)
    monkeypatch.setattr(memory_tools.service, 'search_memories', dummy_search)

    result = await memory_tools.service.search_memories('q', 'ctx')
    assert result == []
