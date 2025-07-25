import pytest

from mcp_memory_server.embeddings.ollama import OllamaEmbeddingProvider
from mcp_memory_server.embeddings.embedding_provider_interface import EmbeddingError

@pytest.mark.asyncio
async def test_health_check_success(monkeypatch):
    provider = OllamaEmbeddingProvider("http://fake", "model")

    class FakeResponse:
        def raise_for_status(self):
            pass

    async def fake_get(url):
        return FakeResponse()

    monkeypatch.setattr(provider.client, "get", fake_get)
    assert await provider.health_check() is True

@pytest.mark.asyncio
async def test_health_check_failure(monkeypatch):
    provider = OllamaEmbeddingProvider("http://fake", "model")

    async def fake_get(url):
        raise Exception("oops")

    monkeypatch.setattr(provider.client, "get", fake_get)
    assert await provider.health_check() is False

@pytest.mark.asyncio
async def test_get_embedding_dimension_success(monkeypatch):
    provider = OllamaEmbeddingProvider("http://fake", "model")

    async def fake_get_embedding(text):
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr(provider, "get_embedding", fake_get_embedding)
    dim = await provider.get_embedding_dimension()
    assert dim == 3

@pytest.mark.asyncio
async def test_get_embedding_dimension_failure():
    provider = OllamaEmbeddingProvider("http://fake", "model")
    # Default get_embedding will raise on empty text
    with pytest.raises(EmbeddingError):
        await provider.get_embedding_dimension()
