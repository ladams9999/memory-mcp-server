"""Ollama embedding provider implementation."""

import asyncio
import logging
from typing import List

import httpx

from .embedding_provider_interface import EmbeddingProvider, EmbeddingError


logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Ollama embedding provider using httpx for async HTTP requests.

    This provider connects to a local Ollama instance to generate
    embeddings using a specified model (e.g., mxbai-embed-large).
    """

    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        """
        Initialize the Ollama embedding provider.

        Args:
            base_url: Base URL for the Ollama API (e.g., "http://localhost:11434")
            model: Name of the embedding model to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._embedding_dimension = None

        # Create httpx client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        if hasattr(self, "client"):
            await self.client.aclose()

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a single text.

        Args:
            text: The text to embed

        Returns:
            List[float]: The embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Text cannot be empty", provider="ollama")

        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {"model": self.model, "prompt": text.strip()}

            logger.debug(f"Requesting embedding for text (length: {len(text)})")

            response = await self.client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            if "embedding" not in data:
                raise EmbeddingError(
                    "Invalid response from Ollama: missing 'embedding' field",
                    provider="ollama",
                )

            embedding = data["embedding"]

            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingError(
                    "Invalid embedding format from Ollama", provider="ollama"
                )

            # Cache embedding dimension on first successful request
            if self._embedding_dimension is None:
                self._embedding_dimension = len(embedding)

            logger.debug(
                f"Successfully generated embedding (dimension: {len(embedding)})"
            )
            return embedding

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(f"Ollama embedding request failed: {error_msg}")
            raise EmbeddingError(error_msg, provider="ollama", original_error=e)

        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(f"Ollama embedding request failed: {error_msg}")
            raise EmbeddingError(error_msg, provider="ollama", original_error=e)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Ollama embedding failed: {error_msg}")
            raise EmbeddingError(error_msg, provider="ollama", original_error=e)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        # Filter out empty texts but maintain order
        valid_texts = []
        indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                indices.append(i)

        if not valid_texts:
            raise EmbeddingError("All texts are empty", provider="ollama")

        try:
            logger.debug(f"Requesting embeddings for {len(valid_texts)} texts")

            # Generate embeddings concurrently with semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests

            async def get_single_embedding(text: str) -> List[float]:
                async with semaphore:
                    return await self.get_embedding(text)

            tasks = [get_single_embedding(text) for text in valid_texts]
            embeddings = await asyncio.gather(*tasks)

            # Create result list with proper typing
            result: List[List[float]] = []

            # Handle case where we need to maintain original order
            if len(indices) == len(texts):
                # All texts were valid, just return embeddings
                result = embeddings
            else:
                # Some texts were empty, need to reconstruct with zero vectors
                if self._embedding_dimension is None and embeddings:
                    self._embedding_dimension = len(embeddings[0])

                zero_vector = [0.0] * (self._embedding_dimension or 0)

                embedding_idx = 0
                for i in range(len(texts)):
                    if i in indices:
                        result.append(embeddings[embedding_idx])
                        embedding_idx += 1
                    else:
                        result.append(zero_vector)

            logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return result

        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            error_msg = f"Batch embedding failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, provider="ollama", original_error=e)

    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Returns:
            int: The embedding dimension

        Raises:
            EmbeddingError: If dimension cannot be determined
        """
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        # Generate a test embedding to determine dimension
        try:
            test_embedding = await self.get_embedding("test")
            return len(test_embedding)
        except Exception as e:
            error_msg = f"Failed to determine embedding dimension: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, provider="ollama", original_error=e)

    async def health_check(self) -> bool:
        """
        Check if the embedding provider is healthy and available.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check if Ollama is running by hitting the API endpoint
            url = f"{self.base_url}/api/tags"
            response = await self.client.get(url)
            response.raise_for_status()

            # Check if the specific model is available
            data = response.json()
            if "models" in data:
                model_names = [model.get("name", "") for model in data["models"]]
                if self.model not in model_names:
                    logger.warning(
                        f"Model '{self.model}' not found in available models: {model_names}"
                    )
                    return False

            # Try a simple embedding to verify the model works
            await self.get_embedding("health check")
            return True

        except Exception as e:
            logger.warning(f"Ollama health check failed: {str(e)}")
            return False
