# Memory MCP Server - PRD & Technical Implementation Plan

## Product Requirements Document (PRD)

### Overview
An MCP (Model Context Protocol) server that provides long-term memory capabilities for AI agents, enabling persistent storage, retrieval, and semantic search of contextual memories across sessions.

### Core Objectives
- Enable AI agents to store and retrieve memories with contextual information
- Provide semantic search capabilities across stored memories
- Support multiple persistence backends with pluggable architecture
- Offer configurable embedding models for semantic operations
- Maintain high performance and scalability

### Key Features

#### Primary Tools
1. **store_memories**: Store one or more memories with specific context metadata
2. **retrieve_memories**: Retrieve all memories associated with a specific context
3. **search_memories**: Perform semantic search across memories within a context

#### Supported Persistence Backends
- **ChromaDB**: Local filesystem-based vector database
- **Redis**: In-memory database with optional persistence
- **PostgreSQL + pgvector**: Traditional RDBMS with vector extension

#### Supported Embedding Models
- **Local Ollama**: Self-hosted embedding models
- **OpenAI**: Remote API-based embeddings

### User Stories
- As an AI agent, I want to store conversation memories so I can reference them in future interactions
- As an AI agent, I want to retrieve memories by context so I can maintain conversation continuity
- As an AI agent, I want to search memories semantically so I can find relevant information efficiently
- As a developer, I want to configure different persistence backends so I can optimize for my deployment environment
- As a developer, I want to use different embedding models so I can balance cost, performance, and privacy

## Technical Implementation Plan

### Project Structure
```
mcp-memory-server/
├── pyproject.toml
├── .env.example
├── README.md
├── src/
│   └── mcp_memory_server/
│       ├── __init__.py
│       ├── main.py                 # FastMCP server definition
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py         # Environment-based configuration
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── storage_interface.py # Storage backend interface
│       │   ├── chroma.py          # ChromaDB implementation
│       │   ├── redis.py           # Redis implementation
│       │   └── postgres.py       # PostgreSQL implementation
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── embedding_provider_interface.py # Embedding model interface
│       │   ├── ollama.py          # Ollama implementation
│       │   └── openai.py          # OpenAI implementation
│       ├── models/
│       │   ├── __init__.py
│       │   └── memory.py          # Memory data models
│       ├── services/
│       │   ├── __init__.py
│       │   └── memory_service.py  # Core business logic
│       └── tools/
│           ├── __init__.py
│           └── memory_tools.py    # MCP tool implementations
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_storage/
│   │   ├── test_embeddings/
│   │   ├── test_services/
│   │   └── test_tools/
│   └── integration/
│       └── test_end_to_end.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

### Core Interfaces

#### Storage Interface (storage/storage_interface.py)
```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..models.memory import Memory

class StorageInterface(ABC):
    @abstractmethod
    async def store_memories(self, memories: List[Memory], context: str) -> List[str]:
        """Store memories with the given context and return their IDs"""
        pass
    
    @abstractmethod
    async def retrieve_memories(self, context: str) -> List[Memory]:
        """Retrieve all memories for a given context"""
        pass
    
    @abstractmethod
    async def search_memories(
        self, 
        query_embedding: List[float], 
        context: str, 
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Memory]:
        """Search memories by semantic similarity"""
        pass
    
    @abstractmethod
    async def delete_memories(self, memory_ids: List[str]) -> bool:
        """Delete memories by IDs"""
        pass
```

#### Embedding Interface (embeddings/embedding_provider_interface.py)
```python
from abc import ABC, abstractmethod
from typing import List

class EmbeddingProviderInterface(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings"""
        pass
```

### Data Models

#### Memory Model
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid7

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid7.uuid7()))
    content: str = Field(..., description="The memory content")
    context: str = Field(..., description="Context identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### Configuration

#### Settings Model
```python
from pydantic_settings import BaseSettings
from typing import Literal, Optional

class Settings(BaseSettings):
    # Storage Configuration
    storage_backend: Literal["chroma", "redis", "postgres"] = "chroma"
    
    # ChromaDB Settings
    chroma_path: str = "./chroma_db"
    chroma_collection_name: str = "memories"
    
    # Redis Settings
    redis_url: str = "redis://localhost:6379"
    redis_key_prefix: str = "memory:"
    
    # PostgreSQL Settings
    postgres_url: str = "postgresql://user:pass@localhost/memories"
    postgres_table_name: str = "memories"
    
    # Embedding Configuration
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    
    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mxbai-embed-large"
    
    # OpenAI Settings
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"
    
    # General Settings
    max_memories_per_request: int = 100
    default_search_limit: int = 10
    similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### MCP Tools Implementation

#### Memory Tools
```python
from fastmcp import FastMCP
from typing import List, Dict, Any
from .services.memory_service import MemoryService
from .models.memory import Memory

class MemoryTools:
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
    
    @fastmcp.tool()
    async def store_memories(
        self,
        memories: List[Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        Store one or more memories with a specific context.
        
        Args:
            memories: List of memory objects with 'content' and optional 'metadata'
            context: Context identifier for grouping memories
        
        Returns:
            Dict with stored memory IDs and success status
        """
        try:
            memory_objects = [
                Memory(content=mem["content"], context=context, metadata=mem.get("metadata", {}))
                for mem in memories
            ]
            
            memory_ids = await self.memory_service.store_memories(memory_objects)
            
            return {
                "success": True,
                "memory_ids": memory_ids,
                "count": len(memory_ids)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @fastmcp.tool()
    async def retrieve_memories(self, context: str) -> Dict[str, Any]:
        """
        Retrieve all memories associated with a specific context.
        
        Args:
            context: Context identifier
        
        Returns:
            Dict with memories and metadata
        """
        try:
            memories = await self.memory_service.retrieve_memories(context)
            
            return {
                "success": True,
                "memories": [mem.dict() for mem in memories],
                "count": len(memories)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "memories": [],
                "count": 0
            }
    
    @fastmcp.tool()
    async def search_memories(
        self,
        query: str,
        context: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform semantic search across memories within a specific context.
        
        Args:
            query: Search query text
            context: Context identifier
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
        
        Returns:
            Dict with matching memories and similarity scores
        """
        try:
            results = await self.memory_service.search_memories(
                query, context, limit, similarity_threshold
            )
            
            return {
                "success": True,
                "results": [
                    {
                        "memory": result["memory"].dict(),
                        "similarity": result["similarity"]
                    }
                    for result in results
                ],
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
```

### Service Layer

#### Memory Service
```python
from typing import List, Dict, Any
from ..storage.storage_interface import StorageInterface
from ..embeddings.embedding_provider_interface import EmbeddingProviderInterface
from ..models.memory import Memory

class MemoryService:
    def __init__(self, storage: StorageInterface, embeddings: EmbeddingProviderInterface):
        self.storage = storage
        self.embeddings = embeddings
    
    async def store_memories(self, memories: List[Memory]) -> List[str]:
        """Store memories with embeddings"""
        # Generate embeddings for all memories
        texts = [memory.content for memory in memories]
        embeddings = await self.embeddings.embed_texts(texts)
        
        # Attach embeddings to memories
        for memory, embedding in zip(memories, embeddings):
            memory.embedding = embedding
        
        # Store in backend
        return await self.storage.store_memories(memories, context)
    
    async def retrieve_memories(self, context: str) -> List[Memory]:
        """Retrieve all memories for a context"""
        return await self.storage.retrieve_memories(context)
    
    async def search_memories(
        self,
        query: str,
        context: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search memories semantically"""
        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)
        
        # Search in storage
        memories = await self.storage.search_memories(
            query_embedding, context, limit, similarity_threshold
        )
        
        # Calculate similarity scores (if not provided by storage)
        results = []
        for memory in memories:
            similarity = self._calculate_similarity(query_embedding, memory.embedding)
            results.append({
                "memory": memory,
                "similarity": similarity
            })
        
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
```

### Development Setup

#### pyproject.toml
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mcp-memory-server"
version = "0.1.0"
description = "MCP server for long-term memory storage and retrieval"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
    "fastmcp>=0.2.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "httpx>=0.25.0",
    "numpy>=1.24.0",
    "chromadb>=0.4.0",
    "redis>=5.0.0",
    "psycopg[binary]>=3.1.0",
    "pgvector>=0.2.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "uuid7>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0",
]

[project.scripts]
mcp-memory-server = "mcp_memory_server.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src/mcp_memory_server --cov-report=html --cov-report=term-missing"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

#### .env.example
```env
# Storage Backend Configuration
STORAGE_BACKEND=chroma  # chroma, redis, postgres

# ChromaDB Settings
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION_NAME=memories

# Redis Settings
REDIS_URL=redis://localhost:6379
REDIS_KEY_PREFIX=memory:

# PostgreSQL Settings
POSTGRES_URL=postgresql://user:password@localhost:5432/memories
POSTGRES_TABLE_NAME=memories

# Embedding Provider Configuration
EMBEDDING_PROVIDER=ollama  # ollama, openai

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mxbai-embed-large

# OpenAI Settings
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=text-embedding-3-small

# General Settings
MAX_MEMORIES_PER_REQUEST=100
DEFAULT_SEARCH_LIMIT=10
SIMILARITY_THRESHOLD=0.7
```

### Testing Strategy

#### Test Categories
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test scalability and performance

#### Test Coverage Requirements
- Minimum 90% code coverage
- All public APIs must be tested
- Error conditions and edge cases covered
- Async functionality properly tested

#### Key Test Scenarios
- Memory storage and retrieval
- Semantic search functionality
- Configuration validation
- Database backend switching
- Embedding provider switching
- Error handling and recovery
- Concurrent operations

### Deployment Requirements

#### CLI Runtime Requirements

**System Requirements:**
- Python 3.11 or higher
- UV package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Minimum 2GB RAM (4GB recommended for large memory sets)
- 1GB available disk space for ChromaDB storage

**Local Dependencies (for CLI usage):**
- **Ollama** (if using local embeddings): Install via `curl -fsSL https://ollama.com/install.sh | sh`
- **Redis** (optional, if using Redis backend): Install via package manager or Redis official installer
- **PostgreSQL with pgvector** (optional, if using PostgreSQL backend):
  - PostgreSQL 12+ with pgvector extension
  - Install pgvector: `CREATE EXTENSION vector;`

**CLI Installation Steps:**
```bash
# Clone repository
git clone <repository-url>
cd mcp-memory-server

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your preferred settings

# For Ollama usage, pull embedding model
ollama pull mxbai-embed-large

# Run the server
uv run mcp-memory-server
```

**Minimal CLI Configuration (.env):**
```env
STORAGE_BACKEND=chroma
EMBEDDING_PROVIDER=ollama
CHROMA_PATH=./data/chroma_db
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
```

#### Docker Runtime Requirements

**System Requirements:**
- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+ (for multi-service setups)
- Minimum 4GB RAM allocated to Docker
- 2GB available disk space for images and volumes

**Docker Installation Steps:**
```bash
# Clone repository
git clone <repository-url>
cd mcp-memory-server

# Build and run with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t mcp-memory-server .
docker run -d --name memory-server \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  mcp-memory-server
```

**Docker Compose Configuration (docker-compose.yml):**
```yaml
version: '3.8'

services:
  memory-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - STORAGE_BACKEND=postgres
      - EMBEDDING_PROVIDER=openai
      - POSTGRES_URL=postgresql://memory_user:password@postgres:5432/memories
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_DB=memories
      - POSTGRES_USER=memory_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_KEEP_ALIVE=24h

volumes:
  postgres_data:
  redis_data:
  ollama_data:
```

**Docker Multi-stage Dockerfile:**
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --frozen --no-cache

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ src/

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "mcp_memory_server.main"]
```

#### Deployment Considerations

**Docker Advantages:**
- Consistent environment across development, staging, and production
- Easy orchestration of multiple services (Redis, PostgreSQL, Ollama)
- Automatic dependency management and isolation
- Built-in health checks and restart policies
- Simplified scaling and load balancing
- Container registries for easy distribution

**CLI Advantages:**
- Direct integration with existing development workflows
- No containerization overhead
- Easier debugging and development
- Native performance
- Simpler configuration for single-machine deployments
- Direct access to system resources

**Performance Optimization (Both Deployments):**
- Connection pooling for databases
- Batch processing for embeddings
- Caching strategies for frequently accessed memories
- Async operations throughout the stack
- Resource limits and monitoring

**Monitoring and Observability:**
- Structured logging with context
- Metrics for memory operations
- Health check endpoints (`/health`, `/metrics`)
- Performance monitoring hooks
- Error tracking and alerting

**Security Considerations:**
- Environment variable validation
- API key management (CLI: .env file, Docker: secrets)
- Network isolation (Docker networks)
- Resource limits and quotas
- Input validation and sanitization

This implementation plan provides a solid foundation for building a scalable, testable, and maintainable MCP memory server with flexible deployment options for both CLI and Docker environments.