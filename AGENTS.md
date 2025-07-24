# Memory MCP Server - Agent Context

## Project Overview
The PRD and technical plan is located in memory_mcp_server_prd.md

This is an MCP (Model Context Protocol) server that provides long-term memory capabilities for AI agents, enabling persistent storage, retrieval, and semantic search of contextual memories across sessions.

## MVP Target Configuration
- **Runtime**: CLI (no Docker for MVP)
- **Storage Backend**: ChromaDB (filesystem-based)
- **Embedding Provider**: Ollama (local, self-hosted)
- **Target Model**: mxbai-embed-large or nomic-embed-text

## Key Architecture Decisions for MVP
1. **Single Storage Backend**: Focus only on ChromaDB implementation to reduce complexity
2. **Single Embedding Provider**: Focus only on Ollama integration for local deployment
3. **Core Tools Only**: Implement store_memories, retrieve_memories, search_memories
4. **Minimal Configuration**: Use sensible defaults with basic .env configuration
5. **No Authentication**: Skip auth complexity for MVP
6. **Basic Error Handling**: Essential error handling without sophisticated retry logic

## Technical Constraints for MVP
- Python 3.11+ required
- UV package manager for dependency management
- Ollama must be running locally on port 11434
- ChromaDB will store data in ./data/chroma_db directory
- FastMCP framework for MCP protocol implementation

## Success Criteria for MVP
1. CLI can start and connect to Ollama
2. Can store memories with context and metadata
3. Can retrieve all memories for a given context
4. Can perform semantic search within a context
5. Memories persist between server restarts
6. Basic error messages for common issues (Ollama down, etc.)

## Dependencies to Install
- fastmcp>=0.2.0
- pydantic>=2.5.0
- pydantic-settings>=2.1.0
- httpx>=0.25.0 (for Ollama API calls)
- numpy>=1.24.0 (for similarity calculations)
- chromadb>=0.4.0
- python-dotenv>=1.0.0
- uuid7>=0.1.0 (for unique memory IDs)

## Environment Setup Requirements
1. Ollama installed and running: `ollama serve`
2. Embedding model pulled: `ollama pull mxbai-embed-large`
3. UV package manager installed
4. Python 3.11+ available

## Key Implementation Details

### Project Structure (MVP)
```
src/
└── mcp_memory_server/
    ├── __init__.py
    ├── main.py                 # FastMCP server entry point
    ├── config/
    │   ├── __init__.py
    │   └── settings.py         # Pydantic settings with .env support
    ├── models/
    │   ├── __init__.py
    │   └── memory.py          # Memory data model with uuid7 IDs
    ├── embeddings/
    │   ├── __init__.py
    │   ├── embedding_provider_interface.py
    │   └── ollama.py          # Ollama API client implementation
    ├── storage/
    │   ├── __init__.py
    │   ├── storage_interface.py
    │   └── chroma.py          # ChromaDB implementation
    ├── services/
    │   ├── __init__.py
    │   └── memory_service.py  # Business logic layer
    └── tools/
        ├── __init__.py
        └── memory_tools.py    # MCP tool implementations
```

### Core MCP Tools to Implement
1. **store_memories(memories: List[Dict], context: str)** - Store batch of memories
2. **retrieve_memories(context: str)** - Get all memories for context
3. **search_memories(query: str, context: str, limit: int, threshold: float)** - Semantic search

### Configuration Variables (.env)
```
STORAGE_BACKEND=chroma
EMBEDDING_PROVIDER=ollama
CHROMA_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=memories
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mxbai-embed-large
MAX_MEMORIES_PER_REQUEST=100
DEFAULT_SEARCH_LIMIT=10
SIMILARITY_THRESHOLD=0.7
```

### Testing Strategy
- Unit tests for each component in isolation
- Integration tests for end-to-end workflows
- Manual testing with sample data and edge cases
- Performance testing with 100+ memories

### Development Milestones
1. **Foundation** (Steps 1-6): Core architecture and data flow
2. **MCP Integration** (Steps 7-10): FastMCP tools and server setup
3. **Testing & Validation** (Steps 11-14): Comprehensive testing
4. **Documentation** (Steps 15-16): User docs and final packaging
