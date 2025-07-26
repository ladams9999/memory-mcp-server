# Memory MCP Server - Agent Context

## Project Overview
The PRD and technical plan is located in memory_mcp_server_prd.md

This is an MCP (Model Context Protocol) server that provides long-term memory capabilities for AI agents, enabling persistent storage, retrieval, and semantic search of contextual memories across sessions.

## Project Task Management via GitHub Issues

All project tasks, milestones, and epics are maintained via GitHub issues in the repository, using the GitHub MCP server for automation and tracking. The local TODO.md is used as a reference and synchronized with GitHub issues as needed.

### Issue Hierarchy and Block Format

Each issue must include a hierarchy block at the top of its description, formatted as follows:

For Milestone issues (top-level):
```
Parent:
Project: <project_link>
Children:
- <child_issue_link>
Related to:
- <related_issue_link>
```

For Epic issues (phase-level):
```
Parent: <parent_issue_link>
Project: <project_link>
Children:
- <child_issue_link>
Depends On:
- <prerequisite_issue_link>
```

Checklist tasks for the epic should be listed below this block.

### Instructions for Maintaining Tasks

1. Create all new tasks, milestones, and epics as GitHub issues.
2. Add the hierarchy block to each issue as described above.
3. Link parent/child issues and project as appropriate.
4. Use the GitHub MCP server to automate issue creation, updates, and tracking.
5. Do not use TODO.md for new tasks; migrate any remaining items to issues.
6. Keep checklists and progress in the issue body for visibility.
7. For Epic issues, always include a link to the previous phase's issue in the 'Depends On' field to indicate dependency order.

### Instructions for Maintaining TODO.md

1. At the start of each milestone, update the top section to identify the milestone, expected outcome, and link to the milestone issue.
2. Group all tasks by their Epic issue, including a link to each Epic issue.
3. Keep task status (checked/unchecked) in sync with GitHub issues.
4. Updates should flow both ways: from GitHub issues into TODO.md, and from TODO.md into GitHub issues.
5. When a new milestone begins, archive or move the previous milestone's tasks as needed.
6. Use this file as a local reference and synchronization point for milestone progress.

This process ensures all work is visible, organized, and tracked in the GitHub project and issues.


## Agent Integration & Setup

To use the MCP Memory Server as a memory backend for your agent, configure your agent or development environment to connect to the server's MCP endpoint.

### VS Code Settings

Create or update your `mcp.json` file (location: VS Code user settings or agent config directory):

```jsonc
{
  "servers": {
    "MCP Memory Server": {
      "url": "http://localhost:8139/mcp/",
      "type": "http"
    }
  },
  "inputs": []
}
```

### Warp Settings

```jsonc
{
  "MCP Memory Server": {
    "url": "http://localhost:8139/mcp/"
  }
}
```

- Ensure the MCP server is running locally (`uv run mcp-memory-server`).
- Update the endpoint if running on a different host or port.

**Notes:**
- Agents may require a restart after changing environment variables or config files.
- If running in a container or remote environment, update the endpoint to match your network setup.

**The MCP server must be running and accessible from your agent's environment.**
- If you use Docker or remote servers, update the endpoint accordingly.
- For more advanced agent integrations, refer to your agent's documentation for MCP memory configuration options.

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

1. Clone this repository
2. Install UV package manager ([installation guide](https://uv.sh/install))
3. Copy `.env.example` to `.env` and configure environment variables
4. Install dependencies: `uv sync --dev`
5. Activate the virtual environment: `uv shell`
6. Install Ollama and start the service: `ollama serve`
7. Pull the embedding model: `ollama pull mxbai-embed-large`
8. Ensure Python 3.11+ is available
9. Run the server: `uv run mcp-memory-server`

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

### Development Commands (UV Package Manager)

```bash
# Note: If uv is not in PATH, use the full path:
# Windows: $env:USERPROFILE\.local\bin\uv.exe

# Install all dependencies including dev dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_models/test_memory.py

# Run tests with coverage
uv run pytest --cov=src/mcp_memory_server

# Run tests with verbose output
uv run pytest -v

# Add development dependencies
uv add --dev package_name

# Add production dependencies
uv add package_name
```

### Development Milestones

1. **Foundation** (Steps 1-6): Core architecture and data flow
2. **MCP Integration** (Steps 7-10): FastMCP tools and server setup
3. **Testing & Validation** (Steps 11-14): Comprehensive testing
4. **Documentation** (Steps 15-16): User docs and final packaging
