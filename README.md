# Memory MCP Server

An MCP (Model Context Protocol) server that provides long-term memory capabilities for AI agents, enabling persistent storage, retrieval, and semantic search of contextual memories across sessions.

## Features

- Store memories with contextual information
- Retrieve memories by context
- Semantic search across stored memories
- ChromaDB backend for persistent storage
- Ollama integration for local embeddings

## Setup

1. Clone this repository
2. Install UV package manager ([installation guide](https://uv.sh/install))
3. Copy `.env.example` to `.env` and configure environment variables
4. Install dependencies: `uv sync`.  This should also create the virtual environment
5. If Ollama is not running, start the service: `ollama serve`
6. Pull the embedding model: `ollama pull mxbai-embed-large`
7. Run the server: `uv run mcp-memory-server`

Uvicorn uses websockets in a way that has been deprecated in websockets >= 14.0.  You may see a warning to this effect when you start the server.     

## Configuration

See `.env.example` for configuration options.

## Status

🚧 **This project is currently under development** 🚧

This is an MVP implementation focusing on ChromaDB + Ollama integration.

## Development

After setting up the project and activating the virtual environment, use the UV package manager to manage dependencies, run the server, and execute tests:

```bash
# Install all dependencies including dev dependencies
uv sync --dev

# Run the server
uv run mcp-memory-server

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_memory_service.py

# Run tests with coverage
uv run pytest --cov=src/mcp_memory_server --cov-report=html --cov-report=term-missing
```

## Agent Integration

To use the MCP Memory Server as a memory backend for your agent, you need to configure your agent or development environment to connect to the server's MCP endpoint.

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

---


**The MCP server must be running and accessible from your agent's environment.**
- If you use Docker or remote servers, update the endpoint accordingly.
- For more advanced agent integrations, refer to your agent's documentation for MCP memory configuration options.
