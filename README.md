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
4. Install dependencies: `uv sync --dev`
5. Activate the virtual environment: `uv shell`
6. Install Ollama and start the service: `ollama serve`
7. Pull the embedding model: `ollama pull mxbai-embed-large`
8. Run the server: `uv run mcp-memory-server`

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

# Activate the virtual environment
uv shell

# Run the server
uv run mcp-memory-server

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_memory_service.py

# Run tests with coverage
uv run pytest --cov=src/mcp_memory_server --cov-report=html --cov-report=term-missing
```

## Agent Integration: VS Code & Warp

To use the MCP Memory Server as a memory backend for your agent, you need to configure your agent or development environment to connect to the server’s MCP endpoint.

### VS Code Settings

If your agent runs in VS Code and supports MCP memory, add the following to your settings (or your agent’s config):

```jsonc
// .vscode/settings.json
{
  "mcp.memory.endpoint": "http://localhost:8000", // Replace with your MCP server URL
  "mcp.memory.api_key": "" // (if authentication is added in future)
}
```

- Ensure the MCP server is running locally (`uv run mcp-memory-server`).
- Update the endpoint if running on a different host or port.

### Warp Terminal Settings

If you use Warp terminal and want to connect an agent or tool to the MCP server:

1. Open Warp settings.
2. Add an environment variable for the MCP endpoint:
   - Name: `MCP_MEMORY_ENDPOINT`
   - Value: `http://localhost:8000` (or your server’s address)
3. Restart Warp to apply the environment variable.

Your agent or CLI tool should read this environment variable to connect to the MCP server.

### Stdio Agent Configuration

If your agent uses stdio (standard input/output) for communication, configure it to send MCP memory requests to the server’s HTTP endpoint. Most agents allow you to specify a memory backend via environment variable or config file:

**Example (environment variable):**

```sh
export MCP_MEMORY_ENDPOINT="http://localhost:8000"
```

Or in your agent’s config file:

```ini
[memory]
backend = "mcp"
endpoint = "http://localhost:8000"
```

- Start the MCP server (`uv run mcp-memory-server`).
- Ensure your agent is configured to use the MCP endpoint for memory operations.
- If your agent supports additional options (e.g., context, API key), set those as needed.

**Note:**
- Stdio agents may require a restart after changing environment variables or config files.
- If running in a container or remote environment, update the endpoint to match your network setup.

---

**Note:**
- The MCP server must be running and accessible from your agent’s environment.
- If you use Docker or remote servers, update the endpoint accordingly.
- For more advanced agent integrations, refer to your agent’s documentation for MCP memory configuration options.
