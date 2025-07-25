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
