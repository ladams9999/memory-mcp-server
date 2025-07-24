# Memory MCP Server

An MCP (Model Context Protocol) server that provides long-term memory capabilities for AI agents, enabling persistent storage, retrieval, and semantic search of contextual memories across sessions.

## Features

- Store memories with contextual information
- Retrieve memories by context
- Semantic search across stored memories
- ChromaDB backend for persistent storage
- Ollama integration for local embeddings

## Setup

1. Install UV package manager
2. Install Ollama and start the service
3. Pull the embedding model: `ollama pull mxbai-embed-large`
4. Clone this repository
5. Install dependencies: `uv sync`
6. Copy `.env.example` to `.env` and configure as needed
7. Run the server: `uv run mcp-memory-server`

## Configuration

See `.env.example` for configuration options.

## Status

🚧 **This project is currently under development** 🚧

This is an MVP implementation focusing on ChromaDB + Ollama integration.
