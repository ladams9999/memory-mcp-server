"""Main entry point for the MCP Memory Server."""

import logging
from fastmcp import FastMCP

# Application settings
# Application settings
from mcp_memory_server.config.settings import get_settings
import mcp_memory_server.tools.memory_tools  # noqa: F401  # register tool implementations
import asyncio
from mcp_memory_server.embeddings.ollama import OllamaEmbeddingProvider


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Create FastMCP app instance
app = FastMCP("MCP Memory Server")

# Register memory tools to the main app (import at top per PEP8)
# noqa: F401

# Add your tools and resources here
# Example:
# @app.tool("store_memory")
# async def store_memory(content: str, tags: Optional[list[str]] = None) -> str:
#     """Store a memory with optional tags."""
#     # Implementation here
#     return "Memory stored successfully"


@app.tool("ping")
async def ping() -> str:
    """Simple ping tool to test the server."""
    return "pong"


def main() -> None:
    """Main entry point."""
    # Load application settings
    settings = get_settings()
    setup_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting MCP Memory Server...")

    # Perform Ollama connectivity check via embedding provider
    provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url, model=settings.ollama_model
    )
    healthy = asyncio.run(provider.health_check())
    if healthy:
        logger.info("Ollama health check passed")
    else:
        logger.error("Ollama health check failed")
        raise SystemExit(1)

    # Run the FastMCP app (handles async event loop internally)
    try:
        app.run()
    except Exception as e:
        logger.error(f"Failed to run FastMCP server: {e}")
        raise


if __name__ == "__main__":
    main()
