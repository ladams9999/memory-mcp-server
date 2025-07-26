

import logging
import os
# Shared FastMCP app instance
from mcp_memory_server.app_instance import app
from mcp_memory_server.config.settings import Settings, get_settings  # noqa: F401
from mcp_memory_server.embeddings.ollama import OllamaEmbeddingProvider  # noqa: F401

# Import memory tools to register them with the app
from mcp_memory_server.tools import memory_tools  # noqa: F401


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


### Note: app is defined at module top to avoid circular imports ###

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

    # Configure FastMCP to use the specified port
    # FastMCP reads FASTMCP_PORT environment variable for port configuration
    os.environ["FASTMCP_PORT"] = str(settings.server_port)
    
    logger.info(f"Server will start on port {settings.server_port}")

    # Health check is skipped in main to prevent blocking startup
    # TODO: Implement health check asynchronously if needed

    # Run the FastMCP app (handles async event loop internally)
    try:
        app.run(
            transport="http",
            host="localhost",
            port=settings.server_port
        )
    except Exception as e:
        logger.error(f"Failed to run FastMCP server: {e}")
        raise


if __name__ == "__main__":
    main()
