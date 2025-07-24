"""Main entry point for the MCP Memory Server."""

import logging
from typing import Optional

from fastmcp import FastMCP
from pydantic_settings import BaseSettings

from .models.memory import Memory


class Settings(BaseSettings):
    """Server settings."""
    
    log_level: str = "INFO"


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Create the FastMCP app instance
app = FastMCP("MCP Memory Server")

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
    settings = Settings()
    setup_logging(settings.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting MCP Memory Server...")
    
    # FastMCP handles the async event loop internally
    app.run()


if __name__ == "__main__":
    main()
