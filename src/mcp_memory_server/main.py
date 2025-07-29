import logging
# Shared FastMCP app instance
from mcp_memory_server.app_instance import app
from mcp_memory_server.config.settings import Settings, get_settings


def setup_logging(level: str = "INFO", log_file: str = "mcp_memory_server.log") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

@app.tool("ping")
async def ping() -> str:
    """Simple ping tool to test the server."""
    return "pong"

def main() -> None:
    """Main entry point."""
    # Load application settings
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)

    logger = logging.getLogger(__name__)
    logger.info("Starting MCP Memory Server...")
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
