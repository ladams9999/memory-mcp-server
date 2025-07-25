"""
FastMCP application instance shared across modules.
"""
from fastmcp import FastMCP

# Create FastMCP app instance for the Memory MCP Server
app = FastMCP("MCP Memory Server")
