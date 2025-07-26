#!/usr/bin/env python3
"""
Store additional project state information to MCP Memory Server.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mcp_memory_server.services.memory_service import MemoryService
from mcp_memory_server.storage.chroma import ChromaStorageProvider
from mcp_memory_server.embeddings.ollama import OllamaEmbeddingProvider
from mcp_memory_server.config.settings import get_settings


async def store_project_state():
    """Store current project state information."""
    print("🏗️ Storing project state information...")
    
    # Initialize components
    settings = get_settings()
    storage_provider = ChromaStorageProvider(
        persist_directory=settings.get_chroma_path(),
        collection_name=settings.chroma_collection_name,
    )
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url, 
        model=settings.ollama_model
    )
    service = MemoryService(
        embedding_provider=embedding_provider,
        storage_provider=storage_provider,
        batch_size=settings.max_memories_per_request,
        similarity_threshold=settings.similarity_threshold,
    )
    
    await service.initialize()
    context = "mcp_memory_server"
    
    # Current project state
    project_state = [
        {
            "content": "Project Status: MVP Implementation - 85% Complete. Core functionality implemented and server operational.",
            "metadata": {"type": "status", "date": "2025-07-25", "completion": "85%"}
        },
        {
            "content": "Architecture: CLI-based runtime using FastMCP framework on port 8139, ChromaDB storage at ./data/chroma_db, Ollama embeddings with mxbai-embed-large model",
            "metadata": {"type": "architecture", "runtime": "CLI", "port": "8139"}
        },
        {
            "content": "Implementation Phase 1: Project Setup & Foundation - COMPLETED. All core components implemented including models, storage, embeddings, and configuration.",
            "metadata": {"type": "phase", "number": "1", "status": "completed"}
        },
        {
            "content": "Implementation Phase 2: MCP Integration & Tools - COMPLETED. FastMCP tools store_memories, retrieve_memories, and search_memories implemented and tested.",
            "metadata": {"type": "phase", "number": "2", "status": "completed"}
        },
        {
            "content": "Implementation Phase 3: Testing & Validation - IN PROGRESS. Unit tests complete, integration tests complete, manual testing verified working.",
            "metadata": {"type": "phase", "number": "3", "status": "in_progress"}
        },
        {
            "content": "Implementation Phase 4: Documentation & Final Packaging - PARTIALLY COMPLETE. README complete, tool documentation and troubleshooting guide pending.",
            "metadata": {"type": "phase", "number": "4", "status": "partial"}
        },
        {
            "content": "GitHub Issues: MVP Milestone (Issue #1) near completion. Epics #5 and #6 closed, Epic #7 and #8 open. Future milestones planned for Redis, PostgreSQL, OpenAI, Docker, monitoring.",
            "metadata": {"type": "github", "milestone": "MVP", "epics_closed": 2, "epics_open": 2}
        },
        {
            "content": "Success Criteria: CLI starts and connects to Ollama ✅, Can store memories ✅, Can retrieve memories ✅, Can perform semantic search ✅, Memories persist ✅, Error handling ✅",
            "metadata": {"type": "success_criteria", "all_met": True}
        },
        {
            "content": "Recent Fix: Added memory_tools import to main.py to properly register MCP tools with FastMCP application. Server restarted and functionality verified.",
            "metadata": {"type": "fix", "date": "2025-07-25", "issue": "missing_imports"}
        },
        {
            "content": "Test Results: All core memory functions tested successfully. Stored 14 PRD features, retrieved all memories, semantic search working with high relevance scores (0.86-0.90).",
            "metadata": {"type": "test_results", "date": "2025-07-25", "success": True}
        }
    ]
    
    # Store project state
    stored_memories = await service.store_memories(project_state, context)
    print(f"✅ Stored {len(stored_memories)} project state memories")
    
    # Get updated stats
    stats = await service.get_memory_stats(context)
    print(f"📊 Total memories in '{context}': {stats['total_memories']}")
    print(f"📊 Embedding coverage: {stats['embedding_coverage']:.1%}")
    
    print("\n🎯 Project state successfully recorded in memory server!")


if __name__ == "__main__":
    asyncio.run(store_project_state())
