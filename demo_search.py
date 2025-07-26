#!/usr/bin/env python3
"""
Demonstrate search functionality with stored PRD features and project state.
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


async def demonstrate_search():
    """Demonstrate search functionality with stored data."""
    print("🔍 Demonstrating MCP Memory Server search capabilities...")
    
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
    
    # Get current memory stats
    stats = await service.get_memory_stats(context)
    print(f"📊 Current state: {stats['total_memories']} memories stored")
    print(f"📊 Embedding coverage: {stats['embedding_coverage']:.1%}")
    print()
    
    # Demonstrate various search queries
    search_queries = [
        "What are the core objectives of this project?",
        "What tools are available in the MCP server?",
        "What storage backends are supported?",
        "What is the current implementation status?",
        "How do I store memories?",
        "What embedding models can be used?",
        "What phase is currently in progress?",
        "Has the server been tested successfully?"
    ]
    
    for query in search_queries:
        print(f"❓ Query: {query}")
        results = await service.search_memories(query, context, limit=3, threshold=0.7)
        
        if results:
            for i, (memory, score) in enumerate(results, 1):
                print(f"   {i}. [{score:.3f}] {memory.content}")
                if memory.metadata:
                    metadata_str = ", ".join([f"{k}: {v}" for k, v in memory.metadata.items()])
                    print(f"      Metadata: {metadata_str}")
        else:
            print("   No results found with threshold >= 0.7")
        print()
    
    print("✅ Search demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_search())
