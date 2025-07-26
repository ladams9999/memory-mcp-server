#!/usr/bin/env python3
"""
Test script to verify MCP Memory Server functionality.
This script will test the store_memories, retrieve_memories, and search_memories functions.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mcp_memory_server.services.memory_service import MemoryService
from mcp_memory_server.storage.chroma import ChromaStorageProvider
from mcp_memory_server.embeddings.ollama import OllamaEmbeddingProvider
from mcp_memory_server.config.settings import get_settings


async def test_memory_functionality():
    """Test the core memory functionality."""
    print("🧪 Testing MCP Memory Server functionality...")
    
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
    
    try:
        # Initialize the service
        await service.initialize()
        print("✅ Service initialized successfully")
        
        # Test context
        context = "mcp_memory_server"
        
        # PRD Features to store
        prd_features = [
            {
                "content": "Overview: An MCP (Model Context Protocol) server that provides long-term memory capabilities for AI agents, enabling persistent storage, retrieval, and semantic search of contextual memories across sessions.",
                "metadata": {"type": "overview", "source": "PRD"}
            },
            {
                "content": "Core Objective: Enable AI agents to store and retrieve memories with contextual information",
                "metadata": {"type": "objective", "source": "PRD"}
            },
            {
                "content": "Core Objective: Provide semantic search capabilities across stored memories",
                "metadata": {"type": "objective", "source": "PRD"}
            },
            {
                "content": "Core Objective: Support multiple persistence backends with pluggable architecture",
                "metadata": {"type": "objective", "source": "PRD"}
            },
            {
                "content": "Core Objective: Offer configurable embedding models for semantic operations",
                "metadata": {"type": "objective", "source": "PRD"}
            },
            {
                "content": "Core Objective: Maintain high performance and scalability",
                "metadata": {"type": "objective", "source": "PRD"}
            },
            {
                "content": "Primary Tool: store_memories - Store one or more memories with specific context metadata",
                "metadata": {"type": "tool", "source": "PRD"}
            },
            {
                "content": "Primary Tool: retrieve_memories - Retrieve all memories associated with a specific context",
                "metadata": {"type": "tool", "source": "PRD"}
            },
            {
                "content": "Primary Tool: search_memories - Perform semantic search across memories within a context",
                "metadata": {"type": "tool", "source": "PRD"}
            },
            {
                "content": "Supported Backend: ChromaDB - Local filesystem-based vector database",
                "metadata": {"type": "backend", "source": "PRD"}
            },
            {
                "content": "Supported Backend: Redis - In-memory database with optional persistence",
                "metadata": {"type": "backend", "source": "PRD"}
            },
            {
                "content": "Supported Backend: PostgreSQL + pgvector - Traditional RDBMS with vector extension",
                "metadata": {"type": "backend", "source": "PRD"}
            },
            {
                "content": "Supported Embedding: Local Ollama - Self-hosted embedding models",
                "metadata": {"type": "embedding", "source": "PRD"}
            },
            {
                "content": "Supported Embedding: OpenAI - Remote API-based embeddings",
                "metadata": {"type": "embedding", "source": "PRD"}
            }
        ]
        
        # Test 1: Store memories
        print(f"📝 Storing {len(prd_features)} PRD features...")
        stored_memories = await service.store_memories(prd_features, context)
        print(f"✅ Stored {len(stored_memories)} memories successfully")
        
        # Test 2: Retrieve memories
        print(f"📖 Retrieving memories for context '{context}'...")
        retrieved_memories = await service.retrieve_memories(context)
        print(f"✅ Retrieved {len(retrieved_memories)} memories")
        
        # Test 3: Search memories
        print("🔍 Testing semantic search...")
        search_queries = [
            "tool for storing memories",
            "database backends supported",
            "embedding models available"
        ]
        
        for query in search_queries:
            results = await service.search_memories(query, context, limit=3, threshold=0.5)
            print(f"  Query: '{query}' -> {len(results)} results")
            for memory, score in results[:2]:  # Show top 2 results
                print(f"    Score: {score:.3f} - {memory.content[:60]}...")
        
        # Test 4: Get stats
        stats = await service.get_memory_stats(context)
        print(f"📊 Memory Stats: {stats}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    

if __name__ == "__main__":
    asyncio.run(test_memory_functionality())
