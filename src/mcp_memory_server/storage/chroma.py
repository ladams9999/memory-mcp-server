"""ChromaDB storage provider implementation."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

import chromadb
from chromadb.config import Settings

from .storage_interface import StorageProvider, StorageError
from ..models.memory import Memory


logger = logging.getLogger(__name__)


class ChromaStorageProvider(StorageProvider):
    """
    ChromaDB storage provider for the Memory MCP Server.
    
    This provider uses ChromaDB as a vector database to store
    and retrieve memories with semantic search capabilities.
    """
    
    def __init__(
        self,
        persist_directory: Union[str, Path],
        collection_name: str = "memories",
        embedding_dimension: Optional[int] = None
    ):
        """
        Initialize the ChromaDB storage provider.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            embedding_dimension: Expected dimension of embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        self._client = None
        self._collection = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure the persist directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Get or create collection
            try:
                self._collection = self._client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Retrieved existing collection '{self.collection_name}'")
            except ValueError:
                # Collection doesn't exist, create it
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Memory MCP Server storage"}
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
            
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg, provider="chromadb", original_error=e)
    
    async def store_memory(self, memory: Memory) -> str:
        """Store a single memory in ChromaDB."""
        # Use the batch store method with a single memory
        result = await self.store_memories([memory])
        return result[0]
    
    async def store_memories(self, memories: List[Memory]) -> List[str]:
        """Store multiple memories in batch."""
        if not self._collection:
            raise StorageError("Storage not initialized", provider="chromadb")
        
        if not memories:
            return []
        
        # Validate all memories have embeddings
        for memory in memories:
            if not memory.embedding:
                raise StorageError(
                    f"Memory {memory.id} must have embedding to store",
                    provider="chromadb"
                )
        
        try:
            async with self._lock:
                ids = []
                embeddings = []
                documents = []
                metadatas = []
                
                for memory in memories:
                    ids.append(memory.id)
                    embeddings.append(memory.embedding)
                    documents.append(memory.content)
                    
                    # Prepare metadata (simple types only)
                    metadata = {
                        "context": memory.context,
                        "timestamp": memory.timestamp.isoformat(),
                    }
                    
                    # Add simple metadata values only
                    for key, value in memory.metadata.items():
                        if isinstance(value, (str, int, float, bool)) and value is not None:
                            metadata[key] = value
                    
                    metadatas.append(metadata)
                
                # Batch store in ChromaDB
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                logger.debug(f"Stored {len(memories)} memories in ChromaDB")
                return ids
                
        except Exception as e:
            error_msg = f"Failed to store {len(memories)} memories: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg, provider="chromadb", original_error=e)
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        if not self._collection:
            raise StorageError("Storage not initialized", provider="chromadb")
        
        try:
            async with self._lock:
                result = self._collection.get(
                    ids=[memory_id],
                    include=["documents", "metadatas", "embeddings"]
                )
                
                if not result["ids"]:
                    return None
                
                # Extract data safely
                document = result["documents"][0] if result["documents"] else ""
                metadata_raw = result["metadatas"][0] if result["metadatas"] else {}
                embedding_raw = result["embeddings"][0] if result["embeddings"] else None
                
                # Convert to proper types
                metadata = dict(metadata_raw) if metadata_raw else {}
                context = str(metadata.pop("context", ""))
                timestamp_str = str(metadata.pop("timestamp", ""))
                
                # Convert embedding
                embedding = None
                if embedding_raw:
                    try:
                        embedding = [float(x) for x in embedding_raw]
                    except (TypeError, ValueError):
                        embedding = None
                
                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()
                except ValueError:
                    timestamp = datetime.utcnow()
                
                memory = Memory(
                    id=memory_id,
                    content=document,
                    context=context,
                    metadata=metadata,
                    embedding=embedding,
                    timestamp=timestamp
                )
                
                logger.debug(f"Retrieved memory {memory_id} from ChromaDB")
                return memory
                
        except Exception as e:
            error_msg = f"Failed to retrieve memory {memory_id}: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg, provider="chromadb", original_error=e)
    
    async def get_memories(self, memory_ids: List[str]) -> List[Optional[Memory]]:
        """Retrieve multiple memories by IDs."""
        # For simplicity, implement using single get calls
        memories = []
        for memory_id in memory_ids:
            memory = await self.get_memory(memory_id)
            memories.append(memory)
        return memories
    
    async def search_memories(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.0,
        context_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for memories using semantic similarity."""
        if not self._collection:
            raise StorageError("Storage not initialized", provider="chromadb")
        
        try:
            async with self._lock:
                # Build where clause for filtering
                where_clause = None
                if context_filter or metadata_filter:
                    where_clause = {}
                    if context_filter:
                        where_clause["context"] = context_filter
                    if metadata_filter:
                        # Only add simple type filters
                        for key, value in metadata_filter.items():
                            if isinstance(value, (str, int, float, bool)):
                                where_clause[key] = value
                
                # Perform similarity search
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause,
                    include=["documents", "metadatas", "embeddings", "distances"]
                )
                
                # Process results
                search_results = []
                if results["ids"] and results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        memory_id = results["ids"][0][i]
                        document = results["documents"][0][i] if results["documents"] else ""
                        metadata_raw = results["metadatas"][0][i] if results["metadatas"] else {}
                        embedding_raw = results["embeddings"][0][i] if results["embeddings"] else None
                        distance = results["distances"][0][i] if results["distances"] else 0.0
                        
                        # Convert distance to similarity (ChromaDB uses L2 distance)
                        similarity = 1.0 / (1.0 + float(distance))
                        
                        # Apply similarity threshold
                        if similarity < similarity_threshold:
                            continue
                        
                        # Extract data
                        metadata = dict(metadata_raw) if metadata_raw else {}
                        context = str(metadata.pop("context", ""))
                        timestamp_str = str(metadata.pop("timestamp", ""))
                        
                        # Convert embedding
                        embedding = None
                        if embedding_raw:
                            try:
                                embedding = [float(x) for x in embedding_raw]
                            except (TypeError, ValueError):
                                embedding = None
                        
                        # Parse timestamp
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()
                        except ValueError:
                            timestamp = datetime.utcnow()
                        
                        memory = Memory(
                            id=memory_id,
                            content=document,
                            context=context,
                            metadata=metadata,
                            embedding=embedding,
                            timestamp=timestamp
                        )
                        
                        search_results.append({
                            "memory": memory,
                            "score": similarity,
                            "distance": float(distance)
                        })
                
                logger.debug(f"Found {len(search_results)} memories matching search criteria")
                return search_results
                
        except Exception as e:
            error_msg = f"Failed to search memories: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg, provider="chromadb", original_error=e)
    
    async def get_memories_by_context(
        self,
        context: str,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Memory]:
        """Retrieve memories by context."""
        if not self._collection:
            raise StorageError("Storage not initialized", provider="chromadb")
        
        try:
            async with self._lock:
                # Build where clause
                where_clause = {"context": context}
                # Note: ChromaDB doesn't support complex timestamp queries easily
                # For now, we'll filter in memory after retrieval
                
                # Query ChromaDB
                result = self._collection.get(
                    where=where_clause,
                    limit=limit,
                    include=["documents", "metadatas", "embeddings"]
                )
                
                memories = []
                for i, memory_id in enumerate(result["ids"]):
                    document = result["documents"][i] if result["documents"] else ""
                    metadata_raw = result["metadatas"][i] if result["metadatas"] else {}
                    embedding_raw = result["embeddings"][i] if result["embeddings"] else None
                    
                    # Extract data
                    metadata = dict(metadata_raw) if metadata_raw else {}
                    context_value = str(metadata.pop("context", ""))
                    timestamp_str = str(metadata.pop("timestamp", ""))
                    
                    # Convert embedding
                    embedding = None
                    if embedding_raw:
                        try:
                            embedding = [float(x) for x in embedding_raw]
                        except (TypeError, ValueError):
                            embedding = None
                    
                    # Parse timestamp
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()
                    except ValueError:
                        timestamp = datetime.utcnow()
                    
                    # Apply since filter if provided
                    if since and timestamp < since:
                        continue
                    
                    memory = Memory(
                        id=memory_id,
                        content=document,
                        context=context_value,
                        metadata=metadata,
                        embedding=embedding,
                        timestamp=timestamp
                    )
                    memories.append(memory)
                
                # Sort by timestamp (newest first)
                memories.sort(key=lambda m: m.timestamp, reverse=True)
                
                logger.debug(f"Retrieved {len(memories)} memories for context '{context}'")
                return memories
                
        except Exception as e:
            error_msg = f"Failed to get memories by context '{context}': {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg, provider="chromadb", original_error=e)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self._collection:
            raise StorageError("Storage not initialized", provider="chromadb")
        
        try:
            async with self._lock:
                # Check if memory exists first
                existing = self._collection.get(ids=[memory_id])
                if not existing["ids"]:
                    return False
                
                # Delete the memory
                self._collection.delete(ids=[memory_id])
                
                logger.debug(f"Deleted memory {memory_id} from ChromaDB")
                return True
                
        except Exception as e:
            error_msg = f"Failed to delete memory {memory_id}: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg, provider="chromadb", original_error=e)
    
    async def count_memories(self, context_filter: Optional[str] = None) -> int:
        """Count the total number of memories."""
        if not self._collection:
            raise StorageError("Storage not initialized", provider="chromadb")
        
        try:
            async with self._lock:
                where_clause = None
                if context_filter:
                    where_clause = {"context": context_filter}
                
                result = self._collection.get(
                    where=where_clause,
                    include=[]  # Don't include any data, just count
                )
                
                count = len(result["ids"])
                logger.debug(f"Counted {count} memories in ChromaDB")
                return count
                
        except Exception as e:
            error_msg = f"Failed to count memories: {str(e)}"
            logger.error(error_msg)
            raise StorageError(error_msg, provider="chromadb", original_error=e)
    
    async def health_check(self) -> bool:
        """Check if ChromaDB is healthy and accessible."""
        try:
            if not self._client or not self._collection:
                return False
            
            # Try a simple operation
            await self.count_memories()
            return True
            
        except Exception as e:
            logger.warning(f"ChromaDB health check failed: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close ChromaDB client and clean up resources."""
        try:
            if self._client:
                # ChromaDB doesn't have an explicit close method
                # The client will be garbage collected
                self._client = None
                self._collection = None
                logger.debug("ChromaDB client closed")
                
        except Exception as e:
            logger.warning(f"Error closing ChromaDB client: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
