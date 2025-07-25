"""Memory data model for the Memory MCP Server."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7


class Memory(BaseModel):
    """
    A memory object that can be stored and retrieved by context.

    Each memory has a unique ID, content, context, optional metadata,
    optional embedding vector, and timestamp information.
    """

    id: str = Field(
        default_factory=lambda: str(uuid7()),
        description="Unique identifier for the memory using UUID7 (time-ordered)",
    )

    content: str = Field(
        ...,
        description="The actual memory content/text to be stored",
        min_length=1,
        max_length=10000,  # Reasonable limit for memory content
    )

    context: str = Field(
        ...,
        description="Context identifier for grouping related memories",
        min_length=1,
        max_length=255,
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the memory",
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding of the memory content for semantic search",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the memory was created",
    )

    # Pydantic V2 configuration
    model_config = ConfigDict(
        # Example for documentation
        json_schema_extra={
            "example": {
                "id": "01234567-89ab-cdef-0123-456789abcdef",
                "content": "User prefers dark mode in their IDE",
                "context": "user_preferences",
                "metadata": {
                    "source": "conversation",
                    "confidence": 0.95,
                    "tags": ["ui", "preferences"],
                },
                "embedding": None,  # Will be populated by embedding provider
                "timestamp": "2025-07-23T23:30:00Z",
            }
        }
    )

    def __str__(self) -> str:
        """String representation of the memory."""
        return f"Memory(id={self.id[:8]}..., context={self.context}, content={self.content[:50]}...)"

    def __repr__(self) -> str:
        """Developer representation of the memory."""
        return (
            f"Memory("
            f"id='{self.id}', "
            f"context='{self.context}', "
            f"content='{self.content[:50]}...', "
            f"has_embedding={self.embedding is not None}, "
            f"timestamp='{self.timestamp.isoformat()}'"
            f")"
        )

    def has_embedding(self) -> bool:
        """Check if this memory has an embedding vector."""
        return self.embedding is not None and len(self.embedding) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage/serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "context": self.context,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create Memory instance from dictionary."""
        # Handle timestamp parsing
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            )

        return cls(**data)
