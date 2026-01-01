"""Memory module for persistent agent state and semantic memory.

This module provides:
- Configurable checkpointers (PostgreSQL, SQLite, Memory)
- Semantic long-term memory with FAISS vector storage
- Conversation summarization for token efficiency
"""

from app.agents.memory.checkpointers import (
    CheckpointerBackend,
    CheckpointerConfig,
    create_checkpointer,
    get_async_checkpointer,
    get_checkpointer,
    reset_checkpointer,
)
from app.agents.memory.semantic_memory import (
    MemoryEntry,
    SemanticMemory,
    get_semantic_memory,
    reset_semantic_memory,
)
from app.agents.memory.summarizer import (
    ConversationSummarizer,
    get_summarizer,
    reset_summarizer,
)

__all__ = [
    # Checkpointers
    "CheckpointerBackend",
    "CheckpointerConfig",
    "create_checkpointer",
    "get_checkpointer",
    "get_async_checkpointer",
    "reset_checkpointer",
    # Semantic Memory
    "MemoryEntry",
    "SemanticMemory",
    "get_semantic_memory",
    "reset_semantic_memory",
    # Summarizer
    "ConversationSummarizer",
    "get_summarizer",
    "reset_summarizer",
]
