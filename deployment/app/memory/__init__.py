"""Memory and persistence module for conversation management.

Provides persistent storage backends for session and conversation data:
- In-memory storage (development/testing)
- Redis storage (production, distributed)
- SQLite storage (single-instance persistence)

Usage:
    from app.memory import (
        # Session stores
        SessionStore, RedisSessionStore, SQLiteSessionStore,
        get_session_store, MemoryConfig,

        # Conversation memory
        ConversationMemory, get_conversation_memory,

        # LangGraph checkpointers
        get_checkpointer, CheckpointerType,
    )

    # Get configured session store
    store = get_session_store()
    session_id = store.create_session("helpdesk", user_id="user123")

    # Use with LangGraph
    checkpointer = get_checkpointer(CheckpointerType.REDIS)
    graph = workflow.compile(checkpointer=checkpointer)
"""

from app.memory.base import (
    BaseSessionStore,
    Message,
    Session,
    SessionMetadata,
)
from app.memory.memory_store import (
    InMemorySessionStore,
)
from app.memory.redis_store import (
    RedisSessionStore,
)
from app.memory.sqlite_store import (
    SQLiteSessionStore,
)
from app.memory.conversation_memory import (
    ConversationMemory,
    ConversationSummary,
    get_conversation_memory,
    reset_conversation_memory,
)
from app.memory.config import (
    CheckpointerType,
    MemoryBackend,
    MemoryConfig,
    get_checkpointer,
    get_memory_config,
    get_session_store,
    reset_memory_config,
    reset_session_store,
)

__all__ = [
    # Base types
    "BaseSessionStore",
    "Session",
    "SessionMetadata",
    "Message",
    # Implementations
    "InMemorySessionStore",
    "RedisSessionStore",
    "SQLiteSessionStore",
    # Conversation memory
    "ConversationMemory",
    "ConversationSummary",
    "get_conversation_memory",
    "reset_conversation_memory",
    # Configuration
    "MemoryBackend",
    "MemoryConfig",
    "CheckpointerType",
    "get_memory_config",
    "get_session_store",
    "reset_memory_config",
    "reset_session_store",
    "get_checkpointer",
]
