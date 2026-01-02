"""Unit tests for the session memory module (app.memory).

Tests cover:
- Message and Session data structures
- InMemorySessionStore implementation
- SQLiteSessionStore implementation
- ConversationMemory integration
- MemoryConfig and factory functions
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.memory.base import (
    BaseSessionStore,
    Message,
    Session,
    SessionMetadata,
)
from app.memory.memory_store import InMemorySessionStore
from app.memory.sqlite_store import SQLiteSessionStore
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
    create_session_store,
)


# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a message with required fields."""
        msg = Message(role="user", content="Hello, world!")

        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.timestamp is not None
        assert msg.metadata == {}

    def test_message_with_optional_fields(self):
        """Test message with all optional fields."""
        metadata = {"source": "web", "priority": "high"}
        msg = Message(
            role="assistant",
            content="How can I help?",
            metadata=metadata,
        )

        assert msg.metadata == metadata

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(role="user", content="Test message")
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert "timestamp" in data

    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "role": "user",
            "content": "Test content",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"key": "value"},
        }
        msg = Message.from_dict(data)

        assert msg.role == "user"
        assert msg.content == "Test content"
        assert msg.metadata == {"key": "value"}


# =============================================================================
# Session Tests
# =============================================================================


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self):
        """Test creating a session with default fields."""
        session = Session()

        assert session.id is not None
        assert session.messages == []
        assert isinstance(session.metadata, SessionMetadata)

    def test_session_with_metadata(self):
        """Test session with metadata."""
        metadata = SessionMetadata(
            user_id="user-123",
            agent_type="helpdesk",
        )
        session = Session(metadata=metadata)

        assert session.metadata.user_id == "user-123"
        assert session.metadata.agent_type == "helpdesk"

    def test_session_add_message(self):
        """Test adding messages to session."""
        session = Session()
        session.add_message("user", "Hello")

        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello"
        assert session.messages[0].role == "user"

    def test_session_add_exchange(self):
        """Test adding user/assistant exchange."""
        session = Session()
        user_msg, ai_msg = session.add_exchange(
            "Hello, I need help",
            "How can I assist you?",
        )

        assert len(session.messages) == 2
        assert user_msg.role == "user"
        assert ai_msg.role == "assistant"

    def test_session_get_history(self):
        """Test getting message history."""
        session = Session()
        for i in range(5):
            session.add_message("user", f"Message {i}")

        # Get all
        history = session.get_history()
        assert len(history) == 5

        # Get limited
        limited = session.get_history(limit=3)
        assert len(limited) == 3

    def test_session_clear_messages(self):
        """Test clearing messages."""
        session = Session()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")

        session.clear_messages()
        assert len(session.messages) == 0

    def test_session_to_dict(self):
        """Test converting session to dictionary."""
        session = Session()
        session.add_message("user", "Test")

        data = session.to_dict()

        assert "id" in data
        assert "metadata" in data
        assert len(data["messages"]) == 1

    def test_session_from_dict(self):
        """Test creating session from dictionary."""
        data = {
            "id": "session-123",
            "metadata": {
                "user_id": "user-789",
                "agent_type": "servicenow",
                "tags": [],
                "custom": {},
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Help me",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {},
                }
            ],
            "context": {"ticket_id": "INC123"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": None,
        }
        session = Session.from_dict(data)

        assert session.id == "session-123"
        assert session.metadata.agent_type == "servicenow"
        assert len(session.messages) == 1
        assert session.context["ticket_id"] == "INC123"

    def test_session_is_expired(self):
        """Test session expiration check."""
        # Not expired
        session = Session(
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert not session.is_expired

        # Expired
        expired_session = Session(
            expires_at=datetime.now() - timedelta(hours=1)
        )
        assert expired_session.is_expired


# =============================================================================
# InMemorySessionStore Tests
# =============================================================================


class TestInMemorySessionStore:
    """Tests for InMemorySessionStore implementation."""

    @pytest.fixture
    def store(self):
        """Create a fresh store for each test."""
        return InMemorySessionStore(max_sessions=100)

    def test_create_session(self, store):
        """Test creating a new session."""
        session_id = store.create_session("helpdesk", user_id="user-123")

        assert session_id is not None
        session = store.get_session(session_id)
        assert session is not None
        assert session.metadata.agent_type == "helpdesk"
        assert session.metadata.user_id == "user-123"

    def test_get_nonexistent_session(self, store):
        """Test getting a session that doesn't exist."""
        session = store.get_session("nonexistent-id")
        assert session is None

    def test_update_session(self, store):
        """Test updating a session with messages."""
        session_id = store.create_session("helpdesk")

        result = store.update_session(
            session_id,
            "Hello, I need help",
            "How can I assist you?",
        )

        assert result is True
        session = store.get_session(session_id)
        assert len(session.messages) == 2

    def test_delete_session(self, store):
        """Test deleting a session."""
        session_id = store.create_session("helpdesk")
        assert store.get_session(session_id) is not None

        result = store.delete_session(session_id)
        assert result is True
        assert store.get_session(session_id) is None

    def test_delete_nonexistent_session(self, store):
        """Test deleting a session that doesn't exist."""
        result = store.delete_session("nonexistent-id")
        assert result is False

    def test_list_sessions(self, store):
        """Test listing all sessions."""
        store.create_session("helpdesk", user_id="user-1")
        store.create_session("servicenow", user_id="user-1")
        store.create_session("helpdesk", user_id="user-2")

        # List all
        all_sessions = store.list_sessions()
        assert len(all_sessions) == 3

        # Filter by user
        user1_sessions = store.list_sessions(user_id="user-1")
        assert len(user1_sessions) == 2

        # Filter by agent type
        helpdesk_sessions = store.list_sessions(agent_type="helpdesk")
        assert len(helpdesk_sessions) == 2

    def test_get_history(self, store):
        """Test getting conversation history."""
        session_id = store.create_session("helpdesk")

        store.update_session(session_id, "Hello", "Hi there!")
        store.update_session(session_id, "Help me", "Sure!")

        # Get all history
        history = store.get_history(session_id)
        assert len(history) == 4  # 2 exchanges = 4 messages

        # Get limited history
        limited = store.get_history(session_id, limit=2)
        assert len(limited) == 2

    def test_clear_session(self, store):
        """Test clearing session messages."""
        session_id = store.create_session("helpdesk")

        store.update_session(session_id, "Hello", "Hi!")

        store.clear_session(session_id)

        session = store.get_session(session_id)
        assert len(session.messages) == 0

    def test_set_and_get_context(self, store):
        """Test setting and getting session context."""
        session_id = store.create_session("helpdesk")

        store.set_context(session_id, {"ticket_id": "INC123", "priority": "high"})

        context = store.get_context(session_id)
        assert context["ticket_id"] == "INC123"
        assert context["priority"] == "high"

    def test_max_sessions_limit(self):
        """Test that max sessions limit is enforced."""
        store = InMemorySessionStore(max_sessions=3)

        # Create 4 sessions
        ids = []
        for i in range(4):
            ids.append(store.create_session("helpdesk"))

        # First session should be cleaned up
        assert store.get_session(ids[0]) is None
        assert store.get_session(ids[3]) is not None

    def test_close(self, store):
        """Test closing the store."""
        store.create_session("helpdesk")
        store.close()
        # Store should still work after close (in-memory)
        store.create_session("helpdesk")

    def test_session_count(self, store):
        """Test session count property."""
        assert store.session_count == 0
        store.create_session("helpdesk")
        assert store.session_count == 1
        store.create_session("servicenow")
        assert store.session_count == 2


# =============================================================================
# SQLiteSessionStore Tests
# =============================================================================


class TestSQLiteSessionStore:
    """Tests for SQLiteSessionStore implementation."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a fresh SQLite store for each test."""
        db_path = str(tmp_path / "test_sessions.db")
        store = SQLiteSessionStore(db_path=db_path)
        yield store
        store.close()

    def test_create_session(self, store):
        """Test creating a new session."""
        session_id = store.create_session("helpdesk", user_id="user-123")

        assert session_id is not None
        session = store.get_session(session_id)
        assert session is not None
        assert session.metadata.agent_type == "helpdesk"

    def test_get_nonexistent_session(self, store):
        """Test getting a session that doesn't exist."""
        session = store.get_session("nonexistent-id")
        assert session is None

    def test_update_session_with_messages(self, store):
        """Test updating session with messages."""
        session_id = store.create_session("helpdesk")

        store.update_session(
            session_id,
            "Hello",
            "Hi there!",
        )
        store.update_session(
            session_id,
            "Help me",
            "Sure!",
        )

        # Reload and verify
        loaded = store.get_session(session_id)
        assert len(loaded.messages) == 4
        assert loaded.messages[0].content == "Hello"

    def test_delete_session(self, store):
        """Test deleting a session."""
        session_id = store.create_session("helpdesk")
        store.delete_session(session_id)

        assert store.get_session(session_id) is None

    def test_list_sessions_with_filters(self, store):
        """Test listing sessions with filters."""
        store.create_session("helpdesk", user_id="user-1")
        store.create_session("servicenow", user_id="user-1")
        store.create_session("helpdesk", user_id="user-2")

        # Filter by agent type
        helpdesk = store.list_sessions(agent_type="helpdesk")
        assert len(helpdesk) == 2

        # Filter by user
        user1 = store.list_sessions(user_id="user-1")
        assert len(user1) == 2

    def test_get_history(self, store):
        """Test getting message history."""
        session_id = store.create_session("helpdesk")

        for i in range(3):
            store.update_session(session_id, f"User {i}", f"AI {i}")

        history = store.get_history(session_id, limit=4)
        assert len(history) == 4

    def test_context_persistence(self, store):
        """Test that context is persisted."""
        session_id = store.create_session("helpdesk")

        store.set_context(session_id, {"ticket_id": "INC456", "status": "open"})

        context = store.get_context(session_id)
        assert context["ticket_id"] == "INC456"
        assert context["status"] == "open"

    def test_vacuum(self, store):
        """Test vacuum operation."""
        session_id = store.create_session("helpdesk")
        store.delete_session(session_id)

        # Should not raise
        store.vacuum()

    def test_get_stats(self, store):
        """Test getting database stats."""
        store.create_session("helpdesk")
        store.create_session("servicenow")

        stats = store.get_stats()
        assert stats["total_sessions"] == 2


# =============================================================================
# ConversationMemory Tests
# =============================================================================


class TestConversationMemory:
    """Tests for ConversationMemory class."""

    @pytest.fixture
    def memory(self):
        """Create a ConversationMemory instance."""
        store = InMemorySessionStore()
        return ConversationMemory(session_store=store)

    def test_create_session(self, memory):
        """Test creating a session through memory."""
        session_id = memory.create_session("helpdesk", user_id="user-123")
        assert session_id is not None

    def test_add_user_message(self, memory):
        """Test adding a user message."""
        session_id = memory.create_session("helpdesk")
        result = memory.add_user_message(session_id, "Hello, I need help")

        assert result is True
        messages = memory.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_add_assistant_message(self, memory):
        """Test adding an AI message."""
        session_id = memory.create_session("helpdesk")
        result = memory.add_assistant_message(session_id, "How can I help you?")

        assert result is True
        messages = memory.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0].role == "assistant"

    def test_add_exchange(self, memory):
        """Test adding a complete exchange."""
        session_id = memory.create_session("helpdesk")
        result = memory.add_exchange(
            session_id,
            "I forgot my password",
            "I can help you reset it.",
        )

        assert result is True
        messages = memory.get_messages(session_id)
        assert len(messages) == 2

    def test_get_langchain_messages(self, memory):
        """Test getting LangChain message format."""
        session_id = memory.create_session("helpdesk")
        memory.add_exchange(session_id, "Hello", "Hi there!")

        lc_messages = memory.get_langchain_messages(session_id)
        assert len(lc_messages) == 2

        from langchain_core.messages import HumanMessage, AIMessage
        assert isinstance(lc_messages[0], HumanMessage)
        assert isinstance(lc_messages[1], AIMessage)

    def test_get_chat_history_string(self, memory):
        """Test getting formatted history string."""
        session_id = memory.create_session("helpdesk")
        memory.add_exchange(session_id, "Hello", "Hi!")

        history_str = memory.get_chat_history_string(session_id)
        assert "Human: Hello" in history_str
        assert "AI: Hi!" in history_str

    def test_clear_session(self, memory):
        """Test clearing conversation."""
        session_id = memory.create_session("helpdesk")
        memory.add_exchange(session_id, "Hello", "Hi!")

        memory.clear_session(session_id)

        messages = memory.get_messages(session_id)
        assert len(messages) == 0

    def test_get_summary(self, memory):
        """Test getting conversation summary."""
        session_id = memory.create_session("helpdesk", user_id="test-user")
        memory.add_exchange(session_id, "Hello", "Hi!")

        summary = memory.get_summary(session_id)
        assert summary is not None
        assert summary.session_id == session_id
        assert summary.message_count == 2

    def test_list_sessions(self, memory):
        """Test listing sessions."""
        memory.create_session("helpdesk", user_id="user-1")
        memory.create_session("servicenow", user_id="user-1")

        summaries = memory.list_sessions(user_id="user-1")
        assert len(summaries) == 2


class TestConversationMemorySingleton:
    """Tests for conversation memory singleton pattern."""

    def test_get_conversation_memory(self):
        """Test getting conversation memory."""
        reset_conversation_memory()

        memory1 = get_conversation_memory()
        memory2 = get_conversation_memory()

        assert memory1 is memory2

        reset_conversation_memory()

    def test_reset_conversation_memory(self):
        """Test resetting conversation memory."""
        reset_conversation_memory()

        memory1 = get_conversation_memory()
        reset_conversation_memory()
        memory2 = get_conversation_memory()

        assert memory1 is not memory2

        reset_conversation_memory()


# =============================================================================
# MemoryConfig Tests
# =============================================================================


class TestMemoryConfig:
    """Tests for MemoryConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()

        assert config.backend == MemoryBackend.MEMORY
        assert config.session_ttl_hours == 24
        assert config.max_sessions == 10000

    def test_from_env_memory_backend(self, monkeypatch):
        """Test loading config from environment."""
        monkeypatch.setenv("MEMORY_BACKEND", "memory")

        config = MemoryConfig.from_env()
        assert config.backend == MemoryBackend.MEMORY

    def test_from_env_redis_backend(self, monkeypatch):
        """Test loading Redis config from environment."""
        monkeypatch.setenv("MEMORY_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", "redis://redis-host:6380")
        monkeypatch.setenv("SESSION_KEY_PREFIX", "app:")

        config = MemoryConfig.from_env()
        assert config.backend == MemoryBackend.REDIS
        assert config.redis_url == "redis://redis-host:6380"
        assert config.key_prefix == "app:"

    def test_from_env_sqlite_backend(self, monkeypatch):
        """Test loading SQLite config from environment."""
        monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_PATH", "/data/sessions.db")

        config = MemoryConfig.from_env()
        assert config.backend == MemoryBackend.SQLITE
        assert config.sqlite_path == "/data/sessions.db"

    def test_from_env_unknown_backend_fallback(self, monkeypatch):
        """Test fallback for unknown backend."""
        monkeypatch.setenv("MEMORY_BACKEND", "unknown")

        config = MemoryConfig.from_env()
        assert config.backend == MemoryBackend.MEMORY


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestSessionStoreFactory:
    """Tests for session store factory functions."""

    def test_get_session_store_memory(self, monkeypatch):
        """Test getting in-memory session store."""
        monkeypatch.setenv("MEMORY_BACKEND", "memory")
        reset_session_store()

        store = get_session_store()
        assert isinstance(store, InMemorySessionStore)

        reset_session_store()

    def test_get_session_store_sqlite(self, monkeypatch, tmp_path):
        """Test getting SQLite session store."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_PATH", db_path)
        reset_memory_config()
        reset_session_store()

        store = get_session_store()
        assert isinstance(store, SQLiteSessionStore)

        reset_memory_config()
        reset_session_store()
        store.close()

    def test_get_session_store_singleton(self, monkeypatch):
        """Test singleton pattern for session store."""
        monkeypatch.setenv("MEMORY_BACKEND", "memory")
        reset_session_store()

        store1 = get_session_store()
        store2 = get_session_store()
        assert store1 is store2

        reset_session_store()

    def test_create_session_store_memory(self):
        """Test creating in-memory store directly."""
        store = create_session_store(MemoryBackend.MEMORY, max_sessions=500)
        assert isinstance(store, InMemorySessionStore)

    def test_create_session_store_sqlite(self, tmp_path):
        """Test creating SQLite store directly."""
        db_path = str(tmp_path / "direct.db")
        store = create_session_store(MemoryBackend.SQLITE, db_path=db_path)
        assert isinstance(store, SQLiteSessionStore)
        store.close()


class TestCheckpointerFactory:
    """Tests for checkpointer factory functions."""

    def test_get_checkpointer_memory(self, monkeypatch):
        """Test getting memory checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        monkeypatch.setenv("MEMORY_BACKEND", "memory")

        checkpointer = get_checkpointer(CheckpointerType.MEMORY)
        assert isinstance(checkpointer, MemorySaver)

    def test_get_checkpointer_default_matches_backend(self, monkeypatch):
        """Test that default checkpointer matches session backend."""
        from langgraph.checkpoint.memory import MemorySaver

        monkeypatch.setenv("MEMORY_BACKEND", "memory")

        checkpointer = get_checkpointer()
        assert isinstance(checkpointer, MemorySaver)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryModuleIntegration:
    """Integration tests for the memory module."""

    def test_full_conversation_flow(self, tmp_path):
        """Test a complete conversation flow with persistence."""
        db_path = str(tmp_path / "integration.db")
        store = SQLiteSessionStore(db_path=db_path)

        # Create memory with store
        memory = ConversationMemory(session_store=store)

        # Create session
        session_id = memory.create_session(
            agent_type="helpdesk",
            user_id="integration-user",
        )

        # Add messages
        memory.add_exchange(
            session_id,
            "I need help with my laptop",
            "What seems to be the problem?",
        )
        memory.add_exchange(
            session_id,
            "It won't turn on",
            "Have you tried checking the power cable?",
        )

        # Set context
        memory.set_context(session_id, {"issue_type": "hardware", "device": "laptop"})

        # Verify messages
        messages = memory.get_messages(session_id)
        assert len(messages) == 4

        # Verify context
        context = memory.get_context(session_id)
        assert context["issue_type"] == "hardware"

        store.close()

    def test_module_exports(self):
        """Test that all module exports are available."""
        from app.memory import (
            BaseSessionStore,
            Session,
            SessionMetadata,
            Message,
            InMemorySessionStore,
            RedisSessionStore,
            SQLiteSessionStore,
            ConversationMemory,
            ConversationSummary,
            get_conversation_memory,
            reset_conversation_memory,
            MemoryBackend,
            MemoryConfig,
            CheckpointerType,
            get_memory_config,
            get_session_store,
            reset_session_store,
            get_checkpointer,
        )

        # Verify types are available
        assert MemoryBackend.MEMORY is not None
        assert CheckpointerType.REDIS is not None
