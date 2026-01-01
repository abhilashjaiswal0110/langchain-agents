"""Unit tests for the agent memory module.

Tests cover:
- Checkpointer factory and backends
- Semantic memory storage and retrieval
- Conversation summarization
"""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


# =============================================================================
# Checkpointer Tests
# =============================================================================

class TestCheckpointerConfig:
    """Tests for CheckpointerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from app.agents.memory.checkpointers import CheckpointerConfig, CheckpointerBackend

        config = CheckpointerConfig()
        assert config.backend == CheckpointerBackend.MEMORY
        assert config.connection_string is None
        assert config.pool_size == 5
        assert config.max_overflow == 10

    def test_from_env_memory_backend(self, monkeypatch):
        """Test loading config from environment with memory backend."""
        from app.agents.memory.checkpointers import CheckpointerConfig, CheckpointerBackend

        monkeypatch.setenv("MEMORY_BACKEND", "memory")
        config = CheckpointerConfig.from_env()
        assert config.backend == CheckpointerBackend.MEMORY

    def test_from_env_postgres_backend(self, monkeypatch):
        """Test loading config from environment with postgres backend."""
        from app.agents.memory.checkpointers import CheckpointerConfig, CheckpointerBackend

        monkeypatch.setenv("MEMORY_BACKEND", "postgres")
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
        monkeypatch.setenv("DB_POOL_SIZE", "10")
        monkeypatch.setenv("DB_MAX_OVERFLOW", "20")

        config = CheckpointerConfig.from_env()
        assert config.backend == CheckpointerBackend.POSTGRES
        assert config.connection_string == "postgresql://user:pass@localhost:5432/db"
        assert config.pool_size == 10
        assert config.max_overflow == 20

    def test_from_env_unknown_backend_fallback(self, monkeypatch):
        """Test fallback to memory for unknown backend."""
        from app.agents.memory.checkpointers import CheckpointerConfig, CheckpointerBackend

        monkeypatch.setenv("MEMORY_BACKEND", "unknown")
        config = CheckpointerConfig.from_env()
        assert config.backend == CheckpointerBackend.MEMORY


class TestCheckpointerFactory:
    """Tests for checkpointer factory functions."""

    def test_create_memory_checkpointer(self):
        """Test creating in-memory checkpointer."""
        from app.agents.memory.checkpointers import (
            CheckpointerConfig,
            CheckpointerBackend,
            create_checkpointer,
        )

        config = CheckpointerConfig(backend=CheckpointerBackend.MEMORY)
        checkpointer = create_checkpointer(config)
        assert isinstance(checkpointer, MemorySaver)

    def test_create_checkpointer_default(self, monkeypatch):
        """Test creating checkpointer with default config from env."""
        from app.agents.memory.checkpointers import create_checkpointer

        monkeypatch.setenv("MEMORY_BACKEND", "memory")
        checkpointer = create_checkpointer()
        assert isinstance(checkpointer, MemorySaver)

    def test_get_checkpointer_singleton(self, monkeypatch):
        """Test singleton pattern for global checkpointer."""
        from app.agents.memory.checkpointers import (
            get_checkpointer,
            reset_checkpointer,
        )

        monkeypatch.setenv("MEMORY_BACKEND", "memory")
        reset_checkpointer()

        cp1 = get_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is cp2

        reset_checkpointer()

    def test_reset_checkpointer(self, monkeypatch):
        """Test resetting global checkpointer."""
        from app.agents.memory.checkpointers import (
            get_checkpointer,
            reset_checkpointer,
        )

        monkeypatch.setenv("MEMORY_BACKEND", "memory")
        reset_checkpointer()

        cp1 = get_checkpointer()
        reset_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is not cp2

        reset_checkpointer()

    def test_postgres_checkpointer_requires_connection_string(self):
        """Test that postgres backend requires DATABASE_URL."""
        from app.agents.memory.checkpointers import (
            CheckpointerConfig,
            CheckpointerBackend,
            create_checkpointer,
        )

        config = CheckpointerConfig(
            backend=CheckpointerBackend.POSTGRES,
            connection_string=None,
        )

        with pytest.raises(ValueError, match="DATABASE_URL"):
            create_checkpointer(config)


# =============================================================================
# Semantic Memory Tests
# =============================================================================

class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        from app.agents.memory.semantic_memory import MemoryEntry

        entry = MemoryEntry(
            id="test-id-123",
            content="Test content",
            memory_type="summary",
            session_id="session-123",
            user_id="user-456",
        )

        assert entry.id == "test-id-123"
        assert entry.content == "Test content"
        assert entry.memory_type == "summary"
        assert entry.session_id == "session-123"
        assert entry.user_id == "user-456"

    def test_memory_entry_optional_fields(self):
        """Test memory entry with optional fields."""
        from app.agents.memory.semantic_memory import MemoryEntry

        entry = MemoryEntry(
            id="test-id-456",
            content="Test",
            memory_type="preference",
            metadata={"key": "value"},
            agent_type="TestAgent",
        )

        assert entry.metadata == {"key": "value"}
        assert entry.agent_type == "TestAgent"
        assert entry.session_id is None
        assert entry.user_id is None

    def test_memory_entry_to_document(self):
        """Test converting memory entry to LangChain document."""
        from app.agents.memory.semantic_memory import MemoryEntry

        entry = MemoryEntry(
            id="test-id",
            content="Test content",
            user_id="user-123",
            session_id="session-456",
            agent_type="TestAgent",
            memory_type="summary",
        )

        doc = entry.to_document()
        assert doc.page_content == "Test content"
        assert doc.metadata["id"] == "test-id"
        assert doc.metadata["user_id"] == "user-123"

    def test_memory_entry_from_document(self):
        """Test creating memory entry from LangChain document."""
        from app.agents.memory.semantic_memory import MemoryEntry
        from langchain_core.documents import Document

        doc = Document(
            page_content="Test content",
            metadata={
                "id": "test-id",
                "user_id": "user-123",
                "memory_type": "fact",
            }
        )

        entry = MemoryEntry.from_document(doc)
        assert entry.id == "test-id"
        assert entry.content == "Test content"
        assert entry.user_id == "user-123"
        assert entry.memory_type == "fact"


class TestSemanticMemory:
    """Tests for SemanticMemory class.

    Note: These tests use mock embeddings to avoid actual API calls.
    """

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings for testing."""
        mock = MagicMock()
        mock.embed_query.return_value = [0.1] * 1536  # OpenAI dimension
        mock.embed_documents.return_value = [[0.1] * 1536]
        return mock

    def test_semantic_memory_initialization(self, mock_embeddings, tmp_path):
        """Test SemanticMemory initialization."""
        from app.agents.memory.semantic_memory import SemanticMemory, reset_semantic_memory

        reset_semantic_memory()

        memory = SemanticMemory(
            embeddings=mock_embeddings,
            persist_directory=str(tmp_path / "semantic_memory"),
        )
        # Vector store is lazy-loaded, so check initialization state
        assert memory._vector_store is None
        assert memory._initialized is False

        reset_semantic_memory()

    def test_search_empty_memory(self, mock_embeddings, tmp_path):
        """Test searching when memory is empty."""
        from app.agents.memory.semantic_memory import SemanticMemory, reset_semantic_memory

        reset_semantic_memory()

        memory = SemanticMemory(
            embeddings=mock_embeddings,
            persist_directory=str(tmp_path / "semantic_memory"),
        )
        results = memory.search("test query")
        assert results == []

        reset_semantic_memory()

    def test_memory_persistence_path(self, mock_embeddings, tmp_path):
        """Test that memory uses configured persistence path."""
        from app.agents.memory.semantic_memory import SemanticMemory, reset_semantic_memory

        custom_path = str(tmp_path / "custom_memory")
        reset_semantic_memory()

        memory = SemanticMemory(
            embeddings=mock_embeddings,
            persist_directory=custom_path,
        )
        assert memory.persist_directory == custom_path

        reset_semantic_memory()

    def test_get_user_context_empty(self, mock_embeddings, tmp_path):
        """Test getting user context when no memories exist."""
        from app.agents.memory.semantic_memory import SemanticMemory, reset_semantic_memory

        reset_semantic_memory()

        memory = SemanticMemory(
            embeddings=mock_embeddings,
            persist_directory=str(tmp_path / "semantic_memory"),
        )
        context = memory.get_user_context("user-123")
        assert context == ""

        reset_semantic_memory()

    def test_get_agent_context_empty(self, mock_embeddings, tmp_path):
        """Test getting agent context when no memories exist."""
        from app.agents.memory.semantic_memory import SemanticMemory, reset_semantic_memory

        reset_semantic_memory()

        memory = SemanticMemory(
            embeddings=mock_embeddings,
            persist_directory=str(tmp_path / "semantic_memory"),
        )
        # get_agent_context requires a query parameter
        context = memory.get_agent_context("ITHelpdesk", "password reset help")
        assert context == ""

        reset_semantic_memory()


class TestSemanticMemorySingleton:
    """Tests for semantic memory singleton pattern."""

    def test_get_semantic_memory_singleton(self, monkeypatch):
        """Test singleton pattern for global semantic memory."""
        from app.agents.memory.semantic_memory import (
            get_semantic_memory,
            reset_semantic_memory,
        )

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        reset_semantic_memory()

        with patch("app.agents.memory.semantic_memory.SemanticMemory") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            sm1 = get_semantic_memory()
            sm2 = get_semantic_memory()

            # Should only create one instance
            assert mock_class.call_count == 1

        reset_semantic_memory()


# =============================================================================
# Conversation Summarizer Tests
# =============================================================================

class TestConversationSummarizer:
    """Tests for ConversationSummarizer class."""

    @pytest.fixture
    def summarizer(self, monkeypatch):
        """Create a summarizer instance for testing."""
        from app.agents.memory.summarizer import ConversationSummarizer, reset_summarizer

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        reset_summarizer()

        summarizer = ConversationSummarizer(summary_threshold=5)
        yield summarizer
        reset_summarizer()

    def test_should_summarize_below_threshold(self, summarizer):
        """Test that summarization is not triggered below threshold."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        assert not summarizer.should_summarize(messages)

    def test_should_summarize_above_threshold(self, summarizer):
        """Test that summarization is triggered above threshold."""
        messages = [HumanMessage(content=f"Message {i}") for i in range(10)]
        assert summarizer.should_summarize(messages)

    def test_should_summarize_by_token_estimate(self, summarizer):
        """Test that summarization is triggered by token estimate."""
        # Create messages with lots of content
        long_content = "x" * 20000  # ~5000 tokens
        messages = [HumanMessage(content=long_content)]
        assert summarizer.should_summarize(messages)

    def test_format_messages_for_summary(self, summarizer):
        """Test formatting messages for summarization."""
        messages = [
            HumanMessage(content="Hello, I need help"),
            AIMessage(content="Sure, how can I assist?"),
            HumanMessage(content="I have a password issue"),
        ]

        formatted = summarizer.format_messages_for_summary(messages)

        assert "User: Hello, I need help" in formatted
        assert "Assistant: Sure, how can I assist?" in formatted
        assert "User: I have a password issue" in formatted

    def test_format_messages_skips_system(self, summarizer):
        """Test that system messages are skipped in formatting."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello"),
        ]

        formatted = summarizer.format_messages_for_summary(messages)

        assert "You are a helpful assistant" not in formatted
        assert "User: Hello" in formatted

    def test_format_messages_truncates_long_content(self, summarizer):
        """Test that long messages are truncated."""
        long_content = "x" * 1000
        messages = [HumanMessage(content=long_content)]

        formatted = summarizer.format_messages_for_summary(messages)

        assert "..." in formatted
        assert len(formatted) < len(long_content) + 50

    def test_fallback_summary_single_message(self, summarizer):
        """Test fallback summary with single message."""
        messages = [HumanMessage(content="I need help with my laptop")]

        summary = summarizer._create_fallback_summary(messages)

        assert "I need help with my laptop" in summary

    def test_fallback_summary_multiple_messages(self, summarizer):
        """Test fallback summary with multiple messages."""
        messages = [
            HumanMessage(content="First topic"),
            HumanMessage(content="Second topic"),
            HumanMessage(content="Third topic"),
        ]

        summary = summarizer._create_fallback_summary(messages)

        assert "First topic" in summary
        assert "Third topic" in summary

    def test_summarize_and_compress_below_threshold(self, summarizer):
        """Test compression when below threshold."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]

        compressed, summary = summarizer.summarize_and_compress(messages)

        assert compressed == messages
        assert summary is None

    def test_summarize_empty_messages(self, summarizer):
        """Test summarizing empty message list."""
        summary = summarizer.summarize([])
        assert summary == ""


class TestSummarizerSingleton:
    """Tests for summarizer singleton pattern."""

    def test_get_summarizer_singleton(self, monkeypatch):
        """Test singleton pattern for global summarizer."""
        from app.agents.memory.summarizer import get_summarizer, reset_summarizer

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        reset_summarizer()

        s1 = get_summarizer()
        s2 = get_summarizer()
        assert s1 is s2

        reset_summarizer()

    def test_reset_summarizer(self, monkeypatch):
        """Test resetting global summarizer."""
        from app.agents.memory.summarizer import get_summarizer, reset_summarizer

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        reset_summarizer()

        s1 = get_summarizer()
        reset_summarizer()
        s2 = get_summarizer()
        assert s1 is not s2

        reset_summarizer()


# =============================================================================
# Integration Tests
# =============================================================================

class TestMemoryModuleIntegration:
    """Integration tests for the memory module."""

    def test_module_imports(self):
        """Test that all module exports are available."""
        from app.agents.memory import (
            CheckpointerBackend,
            CheckpointerConfig,
            create_checkpointer,
            get_checkpointer,
            reset_checkpointer,
            MemoryEntry,
            SemanticMemory,
            get_semantic_memory,
            reset_semantic_memory,
            ConversationSummarizer,
            get_summarizer,
            reset_summarizer,
        )

        # Just verify imports work
        assert CheckpointerBackend.MEMORY is not None
        assert CheckpointerConfig is not None
        assert MemoryEntry is not None

    def test_base_agent_memory_config(self, monkeypatch):
        """Test BaseAgent memory configuration."""
        from app.agents.base.agent_base import AgentConfig

        config = AgentConfig(
            memory_backend="postgres",
            semantic_memory_enabled=True,
            conversation_summarization=True,
        )

        assert config.memory_backend == "postgres"
        assert config.semantic_memory_enabled is True
        assert config.conversation_summarization is True

    def test_base_agent_memory_backend_default(self):
        """Test BaseAgent default memory backend."""
        from app.agents.base.agent_base import AgentConfig

        config = AgentConfig()
        assert config.memory_backend == "auto"
        assert config.semantic_memory_enabled is False
        assert config.conversation_summarization is False
