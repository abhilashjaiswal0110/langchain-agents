"""Unit tests for configurable conversation history limit.

Verifies that MAX_HISTORY_MESSAGES controls how many messages are
retained per session, with 0 (the default) meaning unlimited.
"""

import os
import importlib

import pytest

from app.memory.base import Message, Session
from app.memory.memory_store import InMemorySessionStore


# =============================================================================
# Helpers
# =============================================================================


def _make_store(monkeypatch: pytest.MonkeyPatch, max_messages: str) -> InMemorySessionStore:
    """Reload memory_store with a specific MAX_HISTORY_MESSAGES value.

    The module-level constant is read once at import time, so we must
    reload the module after patching the environment variable.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        max_messages: String value to set for MAX_HISTORY_MESSAGES.

    Returns:
        A fresh InMemorySessionStore instance under the patched limit.
    """
    monkeypatch.setenv("MAX_HISTORY_MESSAGES", max_messages)
    import app.memory.memory_store as mem_mod
    importlib.reload(mem_mod)
    return mem_mod.InMemorySessionStore()


def _fill_session(store: InMemorySessionStore, session_id: str, n_exchanges: int) -> None:
    """Add n_exchanges user/assistant pairs to a session.

    Args:
        store: The session store to write into.
        session_id: Target session identifier.
        n_exchanges: Number of user→assistant pairs to add.
    """
    for i in range(n_exchanges):
        store.update_session(
            session_id=session_id,
            user_message=f"user msg {i}",
            assistant_message=f"assistant reply {i}",
        )


# =============================================================================
# Session dataclass tests (no store involved)
# =============================================================================


class TestSessionDataclass:
    """Tests for the Session dataclass itself."""

    def test_default_unlimited_history(self) -> None:
        """With no external limit, all messages are kept in the Session object."""
        session = Session(id="test-unlimited")
        for i in range(20):
            session.add_message("user", f"msg {i}")

        assert len(session.messages) == 20

    def test_get_history_with_limit(self) -> None:
        """Session.get_history respects the optional limit argument."""
        session = Session(id="test-get-history")
        for i in range(15):
            session.add_message("user", f"msg {i}")

        history = session.get_history(limit=5)
        assert len(history) == 5
        # Should be the most recent 5
        assert history[-1].content == "msg 14"

    def test_get_history_no_limit(self) -> None:
        """Session.get_history returns all messages when limit is None."""
        session = Session(id="test-no-limit")
        for i in range(10):
            session.add_message("user", f"msg {i}")

        history = session.get_history(limit=None)
        assert len(history) == 10


# =============================================================================
# InMemorySessionStore tests
# =============================================================================


class TestInMemorySessionStoreHistoryLimit:
    """Tests for InMemorySessionStore history trimming via env var."""

    def test_default_unlimited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With MAX_HISTORY_MESSAGES=0 (default), all messages are kept."""
        store = _make_store(monkeypatch, "0")
        session_id = store.create_session(agent_type="it_helpdesk")

        _fill_session(store, session_id, 20)  # 40 messages total

        session = store.get_session(session_id)
        assert session is not None
        assert len(session.messages) == 40

    def test_limit_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With MAX_HISTORY_MESSAGES=5, only the last 5 messages are kept."""
        store = _make_store(monkeypatch, "5")
        session_id = store.create_session(agent_type="it_helpdesk")

        _fill_session(store, session_id, 10)  # 20 messages total; expect trim to 5

        session = store.get_session(session_id)
        assert session is not None
        assert len(session.messages) == 5

    def test_limit_keeps_most_recent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The retained messages are the most recent ones, not the oldest."""
        store = _make_store(monkeypatch, "4")
        session_id = store.create_session(agent_type="it_helpdesk")

        for i in range(6):
            store.update_session(
                session_id=session_id,
                user_message=f"user {i}",
                assistant_message=f"assistant {i}",
            )

        session = store.get_session(session_id)
        assert session is not None
        assert len(session.messages) == 4
        # The last exchange (i=5) must be present
        contents = [m.content for m in session.messages]
        assert "user 5" in contents
        assert "assistant 5" in contents

    def test_limit_of_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Edge case: MAX_HISTORY_MESSAGES=1 keeps only the single most recent message."""
        store = _make_store(monkeypatch, "1")
        session_id = store.create_session(agent_type="it_helpdesk")

        store.update_session(
            session_id=session_id,
            user_message="first user",
            assistant_message="first assistant",
        )
        store.update_session(
            session_id=session_id,
            user_message="second user",
            assistant_message="second assistant",
        )

        session = store.get_session(session_id)
        assert session is not None
        assert len(session.messages) == 1
        assert session.messages[0].content == "second assistant"

    def test_no_messages_when_limit_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sanity check: limit=0 means unlimited, messages accumulate."""
        store = _make_store(monkeypatch, "0")
        session_id = store.create_session(agent_type="it_helpdesk")

        _fill_session(store, session_id, 5)

        session = store.get_session(session_id)
        assert session is not None
        assert len(session.messages) == 10  # 5 exchanges = 10 messages

    def test_exactly_at_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When message count exactly equals the limit, no trimming occurs."""
        store = _make_store(monkeypatch, "4")
        session_id = store.create_session(agent_type="it_helpdesk")

        # 2 exchanges = 4 messages — exactly at limit
        _fill_session(store, session_id, 2)

        session = store.get_session(session_id)
        assert session is not None
        assert len(session.messages) == 4


# =============================================================================
# AgentConfig max_history field tests
# =============================================================================


class TestAgentConfigMaxHistory:
    """Tests for the max_history field on AgentConfig."""

    def test_default_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AgentConfig.max_history defaults to MAX_HISTORY_MESSAGES env var."""
        monkeypatch.setenv("MAX_HISTORY_MESSAGES", "15")
        import app.agents.base.agent_base as ab_mod
        importlib.reload(ab_mod)

        config = ab_mod.AgentConfig()
        assert config.max_history == 15

    def test_default_unlimited_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AgentConfig.max_history defaults to 0 (unlimited) when env var is absent."""
        monkeypatch.delenv("MAX_HISTORY_MESSAGES", raising=False)
        import app.agents.base.agent_base as ab_mod
        importlib.reload(ab_mod)

        config = ab_mod.AgentConfig()
        assert config.max_history == 0

    def test_explicit_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit max_history value overrides the env var default."""
        monkeypatch.setenv("MAX_HISTORY_MESSAGES", "5")
        import app.agents.base.agent_base as ab_mod
        importlib.reload(ab_mod)

        config = ab_mod.AgentConfig(max_history=99)
        assert config.max_history == 99
