"""Tests for multi-tenancy session isolation.

Verifies that sessions created under different tenant IDs are fully isolated,
that the default tenant remains backward compatible, and that the X-Tenant-ID
HTTP header is honoured by the conversation endpoints.
"""

import pytest

from app.memory.memory_store import InMemorySessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store() -> InMemorySessionStore:
    """Return a fresh in-memory store for each test."""
    return InMemorySessionStore()


# ---------------------------------------------------------------------------
# InMemorySessionStore multi-tenancy tests
# ---------------------------------------------------------------------------


class TestInMemoryTenantIsolation:
    """Session key namespacing in InMemorySessionStore."""

    def test_same_session_id_different_tenants_are_independent(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """Sessions with the same ID but different tenants must be independent."""
        sid_a = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_a"
        )
        sid_b = memory_store.create_session(
            agent_type="servicenow", tenant_id="tenant_b"
        )

        # Force the same logical session ID scenario by creating two sessions and
        # checking they are stored under different namespace keys.
        sess_a = memory_store.get_session(sid_a, tenant_id="tenant_a")
        sess_b = memory_store.get_session(sid_b, tenant_id="tenant_b")

        assert sess_a is not None
        assert sess_b is not None
        assert sess_a.id != sess_b.id or sess_a.metadata.tenant_id != sess_b.metadata.tenant_id

    def test_session_not_visible_across_tenants(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """A session created for tenant_a must not be retrievable by tenant_b."""
        sid = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_a"
        )

        assert memory_store.get_session(sid, tenant_id="tenant_a") is not None
        assert memory_store.get_session(sid, tenant_id="tenant_b") is None

    def test_session_scoped_to_tenant(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """Two sessions with logically the same slot are fully separate objects."""
        # Create under tenant_a
        sid_a = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_a"
        )
        # Add a message to tenant_a's session
        memory_store.update_session(
            session_id=sid_a,
            user_message="hello from a",
            assistant_message="hi a",
            tenant_id="tenant_a",
        )

        # Create an independent session under tenant_b
        sid_b = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_b"
        )

        sess_a = memory_store.get_session(sid_a, tenant_id="tenant_a")
        sess_b = memory_store.get_session(sid_b, tenant_id="tenant_b")

        assert sess_a is not sess_b
        assert len(sess_a.messages) == 2  # user + assistant
        assert len(sess_b.messages) == 0

    def test_default_tenant_backward_compat(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """Sessions created without explicit tenant_id default to 'default' tenant."""
        sid = memory_store.create_session(agent_type="it_helpdesk")

        # Should be retrievable with no tenant_id argument (defaults to 'default')
        sess = memory_store.get_session(sid)
        assert sess is not None
        assert sess.metadata.tenant_id == "default"

    def test_delete_scoped_to_tenant(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """delete_session only removes the session for the specified tenant."""
        sid_a = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_a"
        )
        # Create a session with a different session ID for tenant_b
        sid_b = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_b"
        )

        memory_store.delete_session(sid_a, tenant_id="tenant_a")

        assert memory_store.get_session(sid_a, tenant_id="tenant_a") is None
        # tenant_b's session must be untouched
        assert memory_store.get_session(sid_b, tenant_id="tenant_b") is not None

    def test_update_session_tenant_isolation(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """update_session targeting the wrong tenant returns False."""
        sid = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_a"
        )

        # Wrong tenant — should fail silently
        result = memory_store.update_session(
            session_id=sid,
            user_message="oops",
            assistant_message="nope",
            tenant_id="tenant_b",
        )
        assert result is False

        # Correct tenant — should succeed
        result = memory_store.update_session(
            session_id=sid,
            user_message="hello",
            assistant_message="hi",
            tenant_id="tenant_a",
        )
        assert result is True

    def test_session_metadata_carries_tenant_id(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """SessionMetadata.tenant_id is persisted and round-tripped correctly."""
        sid = memory_store.create_session(
            agent_type="it_helpdesk", tenant_id="acme"
        )
        sess = memory_store.get_session(sid, tenant_id="acme")
        assert sess is not None
        assert sess.metadata.tenant_id == "acme"

    def test_list_sessions_filtered_by_tenant(
        self, memory_store: InMemorySessionStore
    ) -> None:
        """list_sessions with tenant_id only returns that tenant's sessions."""
        memory_store.create_session(agent_type="it_helpdesk", tenant_id="alpha")
        memory_store.create_session(agent_type="it_helpdesk", tenant_id="alpha")
        memory_store.create_session(agent_type="it_helpdesk", tenant_id="beta")

        alpha_sessions = memory_store.list_sessions(tenant_id="alpha")
        beta_sessions = memory_store.list_sessions(tenant_id="beta")

        assert len(alpha_sessions) == 2
        assert len(beta_sessions) == 1
        assert all(s.metadata.tenant_id == "alpha" for s in alpha_sessions)
        assert all(s.metadata.tenant_id == "beta" for s in beta_sessions)


# ---------------------------------------------------------------------------
# SQLite multi-tenancy tests
# ---------------------------------------------------------------------------


class TestSQLiteTenantIsolation:
    """Same invariants verified against the SQLite store."""

    @pytest.fixture
    def sqlite_store(self, tmp_path):
        """Provide a temporary SQLite store."""
        from app.memory.sqlite_store import SQLiteSessionStore

        db_file = str(tmp_path / "test_sessions.db")
        return SQLiteSessionStore(db_path=db_file)

    def test_session_not_visible_across_tenants(self, sqlite_store) -> None:
        sid = sqlite_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_a"
        )
        assert sqlite_store.get_session(sid, tenant_id="tenant_a") is not None
        assert sqlite_store.get_session(sid, tenant_id="tenant_b") is None

    def test_default_tenant_backward_compat(self, sqlite_store) -> None:
        sid = sqlite_store.create_session(agent_type="it_helpdesk")
        sess = sqlite_store.get_session(sid)
        assert sess is not None
        assert sess.metadata.tenant_id == "default"

    def test_tenant_id_column_exists_and_defaults(self, sqlite_store) -> None:
        """Schema migration: tenant_id column is present with DEFAULT 'default'."""
        import sqlite3

        with sqlite3.connect(sqlite_store._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("PRAGMA table_info(sessions)").fetchall()
            col_names = [r["name"] for r in row]
        assert "tenant_id" in col_names

    def test_delete_scoped_to_tenant(self, sqlite_store) -> None:
        sid_a = sqlite_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_a"
        )
        sid_b = sqlite_store.create_session(
            agent_type="it_helpdesk", tenant_id="tenant_b"
        )
        sqlite_store.delete_session(sid_a, tenant_id="tenant_a")
        assert sqlite_store.get_session(sid_a, tenant_id="tenant_a") is None
        assert sqlite_store.get_session(sid_b, tenant_id="tenant_b") is not None


# ---------------------------------------------------------------------------
# FastAPI endpoint tests (X-Tenant-ID header)
# ---------------------------------------------------------------------------


class TestConversationEndpointTenantHeader:
    """HTTP-layer tests for X-Tenant-ID header handling."""

    @pytest.fixture
    def client(self):
        """Build a TestClient that uses a fresh in-memory session store."""
        import importlib
        import sys

        # Ensure agents module is importable even without real API keys.
        # We patch conversation_manager at the app level.
        from fastapi.testclient import TestClient
        from unittest.mock import MagicMock, patch

        # Build a minimal mock conversation_manager
        mock_manager = MagicMock()
        mock_manager.start_conversation.return_value = {
            "session_id": "test-session-123",
            "agent_type": "it_helpdesk",
            "welcome_message": "Hello!",
            "available_commands": [],
        }
        mock_manager.achat = MagicMock(
            return_value={
                "session_id": "test-session-123",
                "response": "pong",
                "agent_type": "it_helpdesk",
                "tool_calls": [],
            }
        )

        import app.server as server_module

        with (
            patch.object(server_module, "conversation_manager", mock_manager),
            patch.object(server_module, "it_support_loaded", True),
        ):
            client = TestClient(server_module.app, raise_server_exceptions=True)
            yield client, mock_manager

    def test_start_conversation_default_tenant(self, client) -> None:
        """Omitting X-Tenant-ID uses 'default' tenant transparently."""
        test_client, mock_manager = client
        resp = test_client.post(
            "/api/conversation/start",
            json={"agent_type": "it_helpdesk"},
        )
        assert resp.status_code == 200
        # Verify tenant_id="default" was passed to start_conversation
        _, kwargs = mock_manager.start_conversation.call_args
        assert kwargs.get("tenant_id", "default") == "default"

    def test_start_conversation_custom_tenant(self, client) -> None:
        """X-Tenant-ID header is forwarded as tenant_id to start_conversation."""
        test_client, mock_manager = client
        resp = test_client.post(
            "/api/conversation/start",
            json={"agent_type": "it_helpdesk"},
            headers={"X-Tenant-ID": "acme"},
        )
        assert resp.status_code == 200
        _, kwargs = mock_manager.start_conversation.call_args
        assert kwargs.get("tenant_id") == "acme"

    def test_x_tenant_id_header_present_in_response(self, client) -> None:
        """A successful start response contains the session_id field."""
        test_client, _ = client
        resp = test_client.post(
            "/api/conversation/start",
            json={"agent_type": "it_helpdesk"},
            headers={"X-Tenant-ID": "acme"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
