"""Unit and integration tests for the conversation export endpoint.

Tests cover:
- ConversationExporter class (JSON, text, PDF output)
- GET /api/conversation/{session_id}/export endpoint
- 404 for unknown sessions
- 422 for unsupported formats
- 503 when IT Support agents are unavailable
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.memory.base import Message, Session, SessionMetadata
from app.agents.export import ConversationExporter


# =============================================================================
# Helpers
# =============================================================================


def _make_session(session_id: str = "test-session-001") -> Session:
    """Build a minimal Session with two messages for testing."""
    session = Session(
        id=session_id,
        metadata=SessionMetadata(user_id="user-1", agent_type="it_helpdesk"),
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 1, 12, 5, 0),
    )
    session.add_message("user", "Hello, I need help with my laptop.")
    session.add_message("assistant", "Sure! What seems to be the problem?")
    return session


# =============================================================================
# ConversationExporter unit tests
# =============================================================================


class TestConversationExporterToJson:
    def test_returns_valid_json(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        output = exporter.to_json(session)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_contains_session_id(self) -> None:
        exporter = ConversationExporter()
        session = _make_session("abc-123")
        data = json.loads(exporter.to_json(session))
        assert data["session_id"] == "abc-123"

    def test_contains_messages(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        data = json.loads(exporter.to_json(session))
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_message_fields(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        data = json.loads(exporter.to_json(session))
        msg = data["messages"][0]
        assert "role" in msg
        assert "content" in msg
        assert "timestamp" in msg

    def test_contains_agent_type(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        data = json.loads(exporter.to_json(session))
        assert data["agent_type"] == "it_helpdesk"

    def test_empty_messages(self) -> None:
        exporter = ConversationExporter()
        session = Session(id="empty-session")
        data = json.loads(exporter.to_json(session))
        assert data["messages"] == []


class TestConversationExporterToText:
    def test_returns_string(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        output = exporter.to_text(session)
        assert isinstance(output, str)

    def test_contains_session_id(self) -> None:
        exporter = ConversationExporter()
        session = _make_session("xyz-999")
        output = exporter.to_text(session)
        assert "xyz-999" in output

    def test_contains_roles_uppercased(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        output = exporter.to_text(session)
        assert "[USER]" in output
        assert "[ASSISTANT]" in output

    def test_contains_message_content(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        output = exporter.to_text(session)
        assert "Hello, I need help with my laptop." in output
        assert "Sure! What seems to be the problem?" in output

    def test_empty_session_no_crash(self) -> None:
        exporter = ConversationExporter()
        session = Session(id="empty")
        output = exporter.to_text(session)
        assert "empty" in output


class TestConversationExporterToPdf:
    def test_returns_bytes(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        result = exporter.to_pdf(session)
        assert isinstance(result, bytes)

    def test_pdf_magic_bytes(self) -> None:
        exporter = ConversationExporter()
        session = _make_session()
        result = exporter.to_pdf(session)
        assert result[:4] == b"%PDF", "Output does not start with PDF magic bytes"

    def test_pdf_with_long_message(self) -> None:
        """Verify multi-cell wrapping does not crash for long content."""
        exporter = ConversationExporter()
        session = Session(id="long-session")
        long_text = "This is a very long message. " * 50
        session.add_message("user", long_text)
        result = exporter.to_pdf(session)
        assert result[:4] == b"%PDF"

    def test_pdf_empty_session(self) -> None:
        exporter = ConversationExporter()
        session = Session(id="empty-pdf-session")
        result = exporter.to_pdf(session)
        assert result[:4] == b"%PDF"

    def test_import_error_when_fpdf_missing(self) -> None:
        """Simulate environment without fpdf2 installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "fpdf":
                raise ImportError("No module named 'fpdf'")
            return real_import(name, *args, **kwargs)

        exporter = ConversationExporter()
        session = _make_session()
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="fpdf2 is required"):
                exporter.to_pdf(session)


# =============================================================================
# Export endpoint integration tests
# =============================================================================


@pytest.fixture()
def mock_session() -> Session:
    """A pre-built session used across endpoint tests."""
    return _make_session("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


@pytest.fixture()
def client_with_session(mock_session: Session) -> TestClient:
    """TestClient with conversation_manager patched to return mock_session."""
    from app.server import app

    mock_manager = MagicMock()
    mock_manager.session_store.get_session.side_effect = (
        lambda sid: mock_session if sid == mock_session.id else None
    )

    with (
        patch("app.server.it_support_loaded", True),
        patch("app.server.conversation_manager", mock_manager),
    ):
        yield TestClient(app)


class TestExportEndpoint:
    def test_export_json_status_200(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=json")
        assert r.status_code == 200

    def test_export_json_content_type(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=json")
        assert r.headers["content-type"].startswith("application/json")

    def test_export_json_body(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=json")
        data = r.json()
        assert data["session_id"] == mock_session.id
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_export_json_default_format(self, client_with_session: TestClient, mock_session: Session) -> None:
        """Omitting ?format defaults to JSON."""
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")

    def test_export_text_status_200(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=text")
        assert r.status_code == 200

    def test_export_text_content_type(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=text")
        assert "text/plain" in r.headers["content-type"]

    def test_export_text_body(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=text")
        assert mock_session.id in r.text
        assert "[USER]" in r.text

    def test_export_pdf_status_200(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=pdf")
        assert r.status_code == 200

    def test_export_pdf_content_type(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=pdf")
        assert r.headers["content-type"] == "application/pdf"

    def test_export_pdf_magic_bytes(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=pdf")
        assert r.content[:4] == b"%PDF", "Response body does not start with PDF magic bytes"

    def test_export_pdf_content_disposition(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=pdf")
        disposition = r.headers.get("content-disposition", "")
        assert mock_session.id in disposition
        assert "attachment" in disposition

    def test_export_nonexistent_session_404(self, client_with_session: TestClient) -> None:
        r = client_with_session.get("/api/conversation/00000000-0000-0000-0000-000000000000/export?export_format=json")
        assert r.status_code == 404

    def test_export_invalid_format_422(self, client_with_session: TestClient, mock_session: Session) -> None:
        r = client_with_session.get(f"/api/conversation/{mock_session.id}/export?export_format=xml")
        assert r.status_code == 422

    def test_export_agents_unavailable_503(self, mock_session: Session) -> None:
        from app.server import app

        with (
            patch("app.server.it_support_loaded", False),
            patch("app.server.conversation_manager", None),
        ):
            client = TestClient(app)
            r = client.get(f"/api/conversation/{mock_session.id}/export?export_format=json")
        assert r.status_code == 503
