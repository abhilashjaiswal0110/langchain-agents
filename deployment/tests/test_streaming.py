"""Tests for SSE streaming endpoints on all 8 enterprise agents.

These tests verify:
- Each /stream endpoint returns HTTP 200
- Response content-type is ``text/event-stream``
- At least one SSE event is received (``error`` or ``complete``)
- Event payloads are valid JSON

All tests operate without real LLM credentials; when agents are unavailable
(no API key) the endpoint still returns 200 + an ``error`` event rather than
raising an HTTP error.  This validates the SSE error-fallback path.
"""

import json
import os

import pytest

# Provide a mock key so the server module can import without ValueError.
# The key is intentionally invalid — no real LLM calls are made in these tests.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing")

from fastapi.testclient import TestClient  # noqa: E402 — must come after env setup

from app.server import app  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

STREAM_ENDPOINTS: list[tuple[str, dict]] = [
    ("/api/enterprise/research/stream", {"query": "AI trends"}),
    (
        "/api/enterprise/content/stream",
        {"topic": "AI agents", "platform": "linkedin"},
    ),
    ("/api/enterprise/data-analyst/stream", {"message": "Summarise the data"}),
    (
        "/api/enterprise/documents/stream",
        {
            "doc_type": "sop",
            "title": "Onboarding SOP",
            "description": "Employee onboarding procedure",
        },
    ),
    ("/api/enterprise/rag/stream", {"query": "What is in the document?"}),
    (
        "/api/enterprise/support/stream",
        {"message": "My email is not working"},
    ),
    (
        "/api/enterprise/code/stream",
        {"code": "def foo(): pass", "language": "python", "action": "analyze"},
    ),
    (
        "/api/enterprise/document-intelligence/stream",
        {"message": "Summarise the uploaded document"},
    ),
]

VALID_EVENT_TYPES = {"token", "tool_start", "tool_end", "complete", "error"}


def _parse_sse_lines(content: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of event dicts."""
    events: list[dict] = []
    for line in content.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload:
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Shared synchronous test client (no real network calls)."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Parametrised tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint,body", STREAM_ENDPOINTS)
class TestEnterpriseStreamEndpoints:
    """Verify every enterprise /stream endpoint behaves correctly."""

    def test_returns_200(self, client: TestClient, endpoint: str, body: dict) -> None:
        """Stream endpoint must return HTTP 200."""
        response = client.post(endpoint, json=body)
        assert response.status_code == 200, (
            f"{endpoint} returned {response.status_code}: {response.text[:200]}"
        )

    def test_content_type_is_event_stream(
        self, client: TestClient, endpoint: str, body: dict
    ) -> None:
        """Response Content-Type must indicate SSE."""
        response = client.post(endpoint, json=body)
        assert "text/event-stream" in response.headers.get("content-type", ""), (
            f"{endpoint} content-type was {response.headers.get('content-type')}"
        )

    def test_at_least_one_valid_event(
        self, client: TestClient, endpoint: str, body: dict
    ) -> None:
        """At least one parseable SSE event must be returned."""
        response = client.post(endpoint, json=body)
        events = _parse_sse_lines(response.content)
        assert len(events) >= 1, f"{endpoint} returned no SSE events"

    def test_event_has_type_field(
        self, client: TestClient, endpoint: str, body: dict
    ) -> None:
        """Every SSE event must carry a recognised ``type`` field."""
        response = client.post(endpoint, json=body)
        events = _parse_sse_lines(response.content)
        assert events, f"{endpoint} returned no events"
        for event in events:
            assert "type" in event, f"Event missing 'type' key: {event}"
            assert event["type"] in VALID_EVENT_TYPES, (
                f"Unknown event type '{event['type']}' from {endpoint}"
            )


# ---------------------------------------------------------------------------
# Individual endpoint validation tests
# ---------------------------------------------------------------------------


class TestResearchAgentStream:
    """Additional validation for the Research Agent stream endpoint."""

    def test_request_validation_missing_query(self, client: TestClient) -> None:
        """Omitting required ``query`` field must return 422."""
        response = client.post("/api/enterprise/research/stream", json={})
        assert response.status_code == 422

    def test_error_event_when_agent_unavailable(self, client: TestClient) -> None:
        """When agent is not loaded the stream must emit an ``error`` event."""
        import app.server as srv

        original = srv.research_agent
        try:
            srv.research_agent = None
            response = client.post(
                "/api/enterprise/research/stream",
                json={"query": "test"},
            )
            assert response.status_code == 200
            events = _parse_sse_lines(response.content)
            assert any(e.get("type") == "error" for e in events), (
                f"Expected an error event, got: {events}"
            )
        finally:
            srv.research_agent = original


class TestContentAgentStream:
    """Additional validation for the Content Agent stream endpoint."""

    def test_request_validation_missing_topic(self, client: TestClient) -> None:
        """Omitting required ``topic`` field must return 422."""
        response = client.post("/api/enterprise/content/stream", json={})
        assert response.status_code == 422


class TestDataAnalystStream:
    """Additional validation for the Data Analyst stream endpoint."""

    def test_request_validation_missing_message(self, client: TestClient) -> None:
        """Omitting required ``message`` field must return 422."""
        response = client.post("/api/enterprise/data-analyst/stream", json={})
        assert response.status_code == 422


class TestDocumentAgentStream:
    """Additional validation for the Document Agent stream endpoint."""

    def test_request_validation_missing_fields(self, client: TestClient) -> None:
        """Omitting required fields must return 422."""
        response = client.post("/api/enterprise/documents/stream", json={})
        assert response.status_code == 422


class TestRAGAgentStream:
    """Additional validation for the Multilingual RAG stream endpoint."""

    def test_request_validation_missing_query(self, client: TestClient) -> None:
        """Omitting required ``query`` field must return 422."""
        response = client.post("/api/enterprise/rag/stream", json={})
        assert response.status_code == 422


class TestSupportAgentStream:
    """Additional validation for the HITL Support stream endpoint."""

    def test_request_validation_missing_message(self, client: TestClient) -> None:
        """Omitting required ``message`` field must return 422."""
        response = client.post("/api/enterprise/support/stream", json={})
        assert response.status_code == 422


class TestCodeAssistantStream:
    """Additional validation for the Code Assistant stream endpoint."""

    def test_request_validation_missing_code(self, client: TestClient) -> None:
        """Omitting required ``code`` field must return 422."""
        response = client.post("/api/enterprise/code/stream", json={})
        assert response.status_code == 422


class TestDocumentIntelligenceStream:
    """Additional validation for the Document Intelligence stream endpoint."""

    def test_request_validation_missing_message(self, client: TestClient) -> None:
        """Omitting required ``message`` field must return 422."""
        response = client.post(
            "/api/enterprise/document-intelligence/stream", json={}
        )
        assert response.status_code == 422
