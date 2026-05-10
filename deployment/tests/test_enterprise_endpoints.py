"""Parametrized tests for all 8 enterprise agent /invoke and /stream endpoints.

Tests are structured around three concerns:
- Status code: does the endpoint return HTTP 200?
- Response shape: does the body contain the expected fields?
- Validation: does the endpoint return 422 when required fields are missing?

All tests mock at the ``app.server`` module level so no real LLM calls are made.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_agent_state(text: str = "mocked response") -> dict:
    """Return a LangGraph-style state dict with an AIMessage as the last message."""
    return {"messages": [AIMessage(content=text)]}


async def _async_gen_events():
    """Async generator that yields a single token SSE event."""
    yield {"type": "token", "data": {"text": "t"}}


def _make_astream():
    """Return a callable that produces a fresh async generator on each call."""
    def _astream(*args, **kwargs):
        return _async_gen_events()
    return _astream


def _mock_research() -> MagicMock:
    m = MagicMock()
    m.research.return_value = _make_agent_state("Research result")
    m.astream = _make_astream()
    return m


def _mock_content() -> MagicMock:
    m = MagicMock()
    m.create_content.return_value = _make_agent_state("Content result")
    m.astream = _make_astream()
    return m


def _mock_data_analyst() -> MagicMock:
    m = MagicMock()
    m.invoke.return_value = _make_agent_state("Analysis result")
    m.astream = _make_astream()
    return m


def _mock_document() -> MagicMock:
    m = MagicMock()
    m.create_document.return_value = _make_agent_state("Document result")
    m.astream = _make_astream()
    return m


def _mock_rag() -> MagicMock:
    m = MagicMock()
    m.query.return_value = _make_agent_state("RAG answer")
    m.astream = _make_astream()
    return m


def _mock_hitl() -> MagicMock:
    m = MagicMock()
    m.invoke.return_value = _make_agent_state("Support response")
    m.astream = _make_astream()
    return m


def _mock_code_assistant() -> MagicMock:
    m = MagicMock()
    m.analyze.return_value = _make_agent_state("Code analysis")
    m.modernize.return_value = _make_agent_state("Modernized code")
    m.astream = _make_astream()
    return m


def _mock_doc_intelligence() -> MagicMock:
    m = MagicMock()
    m.chat.return_value = {"response": "Document intelligence response", "session_id": "test-session"}
    m.astream = _make_astream()
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Test client with no API keys (agents not loaded by default)."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    from app.server import app
    return TestClient(app)


@pytest.fixture
def mock_enterprise_agents():
    """Patch all enterprise agent module-level variables in app.server.

    Injects mock objects for every agent and sets ``enterprise_agents_loaded``
    to ``True`` so endpoints proceed past the availability guard.
    """
    import app.server as server_module

    patches = {
        "enterprise_agents_loaded": True,
        "research_agent": _mock_research(),
        "content_agent": _mock_content(),
        "data_analyst_agent": _mock_data_analyst(),
        "document_agent": _mock_document(),
        "multilingual_rag_agent": _mock_rag(),
        "hitl_support_agent": _mock_hitl(),
        "code_assistant_agent": _mock_code_assistant(),
        "document_intelligence_agent": _mock_doc_intelligence(),
    }

    original = {k: getattr(server_module, k) for k in patches}
    for k, v in patches.items():
        setattr(server_module, k, v)

    yield patches

    # Restore originals
    for k, v in original.items():
        setattr(server_module, k, v)


# ---------------------------------------------------------------------------
# Parametrize data
# ENTERPRISE_AGENTS: (path_segment, valid_payload, missing_required_key)
# ---------------------------------------------------------------------------

ENTERPRISE_AGENTS = [
    (
        "research",
        {"query": "AI trends 2025"},
        {},  # missing required 'query'
    ),
    (
        "content",
        {"topic": "AI in enterprise", "platform": "linkedin", "tone": "professional"},
        {},  # missing required 'topic'
    ),
    (
        "data-analyst",
        {"message": "Summarize the uploaded data"},
        {},  # missing required 'message'
    ),
    (
        "documents",
        {"doc_type": "sop", "title": "Password Reset SOP", "description": "Steps to reset passwords"},
        {"doc_type": "sop"},  # missing required 'title' and 'description'
    ),
    (
        "rag",
        {"query": "What is the policy on remote work?"},
        {},  # missing required 'query'
    ),
    (
        "support",
        {"message": "My laptop won't start"},
        {},  # missing required 'message'
    ),
    (
        "code",
        {"code": "def hello(): print('hello')", "language": "python", "action": "analyze"},
        {},  # missing required 'code'
    ),
    (
        "document-intelligence",
        {"message": "Summarize the document"},
        {},  # missing required 'message'
    ),
]

# Only agent+payload pairs (no missing payload) for happy-path parametrize
INVOKE_PARAMS = [(agent, payload) for agent, payload, _ in ENTERPRISE_AGENTS]
AGENT_NAMES = [agent for agent, _, _ in ENTERPRISE_AGENTS]
VALIDATION_PARAMS = [(agent, missing) for agent, _, missing in ENTERPRISE_AGENTS]


# ---------------------------------------------------------------------------
# Happy-path tests: /invoke returns 200
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent,payload", INVOKE_PARAMS)
def test_enterprise_invoke_status_200(client, mock_enterprise_agents, agent, payload):
    """POST /api/enterprise/{agent}/invoke should return HTTP 200 when agent is loaded."""
    r = client.post(f"/api/enterprise/{agent}/invoke", json=payload)
    assert r.status_code == 200, (
        f"Agent '{agent}' returned {r.status_code}: {r.text[:300]}"
    )


# ---------------------------------------------------------------------------
# Response shape tests: body contains 'response' field or success key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent,payload", INVOKE_PARAMS)
def test_enterprise_invoke_response_shape(client, mock_enterprise_agents, agent, payload):
    """Response body must contain 'success' and either 'response' or an error string."""
    r = client.post(f"/api/enterprise/{agent}/invoke", json=payload)
    assert r.status_code == 200
    body = r.json()
    # All enterprise agents use EnterpriseAgentResponse which has 'success'
    assert "success" in body, f"Agent '{agent}' body missing 'success': {body}"
    assert body["success"] is True, f"Agent '{agent}' returned success=False: {body}"
    # 'response' should be present (may be empty string but field must exist)
    assert "response" in body, f"Agent '{agent}' body missing 'response': {body}"


# ---------------------------------------------------------------------------
# Validation tests: missing required fields return 422
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent,missing_payload", VALIDATION_PARAMS)
def test_enterprise_invoke_missing_required_field_422(client, agent, missing_payload):
    """POST /api/enterprise/{agent}/invoke with missing required fields should return 422."""
    r = client.post(f"/api/enterprise/{agent}/invoke", json=missing_payload)
    assert r.status_code == 422, (
        f"Agent '{agent}' should return 422 for payload {missing_payload!r}, got {r.status_code}"
    )


# ---------------------------------------------------------------------------
# 503 when agents are NOT loaded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent,payload", INVOKE_PARAMS)
def test_enterprise_invoke_503_when_not_loaded(client, agent, payload):
    """POST /api/enterprise/{agent}/invoke should return 503 when enterprise agents are not loaded."""
    # No mock_enterprise_agents fixture → agents remain None / loaded=False
    r = client.post(f"/api/enterprise/{agent}/invoke", json=payload)
    assert r.status_code == 503, (
        f"Agent '{agent}' should return 503 when not loaded, got {r.status_code}: {r.text[:200]}"
    )


# ---------------------------------------------------------------------------
# Stream endpoints return 200 with text/event-stream (or SSE error when not loaded)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent,payload", INVOKE_PARAMS)
def test_enterprise_stream_returns_200_and_event_stream(client, mock_enterprise_agents, agent, payload):
    """POST /api/enterprise/{agent}/stream should return 200 with text/event-stream content-type."""
    r = client.post(f"/api/enterprise/{agent}/stream", json=payload)
    assert r.status_code == 200, (
        f"Stream for agent '{agent}' returned {r.status_code}: {r.text[:300]}"
    )
    content_type = r.headers.get("content-type", "")
    assert "text/event-stream" in content_type, (
        f"Agent '{agent}' stream content-type is '{content_type}', expected text/event-stream"
    )


@pytest.mark.parametrize("agent,payload", INVOKE_PARAMS)
def test_enterprise_stream_returns_event_stream_when_not_loaded(client, agent, payload):
    """Stream endpoints should still return 200 with SSE error when agents are not loaded."""
    r = client.post(f"/api/enterprise/{agent}/stream", json=payload)
    # Streams degrade gracefully: return SSE error event instead of HTTP error
    assert r.status_code == 200, (
        f"Stream for unloaded agent '{agent}' returned {r.status_code}: {r.text[:200]}"
    )
    content_type = r.headers.get("content-type", "")
    assert "text/event-stream" in content_type


# ---------------------------------------------------------------------------
# List endpoint
# ---------------------------------------------------------------------------

def test_enterprise_agents_list_returns_200(client):
    """GET /api/enterprise/agents should return 200."""
    r = client.get("/api/enterprise/agents")
    assert r.status_code == 200


def test_enterprise_agents_list_has_8_agents(client):
    """GET /api/enterprise/agents should list all 8 agents."""
    r = client.get("/api/enterprise/agents")
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data
    assert len(data["agents"]) >= 8, (
        f"Expected at least 8 agents, got {len(data['agents'])}: {list(data['agents'].keys())}"
    )


def test_enterprise_agents_list_has_required_fields(client):
    """Each agent entry in the list must have 'description', 'endpoint', and 'loaded' fields."""
    r = client.get("/api/enterprise/agents")
    assert r.status_code == 200
    data = r.json()
    for name, info in data["agents"].items():
        assert "description" in info, f"Agent '{name}' missing 'description'"
        assert "endpoint" in info, f"Agent '{name}' missing 'endpoint'"
        assert "loaded" in info, f"Agent '{name}' missing 'loaded'"


# ---------------------------------------------------------------------------
# 404 for unknown agent slug
# ---------------------------------------------------------------------------

def test_unknown_enterprise_agent_invoke_404(client):
    """POST /api/enterprise/nonexistent/invoke should return 404."""
    r = client.post("/api/enterprise/nonexistent/invoke", json={"input": "test"})
    assert r.status_code == 404


def test_unknown_enterprise_agent_stream_404(client):
    """POST /api/enterprise/nonexistent/stream should return 404."""
    r = client.post("/api/enterprise/nonexistent/stream", json={"input": "test"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Code assistant: modernize action
# ---------------------------------------------------------------------------

def test_code_assistant_modernize_action(client, mock_enterprise_agents):
    """Code assistant 'modernize' action should return 200 with response."""
    r = client.post(
        "/api/enterprise/code/invoke",
        json={"code": "var x = 1;", "language": "javascript", "action": "modernize"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "response" in body


def test_code_assistant_invalid_action_422(client):
    """Code assistant with invalid 'action' should return 422."""
    r = client.post(
        "/api/enterprise/code/invoke",
        json={"code": "x = 1", "action": "invalid_action"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Content agent: invalid platform returns 422
# ---------------------------------------------------------------------------

def test_content_agent_invalid_platform_422(client):
    """Content agent with an unsupported platform should return 422."""
    r = client.post(
        "/api/enterprise/content/invoke",
        json={"topic": "AI trends", "platform": "myspace"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Document agent: invalid doc_type returns 422
# ---------------------------------------------------------------------------

def test_document_agent_invalid_doc_type_422(client):
    """Document agent with an unsupported doc_type should return 422."""
    r = client.post(
        "/api/enterprise/documents/invoke",
        json={"doc_type": "spreadsheet", "title": "Test", "description": "Test desc"},
    )
    assert r.status_code == 422
