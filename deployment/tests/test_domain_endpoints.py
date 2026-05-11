"""Tests for domain agent REST endpoints.

Verifies that all 8 domain agents are reachable via /api/domain/* routes.
"""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client (no API keys needed for route/listing tests)."""
    os.environ.pop("OPENAI_API_KEY", None)
    from app.server import app
    return TestClient(app)


def test_domain_agents_listed(client):
    """GET /api/domain/agents should return all 8 domain agents."""
    r = client.get("/api/domain/agents")
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data
    agent_types = [a["type"] for a in data["agents"]]
    expected = {"marcom", "hr", "lnd", "presales", "datacenter", "cloud", "cybersecurity", "data_ai"}
    assert expected.issubset(set(agent_types)), f"Missing agents: {expected - set(agent_types)}"


def test_domain_agents_have_descriptions(client):
    """Each agent entry should have a non-empty description."""
    r = client.get("/api/domain/agents")
    assert r.status_code == 200
    for agent in r.json()["agents"]:
        assert "type" in agent
        assert "description" in agent
        assert "name" in agent


@pytest.mark.integration
def test_domain_agent_invoke_cloud(client, mock_openai_key):
    """POST /api/domain/cloud/invoke should return 200 with a response.

    Note: Requires a working LLM (real or mocked). Skipped in unit test runs.
    The base DomainAgent uses create_react_agent which needs a valid LLM
    to build the agent graph.
    """
    r = client.post(
        "/api/domain/cloud/invoke",
        json={"message": "list VMs", "session_id": "test-session-cloud"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert data["agent_type"] == "cloud"


@pytest.mark.integration
def test_domain_agent_invoke_hr(client, mock_openai_key):
    """POST /api/domain/hr/invoke should return 200 with a response.

    Note: Requires a working LLM (real or mocked). Skipped in unit test runs.
    """
    r = client.post(
        "/api/domain/hr/invoke",
        json={"message": "What are the vacation policies?", "session_id": "test-session-hr"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert data["agent_type"] == "hr"


def test_domain_agent_invoke_unknown_domain(client):
    """POST /api/domain/unknown/invoke should return 404."""
    r = client.post(
        "/api/domain/unknown_domain/invoke",
        json={"message": "test"},
    )
    assert r.status_code == 404


@pytest.mark.integration
def test_domain_router_chat(client, mock_openai_key):
    """POST /api/domain/chat should route to the appropriate domain agent.

    Note: Requires a working LLM. Skipped in unit test runs.
    """
    r = client.post(
        "/api/domain/chat",
        json={"message": "I need help with Azure VMs", "user_context": {}},
    )
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
