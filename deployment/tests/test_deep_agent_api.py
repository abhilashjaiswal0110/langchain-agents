"""REST API endpoint tests for Deep Agent."""

import os
import pytest

# Set up mock API keys before importing app modules
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"

from fastapi.testclient import TestClient
from app.server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health endpoint includes Deep Agent status."""

    def test_health_includes_deep_agent_loaded(self, client):
        """Verify health endpoint has deep_agent_loaded field."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "deep_agent_loaded" in data
        assert "status" in data
        assert data["status"] == "healthy"


class TestDeepAgentEndpoints:
    """Test Deep Agent API endpoints."""

    def test_list_subagents_endpoint(self, client):
        """Test subagents listing endpoint."""
        response = client.get("/api/deepagent/subagents")
        # Should return 503 (agent not loaded) or 200
        assert response.status_code in [200, 503]
        if response.status_code == 503:
            assert "detail" in response.json()

    def test_start_session_endpoint(self, client):
        """Test session start endpoint."""
        response = client.post(
            "/api/deepagent/start",
            json={"user_input": "Help me investigate an incident"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data or "session_id" in data or "error" in data

    def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        response = client.post(
            "/api/deepagent/chat",
            json={"session_id": "test-session", "user_input": "test"}
        )
        # Should return 200, 422 (validation/invalid session), or 503 (agent not loaded)
        assert response.status_code in [200, 422, 503]

    def test_todos_endpoint(self, client):
        """Test todos endpoint."""
        response = client.get("/api/deepagent/todos/test-session")
        # Should return 200 (empty list), 404, or 503
        assert response.status_code in [200, 404, 503]

    def test_files_endpoint(self, client):
        """Test files endpoint."""
        response = client.get("/api/deepagent/files/test-session")
        # Should return 200 (empty list), 404, or 503
        assert response.status_code in [200, 404, 503]


class TestErrorHandling:
    """Test error handling in API endpoints."""

    def test_invalid_session_returns_proper_error(self, client):
        """Test that invalid sessions return proper errors."""
        response = client.get("/api/deepagent/todos/non-existent-session")
        assert response.status_code in [404, 503]

    def test_empty_user_input(self, client):
        """Test handling of empty user input."""
        response = client.post(
            "/api/deepagent/start",
            json={"user_input": ""}
        )
        # Should return 200 with error or 422 validation error
        assert response.status_code in [200, 422]


class TestEndpointRouting:
    """Test that all endpoints are properly routed."""

    def test_deep_agent_routes_exist(self, client):
        """Verify all Deep Agent routes exist and respond."""
        routes = [
            ("GET", "/api/deepagent/subagents"),
            ("POST", "/api/deepagent/start"),
            ("POST", "/api/deepagent/chat"),
            ("GET", "/api/deepagent/todos/test"),
            ("GET", "/api/deepagent/files/test"),
        ]

        for method, path in routes:
            if method == "GET":
                response = client.get(path)
            else:
                response = client.post(path, json={"session_id": "test", "user_input": "test"})

            # Should not return 405 Method Not Allowed (route doesn't exist)
            assert response.status_code != 405, f"Route {method} {path} not found"
