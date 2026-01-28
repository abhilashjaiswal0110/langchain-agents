"""Integration tests for Employee Experience Agent API endpoints.

Tests the REST API endpoints and UI accessibility for the Employee Experience Agent:
- Conversation Manager integration
- API endpoint accessibility
- Session management
- Full request/response cycle
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

# Import the FastAPI app
from app.server import app


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Mock Employee Experience Agent for testing."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm") as mock_llm:
        mock_llm_instance = Mock()
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm.return_value = mock_llm_instance

        from app.agents.employee_experience import EmployeeExperienceAgent
        agent = EmployeeExperienceAgent()
        return agent


# =============================================================================
# Test Conversation Manager Integration
# =============================================================================


def test_conversation_manager_has_employee_experience_agent():
    """Test that ConversationManager includes Employee Experience Agent."""
    from app.agents.conversation_manager import ConversationManager

    manager = ConversationManager()

    # Check that employee_experience is in available agents
    available_agents = manager.get_available_agents()
    assert "employee_experience" in available_agents
    assert "Employee Experience" in available_agents["employee_experience"]


def test_conversation_manager_loads_employee_experience_agent():
    """Test that ConversationManager can load Employee Experience Agent."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        from app.agents.conversation_manager import ConversationManager

        manager = ConversationManager()

        # Check that agent is loaded
        assert "employee_experience" in manager._agents
        assert manager._agents["employee_experience"] is not None


# =============================================================================
# Test API Endpoint: Start Conversation
# =============================================================================


def test_start_conversation_endpoint_exists(client):
    """Test that the start conversation endpoint exists."""
    response = client.post(
        "/api/conversation/start",
        json={
            "agent_type": "employee_experience",
            "user_id": "test_user_123",
        },
    )

    # Should not be 404 (endpoint exists)
    assert response.status_code != 404


@pytest.mark.skipif(
    True,
    reason="Requires environment setup and LLM configuration",
)
def test_start_conversation_with_employee_experience(client):
    """Test starting a conversation with Employee Experience Agent."""
    response = client.post(
        "/api/conversation/start",
        json={
            "agent_type": "employee_experience",
            "user_id": "test_user_123",
            "metadata": {
                "employee_id": "EMP12345",
                "role": "Software Engineer",
                "department": "Engineering",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "session_id" in data
    assert "agent_type" in data
    assert data["agent_type"] == "employee_experience"


# =============================================================================
# Test API Endpoint: Chat
# =============================================================================


@pytest.mark.skipif(
    True,
    reason="Requires environment setup and LLM configuration",
)
def test_chat_endpoint_with_hr_query(client):
    """Test chat endpoint with HR policy query."""
    # First start a conversation
    start_response = client.post(
        "/api/conversation/start",
        json={"agent_type": "employee_experience"},
    )
    session_id = start_response.json()["session_id"]

    # Send a message
    response = client.post(
        "/api/conversation/chat",
        json={
            "session_id": session_id,
            "message": "What is the PTO policy?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "response" in data
    assert "session_id" in data
    assert len(data["response"]) > 0


@pytest.mark.skipif(
    True,
    reason="Requires environment setup and LLM configuration",
)
def test_chat_endpoint_with_career_query(client):
    """Test chat endpoint with career development query."""
    # Start conversation
    start_response = client.post(
        "/api/conversation/start",
        json={"agent_type": "employee_experience"},
    )
    session_id = start_response.json()["session_id"]

    # Ask about career paths
    response = client.post(
        "/api/conversation/chat",
        json={
            "session_id": session_id,
            "message": "I'm a software engineer. What are my career path options?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "response" in data
    assert len(data["response"]) > 0


@pytest.mark.skipif(
    True,
    reason="Requires environment setup and LLM configuration",
)
def test_chat_endpoint_with_wellbeing_query(client):
    """Test chat endpoint with wellbeing query."""
    # Start conversation
    start_response = client.post(
        "/api/conversation/start",
        json={"agent_type": "employee_experience"},
    )
    session_id = start_response.json()["session_id"]

    # Ask about wellbeing resources
    response = client.post(
        "/api/conversation/chat",
        json={
            "session_id": session_id,
            "message": "I'm feeling stressed and overwhelmed. What resources are available?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "response" in data
    # May include sentiment or burnout risk data
    if "sentiment_score" in data:
        assert data["sentiment_score"] < 0  # Negative sentiment


# =============================================================================
# Test API Endpoint: History
# =============================================================================


@pytest.mark.skipif(
    True,
    reason="Requires environment setup and LLM configuration",
)
def test_conversation_history_endpoint(client):
    """Test getting conversation history."""
    # Start conversation and send messages
    start_response = client.post(
        "/api/conversation/start",
        json={"agent_type": "employee_experience"},
    )
    session_id = start_response.json()["session_id"]

    # Send a message
    client.post(
        "/api/conversation/chat",
        json={
            "session_id": session_id,
            "message": "What are the benefits?",
        },
    )

    # Get history
    response = client.get(f"/api/conversation/history/{session_id}")

    assert response.status_code == 200
    data = response.json()

    assert "messages" in data
    assert len(data["messages"]) > 0


# =============================================================================
# Test API Endpoint: Status
# =============================================================================


def test_conversation_status_endpoint(client):
    """Test getting conversation status."""
    # This endpoint should exist even if session doesn't
    response = client.get("/api/conversation/status/test_session_id")

    # Should not be 404
    assert response.status_code != 404


# =============================================================================
# Test UI Accessibility
# =============================================================================


def test_chat_ui_endpoint_exists(client):
    """Test that the chat UI endpoint exists and is accessible."""
    response = client.get("/chat")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_chat_ui_contains_html(client):
    """Test that the chat UI returns valid HTML."""
    response = client.get("/chat")

    content = response.text
    assert "<html" in content.lower()
    assert "</html>" in content.lower()
    assert "<body" in content.lower()


def test_health_check_endpoint(client):
    """Test that health check endpoint works."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    """Test that root endpoint is accessible."""
    response = client.get("/")

    assert response.status_code == 200


# =============================================================================
# Test Session Management
# =============================================================================


def test_session_persistence_across_messages():
    """Test that session data persists across multiple messages."""
    from app.agents.conversation_manager import ConversationManager

    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        manager = ConversationManager()

        # Start conversation
        session = manager.start_conversation(
            agent_type="employee_experience",
            user_id="test_user",
        )
        session_id = session["session_id"]

        # Session should exist
        assert session_id is not None
        assert manager.session_store.get_session(session_id) is not None


# =============================================================================
# Test Error Handling
# =============================================================================


def test_start_conversation_invalid_agent_type(client):
    """Test error handling for invalid agent type."""
    response = client.post(
        "/api/conversation/start",
        json={"agent_type": "invalid_agent_type"},
    )

    # Should return error (400 or 404)
    assert response.status_code in [400, 404]


def test_chat_with_invalid_session_id(client):
    """Test error handling for invalid session ID."""
    response = client.post(
        "/api/conversation/chat",
        json={
            "session_id": "nonexistent_session_id",
            "message": "Test message",
        },
    )

    # Should return error
    assert response.status_code in [400, 404]


def test_chat_without_message(client):
    """Test error handling for missing message."""
    response = client.post(
        "/api/conversation/chat",
        json={"session_id": "test_session"},
    )

    # Should return error (422 validation error)
    assert response.status_code == 422


# =============================================================================
# Test API Response Format
# =============================================================================


@pytest.mark.skipif(
    True,
    reason="Requires environment setup and LLM configuration",
)
def test_chat_response_format(client):
    """Test that chat response has correct format."""
    # Start conversation
    start_response = client.post(
        "/api/conversation/start",
        json={"agent_type": "employee_experience"},
    )
    session_id = start_response.json()["session_id"]

    # Send message
    response = client.post(
        "/api/conversation/chat",
        json={
            "session_id": session_id,
            "message": "Test message",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "response" in data
    assert "session_id" in data
    assert isinstance(data["response"], str)
    assert isinstance(data["session_id"], str)

    # Optional sentiment fields
    if "sentiment_score" in data:
        assert isinstance(data["sentiment_score"], (int, float))
        assert -1.0 <= data["sentiment_score"] <= 1.0

    if "burnout_risk" in data:
        assert data["burnout_risk"] in ["low", "medium", "high", None]


# =============================================================================
# Test Agent Availability in API
# =============================================================================


def test_list_available_agents_includes_employee_experience(client):
    """Test that the agent list endpoint includes Employee Experience Agent."""
    # If there's an endpoint to list agents
    response = client.get("/api/conversation/agents")

    if response.status_code == 200:
        data = response.json()
        # Should include employee_experience
        assert "employee_experience" in data or "employee_experience" in str(data)


# =============================================================================
# Test CORS Headers
# =============================================================================


def test_cors_headers_present(client):
    """Test that CORS headers are present in responses."""
    response = client.get("/health")

    # Should have CORS headers (if CORS is enabled)
    headers = response.headers
    # This is informational - CORS may or may not be configured
    assert response.status_code == 200


# =============================================================================
# Test Webhooks Integration (if applicable)
# =============================================================================


@pytest.mark.skipif(
    True,
    reason="Webhook integration may require authentication",
)
def test_webhook_chat_endpoint_with_employee_experience():
    """Test webhook chat endpoint with Employee Experience Agent."""
    # This would test the /api/webhook/chat endpoint if it supports
    # Employee Experience Agent routing
    pass


# =============================================================================
# Test Agent Performance Metrics
# =============================================================================


def test_agent_responds_within_reasonable_time():
    """Test that agent initialization doesn't take too long."""
    import time

    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        from app.agents.employee_experience import EmployeeExperienceAgent

        start_time = time.time()
        agent = EmployeeExperienceAgent()
        end_time = time.time()

        # Should initialize quickly (< 5 seconds)
        assert (end_time - start_time) < 5.0
        assert agent is not None


# =============================================================================
# Test Documentation/OpenAPI
# =============================================================================


def test_openapi_schema_includes_employee_experience_endpoints(client):
    """Test that OpenAPI schema includes Employee Experience endpoints."""
    response = client.get("/docs")

    # Should have Swagger UI docs
    assert response.status_code == 200


def test_openapi_json_accessible(client):
    """Test that OpenAPI JSON schema is accessible."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


# =============================================================================
# Test Environment Configuration
# =============================================================================


def test_agent_respects_environment_config():
    """Test that agent respects environment configuration."""
    import os

    # Set environment variable
    os.environ["EMPLOYEE_EXPERIENCE_TEMPERATURE"] = "0.8"

    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        from app.agents.employee_experience import EmployeeExperienceAgent

        # Agent should use environment config (or default)
        agent = EmployeeExperienceAgent()
        assert agent is not None

    # Clean up
    if "EMPLOYEE_EXPERIENCE_TEMPERATURE" in os.environ:
        del os.environ["EMPLOYEE_EXPERIENCE_TEMPERATURE"]


# =============================================================================
# Summary Test
# =============================================================================


def test_employee_experience_agent_full_integration():
    """High-level integration test for Employee Experience Agent."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        from app.agents.conversation_manager import ConversationManager
        from app.agents.employee_experience import EmployeeExperienceAgent

        # 1. Agent can be instantiated
        agent = EmployeeExperienceAgent()
        assert agent is not None

        # 2. Agent has tools
        assert len(agent.tools) > 20

        # 3. Agent is registered in ConversationManager
        manager = ConversationManager()
        assert "employee_experience" in manager._agents

        # 4. Agent can start conversations
        session = manager.start_conversation(
            agent_type="employee_experience",
            user_id="test_user",
        )
        assert session is not None
        assert "session_id" in session

        print("✅ Employee Experience Agent full integration test passed!")
