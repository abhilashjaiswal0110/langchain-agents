"""Tests for enterprise agent API endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestEnterpriseAgentsList:
    """Tests for enterprise agents listing endpoint."""

    def test_list_agents_endpoint(self, test_client):
        """Test listing available enterprise agents."""
        response = test_client.get("/api/enterprise/agents")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "agents" in data

        # Check all agent types are listed
        expected_agents = [
            "research",
            "content",
            "data_analyst",
            "document",
            "multilingual_rag",
            "hitl_support",
            "code_assistant",
        ]
        for agent in expected_agents:
            assert agent in data["agents"]
            assert "description" in data["agents"][agent]
            assert "endpoint" in data["agents"][agent]
            assert "loaded" in data["agents"][agent]


class TestResearchAgentEndpoint:
    """Tests for Research Agent endpoint."""

    def test_research_unavailable_without_api_key(self, test_client):
        """Test research agent returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/research/invoke",
            json={"query": "What is LangGraph?"},
        )
        assert response.status_code == 503

    def test_research_request_validation(self, test_client):
        """Test research agent request validation."""
        response = test_client.post(
            "/api/enterprise/research/invoke",
            json={},  # Missing required query
        )
        assert response.status_code == 422  # Validation error


class TestContentAgentEndpoint:
    """Tests for Content Generation Agent endpoint."""

    def test_content_unavailable_without_api_key(self, test_client):
        """Test content agent returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/content/invoke",
            json={
                "topic": "AI agents",
                "platform": "linkedin",
            },
        )
        assert response.status_code == 503

    def test_content_request_validation(self, test_client):
        """Test content agent request validation."""
        response = test_client.post(
            "/api/enterprise/content/invoke",
            json={},  # Missing required topic
        )
        assert response.status_code == 422

    def test_content_platform_validation(self, test_client):
        """Test content agent platform validation."""
        response = test_client.post(
            "/api/enterprise/content/invoke",
            json={
                "topic": "AI agents",
                "platform": "invalid_platform",
            },
        )
        assert response.status_code == 422


class TestDataAnalystEndpoint:
    """Tests for Data Analyst Agent endpoint."""

    def test_data_analyst_unavailable_without_api_key(self, test_client):
        """Test data analyst returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/data-analyst/invoke",
            json={"message": "Analyze the data"},
        )
        assert response.status_code == 503

    def test_data_analyst_upload_unavailable(self, test_client):
        """Test data analyst upload returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/data-analyst/upload",
            files={"file": ("test.csv", b"col1,col2\n1,2", "text/csv")},
        )
        assert response.status_code == 503


class TestDocumentAgentEndpoint:
    """Tests for Document Generator Agent endpoint."""

    def test_document_unavailable_without_api_key(self, test_client):
        """Test document agent returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/documents/invoke",
            json={
                "doc_type": "sop",
                "title": "Password Reset",
                "description": "Standard procedure for password reset",
            },
        )
        assert response.status_code == 503

    def test_document_request_validation(self, test_client):
        """Test document agent request validation."""
        response = test_client.post(
            "/api/enterprise/documents/invoke",
            json={"doc_type": "sop"},  # Missing required fields
        )
        assert response.status_code == 422

    def test_document_type_validation(self, test_client):
        """Test document type validation."""
        response = test_client.post(
            "/api/enterprise/documents/invoke",
            json={
                "doc_type": "invalid_type",
                "title": "Test",
                "description": "Test",
            },
        )
        assert response.status_code == 422


class TestRAGAgentEndpoint:
    """Tests for Multilingual RAG Agent endpoint."""

    def test_rag_unavailable_without_api_key(self, test_client):
        """Test RAG agent returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/rag/invoke",
            json={"query": "What is in the document?"},
        )
        assert response.status_code == 503

    def test_rag_upload_unavailable(self, test_client):
        """Test RAG upload returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/rag/upload",
            files={"file": ("test.txt", b"Test content", "text/plain")},
        )
        assert response.status_code == 503


class TestHITLSupportEndpoint:
    """Tests for HITL Support Agent endpoint."""

    def test_support_unavailable_without_api_key(self, test_client):
        """Test support agent returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/support/invoke",
            json={"message": "I need help with my email"},
        )
        assert response.status_code == 503

    def test_approval_unavailable_without_api_key(self, test_client):
        """Test approval endpoint returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/support/approve",
            json={
                "session_id": "test-session",
                "action_id": "test-action",
                "approved": True,
            },
        )
        assert response.status_code == 503


class TestCodeAssistantEndpoint:
    """Tests for Code Assistant Agent endpoint."""

    def test_code_unavailable_without_api_key(self, test_client):
        """Test code assistant returns 503 without API key."""
        response = test_client.post(
            "/api/enterprise/code/invoke",
            json={
                "code": "def foo(): pass",
                "language": "python",
                "action": "analyze",
            },
        )
        assert response.status_code == 503

    def test_code_action_validation(self, test_client):
        """Test code assistant action validation."""
        response = test_client.post(
            "/api/enterprise/code/invoke",
            json={
                "code": "def foo(): pass",
                "action": "invalid_action",
            },
        )
        assert response.status_code == 422


class TestHealthCheckWithEnterpriseAgents:
    """Tests for health check with enterprise agents."""

    def test_health_includes_enterprise_status(self, test_client):
        """Test health check includes enterprise agents status."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "enterprise_agents_loaded" in data
        assert isinstance(data["enterprise_agents_loaded"], bool)


class TestWebhookIntegration:
    """Tests for webhook integration endpoints."""

    def test_webhook_chat_endpoint_exists(self, test_client):
        """Test webhook chat endpoint exists."""
        response = test_client.post(
            "/api/webhook/chat",
            json={
                "event_type": "conversation.start",
                "agent_type": "it_helpdesk",
            },
        )
        # Should return response (may fail without agents, but endpoint exists)
        assert response.status_code in [200, 503]

    def test_webhook_conversation_message(self, test_client):
        """Test webhook conversation message."""
        response = test_client.post(
            "/api/webhook/chat",
            json={
                "event_type": "conversation.message",
                "session_id": "test-session",
                "message": "Hello",
            },
        )
        # Should return response (endpoint exists)
        assert response.status_code in [200, 503]

    def test_webhook_unknown_event(self, test_client):
        """Test webhook with unknown event type."""
        response = test_client.post(
            "/api/webhook/chat",
            json={
                "event_type": "unknown.event",
            },
        )
        # Should return error about unknown event
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is False


class TestThirdPartyWebhooks:
    """Tests for 3rd party platform webhook endpoints."""

    def test_copilot_studio_endpoint_exists(self, test_client):
        """Test Copilot Studio webhook endpoint exists."""
        response = test_client.post(
            "/api/webhooks/copilot-studio",
            json={
                "query": "What is Python?",
                "agent_type": "research",
            },
        )
        # Should return response (may fail without agents, but endpoint exists)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "source" in data
        assert data["source"] == "copilot-studio"

    def test_copilot_studio_with_metadata(self, test_client):
        """Test Copilot Studio webhook with full metadata."""
        response = test_client.post(
            "/api/webhooks/copilot-studio",
            json={
                "query": "Explain machine learning",
                "agent_type": "research",
                "user_id": "user-123",
                "conversation_id": "conv-456",
                "channel": "teams",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "copilot-studio"
        assert data["agent_type"] == "research"

    def test_azure_ai_endpoint_exists(self, test_client):
        """Test Azure AI webhook endpoint exists."""
        response = test_client.post(
            "/api/webhooks/azure-ai",
            json={
                "query": "Analyze this data",
                "agent_type": "data-analyst",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "source" in data
        assert data["source"] == "azure-ai"

    def test_azure_ai_with_deployment_info(self, test_client):
        """Test Azure AI webhook with deployment metadata."""
        response = test_client.post(
            "/api/webhooks/azure-ai",
            json={
                "query": "Generate a report",
                "agent_type": "document",
                "deployment_id": "dep-123",
                "resource_group": "rg-ai",
                "subscription_id": "sub-456",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "azure-ai"
        assert data["agent_type"] == "document"

    def test_aws_lex_endpoint_exists(self, test_client):
        """Test AWS Lex webhook endpoint exists."""
        response = test_client.post(
            "/api/webhooks/aws-lex",
            json={
                "query": "Review this code",
                "agent_type": "code-assistant",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "source" in data
        assert data["source"] == "aws-lex"

    def test_aws_lex_with_bot_info(self, test_client):
        """Test AWS Lex webhook with bot metadata."""
        response = test_client.post(
            "/api/webhooks/aws-lex",
            json={
                "query": "Translate this text",
                "agent_type": "multilingual-rag",
                "bot_id": "bot-123",
                "bot_alias_id": "alias-456",
                "locale_id": "de_DE",
                "session_attributes": {"user_tier": "premium"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "aws-lex"
        assert data["agent_type"] == "multilingual-rag"

    def test_invalid_agent_type_copilot(self, test_client):
        """Test Copilot Studio with invalid agent type."""
        response = test_client.post(
            "/api/webhooks/copilot-studio",
            json={
                "query": "Test query",
                "agent_type": "invalid-agent",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Either success=False (agent not found) or returns error
        if not data["success"]:
            assert data["error"] is not None

    def test_missing_query_validation(self, test_client):
        """Test validation for missing query parameter."""
        response = test_client.post(
            "/api/webhooks/azure-ai",
            json={
                "agent_type": "research",
            },
        )
        # Should fail validation (422) because query is required
        assert response.status_code == 422


class TestConversationAPI:
    """Tests for conversation API endpoints."""

    def test_conversation_start_endpoint(self, test_client):
        """Test conversation start endpoint."""
        response = test_client.post(
            "/api/conversation/start",
            json={
                "agent_type": "it_helpdesk",
            },
        )
        # Should exist (may fail without API keys)
        assert response.status_code in [200, 503]

    def test_agents_list_endpoint(self, test_client):
        """Test agents list endpoint."""
        response = test_client.get("/api/agents")
        assert response.status_code == 200

        data = response.json()
        assert "agents" in data
        assert "status" in data
