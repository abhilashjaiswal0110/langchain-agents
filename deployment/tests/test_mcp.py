"""Unit tests for the MCP integration module.

Tests cover:
- MCP Gateway (auth, rate limiting, permissions)
- ServiceNow client (live and simulation modes)
- MCP server tools
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Gateway tests
from app.mcp.gateway import (
    MCPClientInfo,
    MCPGateway,
    MCPGatewayConfig,
    MCPRateLimiter,
    MCPRequest,
    check_mcp_permission,
    get_mcp_gateway,
    reset_mcp_gateway,
)

# ServiceNow tests
from app.mcp.servicenow_client import (
    ServiceNowClient,
    ServiceNowConfig,
    ServiceNowSimulator,
    get_servicenow_client,
    reset_servicenow_client,
)


# ==================== Gateway Tests ====================


class TestMCPGatewayConfig:
    """Tests for MCPGatewayConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MCPGatewayConfig()
        assert config.enabled is True
        assert config.require_auth is False
        assert config.rate_limit_enabled is True
        assert config.rate_limit_per_minute == 60

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        with patch.dict("os.environ", {
            "MCP_GATEWAY_ENABLED": "false",
            "MCP_REQUIRE_AUTH": "true",
            "MCP_RATE_LIMIT_PER_MINUTE": "100",
        }):
            config = MCPGatewayConfig.from_env()
            assert config.enabled is False
            assert config.require_auth is True
            assert config.rate_limit_per_minute == 100

    def test_allowed_tools_parsing(self) -> None:
        """Test parsing allowed tools from environment."""
        with patch.dict("os.environ", {
            "MCP_ALLOWED_TOOLS": "research_topic, create_incident, get_incident",
        }):
            config = MCPGatewayConfig.from_env()
            assert "research_topic" in config.allowed_tools
            assert "create_incident" in config.allowed_tools
            assert len(config.allowed_tools) == 3


class TestMCPClientInfo:
    """Tests for MCPClientInfo."""

    def test_client_info_creation(self) -> None:
        """Test creating client info."""
        info = MCPClientInfo(client_id="client1", user_id="user123", role="admin")
        assert info.client_id == "client1"
        assert info.user_id == "user123"
        assert info.role == "admin"

    def test_default_values(self) -> None:
        """Test default values."""
        info = MCPClientInfo(client_id="client1")
        assert info.user_id is None
        assert info.role == "user"
        assert info.metadata == {}


class TestMCPRequest:
    """Tests for MCPRequest."""

    def test_request_creation(self) -> None:
        """Test creating a request."""
        client = MCPClientInfo(client_id="client1")
        request = MCPRequest(
            tool_name="research_topic",
            arguments={"query": "test"},
            client=client,
        )
        assert request.tool_name == "research_topic"
        assert request.arguments == {"query": "test"}
        assert request.request_id != ""  # Auto-generated

    def test_request_id_generation(self) -> None:
        """Test that request IDs are unique."""
        client = MCPClientInfo(client_id="client1")
        request1 = MCPRequest(tool_name="tool1", arguments={}, client=client)
        request2 = MCPRequest(tool_name="tool2", arguments={}, client=client)
        assert request1.request_id != request2.request_id


class TestMCPRateLimiter:
    """Tests for MCPRateLimiter."""

    def test_allows_within_limit(self) -> None:
        """Test that requests within limit are allowed."""
        limiter = MCPRateLimiter(requests_per_minute=10)

        for _ in range(10):
            assert limiter.check("client1") is True

    def test_blocks_over_limit(self) -> None:
        """Test that requests over limit are blocked."""
        limiter = MCPRateLimiter(requests_per_minute=5)

        for _ in range(5):
            limiter.check("client1")

        assert limiter.check("client1") is False

    def test_separate_clients(self) -> None:
        """Test that clients have separate limits."""
        limiter = MCPRateLimiter(requests_per_minute=3)

        # Exhaust client1's limit
        for _ in range(3):
            limiter.check("client1")

        # client2 should still have quota
        assert limiter.check("client2") is True

    def test_reset_client(self) -> None:
        """Test resetting a client's rate limit."""
        limiter = MCPRateLimiter(requests_per_minute=2)

        limiter.check("client1")
        limiter.check("client1")
        assert limiter.check("client1") is False

        limiter.reset("client1")
        assert limiter.check("client1") is True


class TestMCPGateway:
    """Tests for MCPGateway."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_mcp_gateway()

    def test_gateway_creation(self) -> None:
        """Test creating a gateway."""
        gateway = MCPGateway()
        assert gateway is not None
        assert gateway.config.enabled is True

    def test_register_and_validate_token(self) -> None:
        """Test token registration and validation."""
        gateway = MCPGateway()
        client_info = MCPClientInfo(client_id="client1", user_id="user1")

        gateway.register_token("my-token", client_info)

        validated = gateway.validate_token("my-token")
        assert validated is not None
        assert validated.client_id == "client1"

        # Invalid token
        assert gateway.validate_token("invalid") is None

    def test_tool_allowed_default(self) -> None:
        """Test that tools are allowed by default."""
        gateway = MCPGateway()
        assert gateway.is_tool_allowed("research_topic") is True
        assert gateway.is_tool_allowed("any_tool") is True

    def test_tool_blocked(self) -> None:
        """Test blocking specific tools."""
        config = MCPGatewayConfig(blocked_tools=["dangerous_tool"])
        gateway = MCPGateway(config)

        assert gateway.is_tool_allowed("research_topic") is True
        assert gateway.is_tool_allowed("dangerous_tool") is False

    def test_tool_allowlist(self) -> None:
        """Test allowing only specific tools."""
        config = MCPGatewayConfig(allowed_tools=["research_topic", "get_incident"])
        gateway = MCPGateway(config)

        assert gateway.is_tool_allowed("research_topic") is True
        assert gateway.is_tool_allowed("get_incident") is True
        assert gateway.is_tool_allowed("other_tool") is False

    @pytest.mark.asyncio
    async def test_check_tool_permission_allowed(self) -> None:
        """Test permission check for allowed tool."""
        gateway = MCPGateway()
        await gateway.check_tool_permission("research_topic")  # Should not raise

    @pytest.mark.asyncio
    async def test_check_tool_permission_blocked(self) -> None:
        """Test permission check for blocked tool."""
        config = MCPGatewayConfig(blocked_tools=["blocked_tool"])
        gateway = MCPGateway(config)

        with pytest.raises(PermissionError):
            await gateway.check_tool_permission("blocked_tool")

    @pytest.mark.asyncio
    async def test_check_tool_permission_rate_limited(self) -> None:
        """Test that rate limiting blocks requests."""
        config = MCPGatewayConfig(rate_limit_per_minute=2)
        gateway = MCPGateway(config)

        await gateway.check_tool_permission("tool1")
        await gateway.check_tool_permission("tool2")

        with pytest.raises(PermissionError) as exc:
            await gateway.check_tool_permission("tool3")
        assert "Rate limit exceeded" in str(exc.value)

    def test_create_request(self) -> None:
        """Test creating an MCP request."""
        gateway = MCPGateway()
        request = gateway.create_request(
            tool_name="research_topic",
            arguments={"query": "test"},
        )

        assert request.tool_name == "research_topic"
        assert request.client.client_id == "anonymous"

    def test_create_request_with_auth(self) -> None:
        """Test creating request with authenticated token."""
        gateway = MCPGateway()
        client_info = MCPClientInfo(client_id="client1", user_id="user1")
        gateway.register_token("my-token", client_info)

        request = gateway.create_request(
            tool_name="research_topic",
            arguments={},
            token="my-token",
        )

        assert request.client.client_id == "client1"
        assert request.client.user_id == "user1"

    def test_audit_logging(self) -> None:
        """Test that requests are logged."""
        config = MCPGatewayConfig(audit_enabled=True)
        gateway = MCPGateway(config)

        request = gateway.create_request("tool1", {})
        gateway.complete_request(request.request_id, success=True, duration_ms=100)

        entries = gateway.get_audit_log()
        assert len(entries) == 2  # Request + Response

    def test_get_stats(self) -> None:
        """Test getting gateway statistics."""
        gateway = MCPGateway()

        gateway.create_request("tool1", {})
        gateway.create_request("tool2", {})

        stats = gateway.get_stats()
        assert stats["total_requests"] == 2
        assert "tool1" in stats["tools_used"]
        assert "tool2" in stats["tools_used"]


class TestMCPGatewayGlobal:
    """Tests for global gateway functions."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_mcp_gateway()

    def test_get_mcp_gateway_singleton(self) -> None:
        """Test that get_mcp_gateway returns singleton."""
        gateway1 = get_mcp_gateway()
        gateway2 = get_mcp_gateway()
        assert gateway1 is gateway2

    def test_reset_mcp_gateway(self) -> None:
        """Test resetting the gateway."""
        gateway1 = get_mcp_gateway()
        reset_mcp_gateway()
        gateway2 = get_mcp_gateway()
        assert gateway1 is not gateway2

    def test_check_mcp_permission_function(self) -> None:
        """Test convenience permission check function."""
        reset_mcp_gateway()
        assert check_mcp_permission("research_topic") is True


# ==================== ServiceNow Tests ====================


class TestServiceNowConfig:
    """Tests for ServiceNowConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ServiceNowConfig()
        assert config.mode == "simulation"
        assert config.timeout == 30
        assert config.is_configured is False

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        with patch.dict("os.environ", {
            "SERVICENOW_INSTANCE": "dev12345",
            "SERVICENOW_USERNAME": "admin",
            "SERVICENOW_PASSWORD": "password",
            "SERVICENOW_MODE": "live",
        }):
            config = ServiceNowConfig.from_env()
            assert config.instance == "dev12345"
            assert config.username == "admin"
            assert config.mode == "live"
            assert config.is_configured is True

    def test_base_url(self) -> None:
        """Test base URL generation."""
        config = ServiceNowConfig(instance="dev12345")
        assert config.base_url == "https://dev12345.service-now.com"


class TestServiceNowSimulator:
    """Tests for ServiceNowSimulator."""

    @pytest.mark.asyncio
    async def test_create_incident(self) -> None:
        """Test creating a simulated incident."""
        simulator = ServiceNowSimulator()

        result = await simulator.create_incident(
            short_description="Test incident",
            description="This is a test",
            priority="2",
        )

        incident = result["result"]
        assert incident["number"].startswith("INC")
        assert incident["short_description"] == "Test incident"
        assert incident["priority"] == "2"
        assert incident["state"] == "1"  # New

    @pytest.mark.asyncio
    async def test_get_incident(self) -> None:
        """Test getting a simulated incident."""
        simulator = ServiceNowSimulator()

        # Create first
        create_result = await simulator.create_incident(
            short_description="Test",
            description="Description",
        )
        incident_number = create_result["result"]["number"]

        # Get it
        result = await simulator.get_incident(incident_number)
        assert result["result"][0]["number"] == incident_number

    @pytest.mark.asyncio
    async def test_get_nonexistent_incident(self) -> None:
        """Test getting a non-existent incident."""
        simulator = ServiceNowSimulator()
        result = await simulator.get_incident("INC9999999")
        assert result["result"] == []

    @pytest.mark.asyncio
    async def test_search_incidents(self) -> None:
        """Test searching simulated incidents."""
        simulator = ServiceNowSimulator()

        # Create some incidents
        await simulator.create_incident("Password reset needed", "User forgot password")
        await simulator.create_incident("Network issue", "Cannot connect to VPN")
        await simulator.create_incident("Password locked", "Account locked out")

        # Search
        result = await simulator.search_incidents("Password")
        assert len(result["result"]) == 2

    @pytest.mark.asyncio
    async def test_update_incident(self) -> None:
        """Test updating a simulated incident."""
        simulator = ServiceNowSimulator()

        # Create
        create_result = await simulator.create_incident("Test", "Description")
        incident_number = create_result["result"]["number"]

        # Update
        result = await simulator.update_incident(
            incident_number,
            {"state": "2", "work_notes": "Working on it"},
        )

        assert result["result"]["state"] == "2"
        assert result["result"]["work_notes"] == "Working on it"

    @pytest.mark.asyncio
    async def test_query_cmdb(self) -> None:
        """Test querying simulated CMDB."""
        simulator = ServiceNowSimulator()

        # Query servers
        result = await simulator.query_cmdb("cmdb_ci_server", "web")
        assert len(result["result"]) >= 1
        assert "web-server" in result["result"][0]["name"]


class TestServiceNowClient:
    """Tests for ServiceNowClient."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_servicenow_client()

    @pytest.mark.asyncio
    async def test_simulation_mode_create(self) -> None:
        """Test creating incident in simulation mode."""
        config = ServiceNowConfig(mode="simulation")
        client = ServiceNowClient(config)

        result = await client.create_incident(
            short_description="Test incident",
            description="This is a test",
        )

        assert "result" in result
        assert result["result"]["number"].startswith("INC")

    @pytest.mark.asyncio
    async def test_simulation_mode_get(self) -> None:
        """Test getting incident in simulation mode."""
        config = ServiceNowConfig(mode="simulation")
        client = ServiceNowClient(config)

        # Create first
        create_result = await client.create_incident("Test", "Description")
        incident_number = create_result["result"]["number"]

        # Get it
        incident = await client.get_incident(incident_number)
        assert incident["number"] == incident_number

    @pytest.mark.asyncio
    async def test_simulation_mode_search(self) -> None:
        """Test searching incidents in simulation mode."""
        config = ServiceNowConfig(mode="simulation")
        client = ServiceNowClient(config)

        await client.create_incident("Email issue", "Cannot send emails")
        await client.create_incident("Printer issue", "Printer not working")

        results = await client.search_incidents("Email")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_simulation_mode_update(self) -> None:
        """Test updating incident in simulation mode."""
        config = ServiceNowConfig(mode="simulation")
        client = ServiceNowClient(config)

        create_result = await client.create_incident("Test", "Description")
        incident_number = create_result["result"]["number"]

        updated = await client.update_incident(
            incident_number,
            {"state": "6"},  # Resolved
        )

        assert updated["state"] == "6"

    @pytest.mark.asyncio
    async def test_simulation_mode_cmdb(self) -> None:
        """Test CMDB query in simulation mode."""
        config = ServiceNowConfig(mode="simulation")
        client = ServiceNowClient(config)

        results = await client.query_cmdb("cmdb_ci_server", "server")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_live_mode_not_configured(self) -> None:
        """Test live mode without configuration."""
        config = ServiceNowConfig(mode="live")  # No credentials
        client = ServiceNowClient(config)

        result = await client.create_incident("Test", "Description")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_change_request_simulation(self) -> None:
        """Test creating change request in simulation mode."""
        config = ServiceNowConfig(mode="simulation")
        client = ServiceNowClient(config)

        result = await client.create_change_request(
            short_description="Server upgrade",
            description="Upgrading production servers",
            type="normal",
        )

        assert "result" in result
        assert result["result"]["number"].startswith("CHG")


class TestServiceNowClientGlobal:
    """Tests for global client functions."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_servicenow_client()

    def test_get_servicenow_client_singleton(self) -> None:
        """Test that get_servicenow_client returns singleton."""
        client1 = get_servicenow_client()
        client2 = get_servicenow_client()
        assert client1 is client2

    def test_reset_servicenow_client(self) -> None:
        """Test resetting the client."""
        client1 = get_servicenow_client()
        reset_servicenow_client()
        client2 = get_servicenow_client()
        assert client1 is not client2


# ==================== MCP Server Tests ====================


class TestMCPServer:
    """Tests for MCP server tools."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_mcp_gateway()
        reset_servicenow_client()

    @pytest.mark.asyncio
    async def test_server_import(self) -> None:
        """Test that server can be imported."""
        from app.mcp.server import get_mcp_server, mcp

        server = get_mcp_server()
        assert server is mcp

    @pytest.mark.asyncio
    async def test_extract_response_helper(self) -> None:
        """Test response extraction helper."""
        from app.mcp.server import _extract_response

        # Test string input
        assert _extract_response("Hello") == "Hello"

        # Test dict with output
        assert _extract_response({"output": "Result"}) == "Result"

        # Test dict with response
        assert _extract_response({"response": "Answer"}) == "Answer"

        # Test dict with messages (LangGraph format)
        class MockMessage:
            content = "Message content"

        result = _extract_response({"messages": [MockMessage()]})
        assert result == "Message content"


# ==================== Integration Tests ====================


class TestMCPIntegration:
    """Integration tests for MCP module."""

    def setup_method(self) -> None:
        """Reset all global state."""
        reset_mcp_gateway()
        reset_servicenow_client()

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Test a complete MCP workflow."""
        # 1. Configure gateway
        gateway = get_mcp_gateway()
        client_info = MCPClientInfo(client_id="test-client", user_id="user1")
        gateway.register_token("test-token", client_info)

        # 2. Create request
        request = gateway.create_request(
            tool_name="create_incident",
            arguments={"short_description": "Test", "description": "Full workflow test"},
            token="test-token",
        )

        # 3. Use ServiceNow client
        sn_client = get_servicenow_client()
        incident = await sn_client.create_incident(
            short_description=request.arguments["short_description"],
            description=request.arguments["description"],
        )

        # 4. Complete request
        gateway.complete_request(request.request_id, success=True, duration_ms=50)

        # Verify
        assert incident["result"]["number"].startswith("INC")
        assert gateway.get_stats()["total_requests"] == 1
        assert gateway.get_stats()["successful_responses"] == 1

    @pytest.mark.asyncio
    async def test_rate_limit_across_tools(self) -> None:
        """Test rate limiting works across different tools."""
        config = MCPGatewayConfig(rate_limit_per_minute=3)
        gateway = MCPGateway(config)

        # Should allow first 3
        await gateway.check_tool_permission("tool1")
        await gateway.check_tool_permission("tool2")
        await gateway.check_tool_permission("tool3")

        # Should block 4th
        with pytest.raises(PermissionError):
            await gateway.check_tool_permission("tool4")
