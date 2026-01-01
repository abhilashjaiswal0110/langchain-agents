"""MCP (Model Context Protocol) integration for LangChain agents.

This module provides MCP server functionality to expose enterprise agents
as tools that can be used by MCP-compatible clients.

Components:
- server.py: FastMCP server with tool definitions
- gateway.py: Access control and rate limiting
- servicenow_client.py: Real ServiceNow API client

Usage:
    # Run standalone MCP server
    python -m app.mcp.server

    # Or import components
    from app.mcp import (
        get_mcp_server,
        get_mcp_gateway,
        get_servicenow_client,
    )

    # Get server instance
    mcp = get_mcp_server()

    # Configure gateway
    gateway = get_mcp_gateway()
    gateway.register_token("my-token", MCPClientInfo(client_id="client1"))

    # Use ServiceNow client
    client = get_servicenow_client()
    incident = await client.create_incident("Issue", "Description")
"""

# Server exports
from app.mcp.server import (
    get_mcp_server,
    mcp,
    run_mcp_server,
)

# Gateway exports
from app.mcp.gateway import (
    MCPClientInfo,
    MCPGateway,
    MCPGatewayConfig,
    MCPRequest,
    check_mcp_permission,
    get_mcp_gateway,
    reset_mcp_gateway,
)

# ServiceNow exports
from app.mcp.servicenow_client import (
    ServiceNowClient,
    ServiceNowConfig,
    create_incident,
    get_incident,
    get_servicenow_client,
    reset_servicenow_client,
    search_incidents,
)

__all__ = [
    # Server
    "mcp",
    "get_mcp_server",
    "run_mcp_server",
    # Gateway
    "MCPGateway",
    "MCPGatewayConfig",
    "MCPClientInfo",
    "MCPRequest",
    "get_mcp_gateway",
    "reset_mcp_gateway",
    "check_mcp_permission",
    # ServiceNow
    "ServiceNowClient",
    "ServiceNowConfig",
    "get_servicenow_client",
    "reset_servicenow_client",
    "create_incident",
    "get_incident",
    "search_incidents",
]
