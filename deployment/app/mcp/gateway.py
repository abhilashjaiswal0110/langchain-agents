"""MCP Gateway for access control and request management.

Provides:
- Authentication for MCP connections
- Rate limiting per MCP client
- Request/response audit logging
- Tool-level permission checking
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import Context


@dataclass
class MCPGatewayConfig:
    """Configuration for MCP gateway.

    Attributes:
        enabled: Whether gateway controls are enabled
        require_auth: Whether to require authentication
        auth_header: Header name for authentication token
        rate_limit_enabled: Whether rate limiting is enabled
        rate_limit_per_minute: Requests per minute per client
        audit_enabled: Whether audit logging is enabled
        allowed_tools: List of allowed tools (empty = all)
        blocked_tools: List of blocked tools
    """

    enabled: bool = True
    require_auth: bool = False
    auth_header: str = "X-MCP-Token"
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    audit_enabled: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "MCPGatewayConfig":
        """Create config from environment variables."""
        allowed = os.getenv("MCP_ALLOWED_TOOLS", "")
        blocked = os.getenv("MCP_BLOCKED_TOOLS", "")

        return cls(
            enabled=os.getenv("MCP_GATEWAY_ENABLED", "true").lower() == "true",
            require_auth=os.getenv("MCP_REQUIRE_AUTH", "false").lower() == "true",
            auth_header=os.getenv("MCP_AUTH_HEADER", "X-MCP-Token"),
            rate_limit_enabled=os.getenv("MCP_RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_per_minute=int(os.getenv("MCP_RATE_LIMIT_PER_MINUTE", "60")),
            audit_enabled=os.getenv("MCP_AUDIT_ENABLED", "true").lower() == "true",
            allowed_tools=[t.strip() for t in allowed.split(",") if t.strip()],
            blocked_tools=[t.strip() for t in blocked.split(",") if t.strip()],
        )


@dataclass
class MCPClientInfo:
    """Information about an MCP client.

    Attributes:
        client_id: Unique client identifier
        user_id: Associated user ID (if authenticated)
        role: Client role for permissions
        metadata: Additional client metadata
    """

    client_id: str
    user_id: str | None = None
    role: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPRequest:
    """An MCP tool request.

    Attributes:
        tool_name: Name of the tool being called
        arguments: Tool arguments
        client: Client information
        timestamp: Request timestamp
        request_id: Unique request identifier
    """

    tool_name: str
    arguments: dict[str, Any]
    client: MCPClientInfo
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""

    def __post_init__(self) -> None:
        """Generate request ID if not provided."""
        if not self.request_id:
            import uuid
            self.request_id = str(uuid.uuid4())


class MCPRateLimiter:
    """Simple in-memory rate limiter for MCP clients."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per client.
        """
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}

    def check(self, client_id: str) -> bool:
        """Check if client can make a request.

        Args:
            client_id: Client identifier.

        Returns:
            True if request is allowed.
        """
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Get client's requests
        if client_id not in self._requests:
            self._requests[client_id] = []

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        # Check limit
        if len(self._requests[client_id]) >= self.requests_per_minute:
            return False

        # Record request
        self._requests[client_id].append(now)
        return True

    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client.

        Args:
            client_id: Client identifier.
        """
        if client_id in self._requests:
            del self._requests[client_id]


class MCPAuditLogger:
    """Simple audit logger for MCP requests."""

    def __init__(self) -> None:
        """Initialize audit logger."""
        self._entries: list[dict[str, Any]] = []

    def log_request(self, request: MCPRequest) -> None:
        """Log an MCP request.

        Args:
            request: The MCP request to log.
        """
        self._entries.append({
            "type": "request",
            "request_id": request.request_id,
            "tool_name": request.tool_name,
            "client_id": request.client.client_id,
            "user_id": request.client.user_id,
            "timestamp": request.timestamp,
            "arguments_keys": list(request.arguments.keys()),
        })

    def log_response(
        self,
        request_id: str,
        success: bool,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Log an MCP response.

        Args:
            request_id: The request ID.
            success: Whether the request succeeded.
            duration_ms: Request duration in milliseconds.
            error: Error message if failed.
        """
        self._entries.append({
            "type": "response",
            "request_id": request_id,
            "success": success,
            "duration_ms": duration_ms,
            "error": error,
            "timestamp": time.time(),
        })

    def get_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of audit entries.
        """
        return self._entries[-limit:]


class MCPGateway:
    """Gateway for MCP access control and monitoring.

    Provides:
    - Authentication validation
    - Rate limiting
    - Tool permission checking
    - Audit logging
    """

    def __init__(self, config: MCPGatewayConfig | None = None) -> None:
        """Initialize MCP gateway.

        Args:
            config: Gateway configuration.
        """
        self.config = config or MCPGatewayConfig.from_env()
        self._rate_limiter = MCPRateLimiter(self.config.rate_limit_per_minute)
        self._audit_logger = MCPAuditLogger()
        self._auth_tokens: dict[str, MCPClientInfo] = {}

    def register_token(self, token: str, client_info: MCPClientInfo) -> None:
        """Register an authentication token.

        Args:
            token: Authentication token.
            client_info: Associated client information.
        """
        self._auth_tokens[token] = client_info

    def validate_token(self, token: str | None) -> MCPClientInfo | None:
        """Validate an authentication token.

        Args:
            token: Token to validate.

        Returns:
            Client info if valid, None otherwise.
        """
        if not token:
            return None
        return self._auth_tokens.get(token)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed.

        Args:
            tool_name: Name of the tool.

        Returns:
            True if tool is allowed.
        """
        # Check blocked list first
        if tool_name in self.config.blocked_tools:
            return False

        # If allowed list is specified, tool must be in it
        if self.config.allowed_tools:
            return tool_name in self.config.allowed_tools

        return True

    async def check_tool_permission(
        self,
        tool_name: str,
        ctx: Context | None = None,
    ) -> None:
        """Check if a tool call is permitted.

        Args:
            tool_name: Name of the tool being called.
            ctx: MCP context.

        Raises:
            PermissionError: If tool call is not permitted.
        """
        if not self.config.enabled:
            return

        # Check if tool is allowed
        if not self.is_tool_allowed(tool_name):
            raise PermissionError(f"Tool '{tool_name}' is not allowed")

        # Get client ID from context or use default
        client_id = "default"
        if ctx and hasattr(ctx, "client_id"):
            client_id = ctx.client_id

        # Check rate limit
        if self.config.rate_limit_enabled:
            if not self._rate_limiter.check(client_id):
                raise PermissionError(
                    f"Rate limit exceeded for client {client_id}. "
                    f"Max {self.config.rate_limit_per_minute} requests per minute."
                )

    def create_request(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        token: str | None = None,
    ) -> MCPRequest:
        """Create an MCP request object.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            token: Optional auth token.

        Returns:
            MCPRequest object.
        """
        client_info = self.validate_token(token)
        if not client_info:
            client_info = MCPClientInfo(client_id="anonymous")

        request = MCPRequest(
            tool_name=tool_name,
            arguments=arguments,
            client=client_info,
        )

        if self.config.audit_enabled:
            self._audit_logger.log_request(request)

        return request

    def complete_request(
        self,
        request_id: str,
        success: bool,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Complete an MCP request (log response).

        Args:
            request_id: The request ID.
            success: Whether the request succeeded.
            duration_ms: Request duration.
            error: Error message if failed.
        """
        if self.config.audit_enabled:
            self._audit_logger.log_response(request_id, success, duration_ms, error)

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit log entries.

        Args:
            limit: Maximum entries.

        Returns:
            Audit log entries.
        """
        return self._audit_logger.get_entries(limit)

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics.

        Returns:
            Dictionary of statistics.
        """
        entries = self._audit_logger.get_entries(1000)
        requests = [e for e in entries if e["type"] == "request"]
        responses = [e for e in entries if e["type"] == "response"]

        successful = len([r for r in responses if r.get("success")])
        failed = len([r for r in responses if not r.get("success")])

        return {
            "total_requests": len(requests),
            "successful_responses": successful,
            "failed_responses": failed,
            "success_rate": successful / len(responses) if responses else 0,
            "unique_clients": len(set(r.get("client_id") for r in requests)),
            "tools_used": list(set(r.get("tool_name") for r in requests)),
        }


# Global gateway instance
_mcp_gateway: MCPGateway | None = None


def get_mcp_gateway() -> MCPGateway:
    """Get or create the global MCP gateway.

    Returns:
        MCPGateway instance.
    """
    global _mcp_gateway
    if _mcp_gateway is None:
        _mcp_gateway = MCPGateway()
    return _mcp_gateway


def reset_mcp_gateway() -> None:
    """Reset the global MCP gateway."""
    global _mcp_gateway
    _mcp_gateway = None


# Convenience functions


def check_mcp_permission(tool_name: str, token: str | None = None) -> bool:
    """Check if an MCP tool call is permitted.

    Args:
        tool_name: Name of the tool.
        token: Optional auth token.

    Returns:
        True if permitted.
    """
    gateway = get_mcp_gateway()

    if not gateway.is_tool_allowed(tool_name):
        return False

    if gateway.config.require_auth and not gateway.validate_token(token):
        return False

    return True
