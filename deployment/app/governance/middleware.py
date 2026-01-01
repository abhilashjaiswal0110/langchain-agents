"""FastAPI middleware for governance integration.

Provides:
- Request authentication and RBAC
- Rate limiting middleware
- Audit logging middleware
- Governance context injection
- Error handling for governance exceptions
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.governance.audit_logger import (
    AuditAction,
    AuditLevel,
    get_audit_logger,
)
from app.governance.rate_limiter import (
    RateLimitExceededError,
    RateLimitResult,
    get_rate_limiter,
)
from app.governance.rbac import (
    Permission,
    PermissionDeniedError,
    Role,
    UserContext,
    get_rbac_manager,
)


@dataclass
class GovernanceContext:
    """Context for governance information in requests.

    Attributes:
        user_context: RBAC user context
        rate_limit_result: Result of rate limit check
        request_id: Unique request identifier
        start_time: Request start timestamp
    """

    user_context: UserContext
    rate_limit_result: RateLimitResult | None = None
    request_id: str = ""
    start_time: float = field(default_factory=time.time)


# Request state key for governance context
GOVERNANCE_CONTEXT_KEY = "governance_context"


def get_governance_context(request: Request) -> GovernanceContext | None:
    """Get governance context from request.

    Args:
        request: FastAPI request.

    Returns:
        Governance context or None if not set.
    """
    return getattr(request.state, GOVERNANCE_CONTEXT_KEY, None)


def get_user_context(request: Request) -> UserContext | None:
    """Get user context from request.

    Args:
        request: FastAPI request.

    Returns:
        User context or None if not set.
    """
    ctx = get_governance_context(request)
    return ctx.user_context if ctx else None


class RBACMiddleware(BaseHTTPMiddleware):
    """Middleware for role-based access control.

    Extracts API key from headers and creates user context.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key_header: str = "X-API-Key",
        user_id_header: str = "X-User-ID",
    ) -> None:
        """Initialize RBAC middleware.

        Args:
            app: ASGI application.
            api_key_header: Header name for API key.
            user_id_header: Header name for user ID override.
        """
        super().__init__(app)
        self.api_key_header = api_key_header
        self.user_id_header = user_id_header

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with RBAC.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler.
        """
        # Extract API key
        api_key = request.headers.get(self.api_key_header)
        user_id = request.headers.get(self.user_id_header)

        # Get user context from RBAC manager
        rbac_manager = get_rbac_manager()
        user_context = rbac_manager.get_user_context(api_key, user_id)

        # Create governance context
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        gov_context = GovernanceContext(
            user_context=user_context,
            request_id=request_id,
        )

        # Store in request state
        request.state.governance_context = gov_context

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] | None = None,
    ) -> None:
        """Initialize rate limit middleware.

        Args:
            app: ASGI application.
            exclude_paths: Paths to exclude from rate limiting.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/ready", "/docs", "/openapi.json"]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limiting.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler.
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get governance context
        gov_context = get_governance_context(request)
        if not gov_context:
            return await call_next(request)

        # Check rate limit
        rate_limiter = get_rate_limiter()
        user_id = gov_context.user_context.user_id

        try:
            result = await rate_limiter.check_user(user_id)
            gov_context.rate_limit_result = result

            if not result.allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": result.retry_after,
                        "limit": result.limit,
                        "remaining": result.remaining,
                    },
                    headers={
                        "Retry-After": str(int(result.retry_after or 60)),
                        "X-RateLimit-Limit": str(result.limit),
                        "X-RateLimit-Remaining": str(result.remaining),
                        "X-RateLimit-Reset": str(int(result.reset_at)),
                    },
                )

            # Add rate limit headers to response
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))

            return response

        except Exception as e:
            # Log but don't block on rate limiter errors
            import logging
            logging.getLogger(__name__).warning(f"Rate limiter error: {e}")
            return await call_next(request)


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] | None = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ) -> None:
        """Initialize audit middleware.

        Args:
            app: ASGI application.
            exclude_paths: Paths to exclude from audit logging.
            log_request_body: Whether to log request bodies.
            log_response_body: Whether to log response bodies.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/ready"]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with audit logging.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler.
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get governance context
        gov_context = get_governance_context(request)
        user_id = gov_context.user_context.user_id if gov_context else "anonymous"
        start_time = time.time()

        # Capture request body if needed
        request_body = None
        if self.log_request_body:
            try:
                request_body = await request.body()
                # Reset body for downstream handlers
                request._body = request_body
            except Exception:
                pass

        # Process request
        try:
            response = await call_next(request)
            status = "success" if response.status_code < 400 else "failure"

            # Log the request
            duration_ms = int((time.time() - start_time) * 1000)
            audit_logger = get_audit_logger()

            # Determine action from path/method
            action = self._determine_action(request.method, request.url.path)

            await audit_logger.log_async(
                action=action,
                user_id=user_id,
                level=AuditLevel.INFO if status == "success" else AuditLevel.WARNING,
                agent_type=self._extract_agent_type(request.url.path),
                duration_ms=duration_ms,
                status=status,  # type: ignore[arg-type]
                metadata={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "request_id": gov_context.request_id if gov_context else None,
                },
            )

            return response

        except Exception as e:
            # Log failed request
            duration_ms = int((time.time() - start_time) * 1000)
            audit_logger = get_audit_logger()

            await audit_logger.log_async(
                action=AuditAction.SYSTEM_ERROR,
                user_id=user_id,
                level=AuditLevel.ERROR,
                duration_ms=duration_ms,
                status="failure",
                error_message=str(e),
                metadata={
                    "method": request.method,
                    "path": request.url.path,
                    "exception_type": type(e).__name__,
                },
            )

            raise

    def _determine_action(self, method: str, path: str) -> AuditAction:
        """Determine audit action from request.

        Args:
            method: HTTP method.
            path: Request path.

        Returns:
            Audit action.
        """
        # Agent invocations
        if "/invoke" in path or "/chat" in path:
            return AuditAction.AGENT_INVOKE

        # Conversation endpoints
        if "/conversation" in path:
            if method == "POST":
                return AuditAction.SESSION_START
            return AuditAction.AGENT_INVOKE

        # API calls
        if "/api/" in path:
            return AuditAction.AGENT_INVOKE

        # Default
        return AuditAction.SYSTEM_ERROR

    def _extract_agent_type(self, path: str) -> str | None:
        """Extract agent type from path.

        Args:
            path: Request path.

        Returns:
            Agent type or None.
        """
        # Common patterns
        agent_patterns = [
            "/chat",
            "/rag",
            "/agent",
            "/langgraph",
            "/research",
            "/servicenow",
            "/helpdesk",
            "/onboarding",
            "/compliance",
            "/document",
            "/enterprise",
        ]

        for pattern in agent_patterns:
            if pattern in path.lower():
                return pattern.strip("/")

        return None


class GovernanceExceptionMiddleware(BaseHTTPMiddleware):
    """Middleware for handling governance exceptions."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and handle governance exceptions.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler or error response.
        """
        try:
            return await call_next(request)

        except PermissionDeniedError as e:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Permission denied",
                    "message": str(e),
                    "user_id": e.user_id,
                    "permission": e.permission.value if e.permission else None,
                },
            )

        except RateLimitExceededError as e:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": str(e),
                    "retry_after": e.retry_after,
                },
                headers={
                    "Retry-After": str(int(e.retry_after or 60)),
                },
            )


def setup_governance_middleware(
    app: FastAPI,
    enable_rbac: bool = True,
    enable_rate_limit: bool = True,
    enable_audit: bool = True,
    api_key_header: str = "X-API-Key",
    exclude_paths: list[str] | None = None,
) -> None:
    """Set up governance middleware stack on FastAPI app.

    Args:
        app: FastAPI application.
        enable_rbac: Whether to enable RBAC middleware.
        enable_rate_limit: Whether to enable rate limiting.
        enable_audit: Whether to enable audit logging.
        api_key_header: Header name for API key.
        exclude_paths: Paths to exclude from governance.

    Note:
        Middleware is applied in reverse order of addition.
        Order will be: Exception -> Audit -> RateLimit -> RBAC
    """
    default_exclude = ["/health", "/ready", "/docs", "/openapi.json", "/redoc"]
    exclude = exclude_paths or default_exclude

    # Add exception handler first (will be outermost)
    app.add_middleware(GovernanceExceptionMiddleware)

    # Add audit logging
    if enable_audit:
        app.add_middleware(AuditMiddleware, exclude_paths=exclude)

    # Add rate limiting
    if enable_rate_limit:
        app.add_middleware(RateLimitMiddleware, exclude_paths=exclude)

    # Add RBAC (will be innermost, runs first)
    if enable_rbac:
        app.add_middleware(RBACMiddleware, api_key_header=api_key_header)


# Dependency injection helpers for FastAPI routes


async def require_permission(
    request: Request,
    permission: Permission,
) -> UserContext:
    """Dependency to require a specific permission.

    Args:
        request: FastAPI request.
        permission: Required permission.

    Returns:
        User context if authorized.

    Raises:
        PermissionDeniedError: If permission is denied.
    """
    gov_context = get_governance_context(request)
    if not gov_context:
        raise PermissionDeniedError(
            "No governance context",
            user_id="unknown",
            permission=permission,
        )

    user_context = gov_context.user_context
    if not user_context.has_permission(permission):
        raise PermissionDeniedError(
            f"Permission {permission.value} required",
            user_id=user_context.user_id,
            permission=permission,
        )

    return user_context


async def require_role(
    request: Request,
    role: Role,
) -> UserContext:
    """Dependency to require a specific role.

    Args:
        request: FastAPI request.
        role: Required role.

    Returns:
        User context if authorized.

    Raises:
        PermissionDeniedError: If role requirement not met.
    """
    gov_context = get_governance_context(request)
    if not gov_context:
        raise PermissionDeniedError(
            "No governance context",
            user_id="unknown",
        )

    user_context = gov_context.user_context

    # Admin always passes
    if user_context.role == Role.ADMIN:
        return user_context

    # Check role hierarchy
    role_hierarchy = [Role.VIEWER, Role.USER, Role.OPERATOR, Role.ADMIN]
    user_level = role_hierarchy.index(user_context.role) if user_context.role in role_hierarchy else -1
    required_level = role_hierarchy.index(role) if role in role_hierarchy else -1

    if user_level < required_level:
        raise PermissionDeniedError(
            f"Role {role.value} or higher required",
            user_id=user_context.user_id,
        )

    return user_context


def create_permission_dependency(permission: Permission) -> Callable:
    """Create a dependency function for a specific permission.

    Args:
        permission: Permission to require.

    Returns:
        Dependency function.
    """
    async def dependency(request: Request) -> UserContext:
        return await require_permission(request, permission)

    return dependency


def create_role_dependency(role: Role) -> Callable:
    """Create a dependency function for a specific role.

    Args:
        role: Role to require.

    Returns:
        Dependency function.
    """
    async def dependency(request: Request) -> UserContext:
        return await require_role(request, role)

    return dependency


# Pre-built dependencies for common use cases
require_admin = create_role_dependency(Role.ADMIN)
require_operator = create_role_dependency(Role.OPERATOR)
require_user = create_role_dependency(Role.USER)
require_agent_invoke = create_permission_dependency(Permission.AGENT_INVOKE)
require_audit_read = create_permission_dependency(Permission.AUDIT_READ)
