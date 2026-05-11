"""FastAPI middleware for governance integration.

Provides:
- Request authentication and RBAC
- Rate limiting middleware
- Audit logging middleware
- PII detection and masking middleware
- Anomaly detection middleware
- Governance context injection
- Error handling for governance exceptions
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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

_inj_logger = logging.getLogger(__name__)


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


class PIIMiddleware(BaseHTTPMiddleware):
    """Middleware for PII detection in requests."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] | None = None,
        block_on_pii: bool = False,
        mask_response: bool = False,
    ) -> None:
        """Initialize PII middleware.

        Args:
            app: ASGI application.
            exclude_paths: Paths to exclude from PII detection.
            block_on_pii: Whether to block requests with critical PII.
            mask_response: Whether to mask PII in responses.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/ready", "/docs"]
        self.block_on_pii = block_on_pii
        self.mask_response = mask_response

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with PII detection.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler.
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Try to import PII detector (may not be initialized)
        try:
            from app.governance.pii_detector import (
                PIIBlockedError,
                PIISeverity,
                get_pii_detector,
            )

            detector = get_pii_detector()

            # Check request body for PII if blocking is enabled
            if self.block_on_pii:
                try:
                    body = await request.body()
                    if body:
                        body_text = body.decode("utf-8", errors="ignore")
                        result = detector.analyze(body_text)

                        # Block on critical PII
                        if result.severity == PIISeverity.CRITICAL:
                            critical_types = {m.pii_type for m in result.matches if m.severity == PIISeverity.CRITICAL}
                            raise PIIBlockedError(
                                "Request contains sensitive PII",
                                pii_types=critical_types,
                                severity=PIISeverity.CRITICAL,
                            )

                        # Store for later use
                        request.state._body = body
                except UnicodeDecodeError:
                    pass

            return await call_next(request)

        except Exception as e:
            # Re-raise PII blocked errors
            if "PIIBlockedError" in type(e).__name__:
                raise
            # Log but don't block on PII detector errors
            import logging

            logging.getLogger(__name__).warning(f"PII middleware error: {e}")
            return await call_next(request)


class InjectionMiddleware(BaseHTTPMiddleware):
    """Middleware for prompt injection and jailbreak detection.

    Inspects incoming JSON request bodies for fields named ``message``,
    ``input``, or ``query`` and applies :class:`InjectionDetector` to each
    value found.

    Behaviour by score:
    - score >= 0.9 → block with HTTP 400
    - score >= 0.85 → log a warning and pass through
    - score < 0.85 → pass through silently

    Only endpoints that accept user message input are scanned (conversation,
    enterprise agents, deep agents, sales/recruitment agents, software-dev
    agent).  Health, docs, analytics, and other non-input endpoints are
    excluded.
    """

    # Paths that are NOT checked (non-input endpoints)
    _DEFAULT_EXCLUDE: list[str] = [
        "/health",
        "/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/metrics",
        "/analytics",
        "/audit",
        "/favicon.ico",
    ]

    # JSON body fields that carry user-supplied free text
    _INPUT_FIELDS: tuple[str, ...] = ("message", "input", "query")

    # Only scan paths that match one of these prefixes
    _SCAN_PREFIXES: tuple[str, ...] = (
        "/api/conversation",
        "/api/enterprise",
        "/api/deepagent",
        "/api/sales-agent",
        "/api/recruitment-agent",
        "/api/software-dev-agent",
        "/chat",
        "/rag",
        "/agent",
    )

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] | None = None,
        block_score: float = 0.9,
        warn_score: float = 0.85,
    ) -> None:
        """Initialize injection middleware.

        Args:
            app: ASGI application.
            exclude_paths: Paths to skip entirely (in addition to the built-in
                list).
            block_score: Minimum score at which requests are blocked (default
                0.9).
            warn_score: Minimum score at which a warning is logged but the
                request is allowed through (default 0.85).
        """
        super().__init__(app)
        self.exclude_paths: list[str] = list(self._DEFAULT_EXCLUDE) + (exclude_paths or [])
        self.block_score = block_score
        self.warn_score = warn_score

    def _should_scan(self, path: str) -> bool:
        """Determine whether a request path should be scanned.

        Args:
            path: URL path of the request.

        Returns:
            ``True`` when the path should be scanned for injections.
        """
        if path in self.exclude_paths:
            return False
        return any(path.startswith(prefix) for prefix in self._SCAN_PREFIXES)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with injection detection.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            HTTP 400 response when a high-confidence injection is detected,
            otherwise the normal downstream response.
        """
        if not self._should_scan(request.url.path):
            return await call_next(request)

        # Only scan POST/PUT/PATCH requests that may carry a body
        if request.method not in {"POST", "PUT", "PATCH"}:
            return await call_next(request)

        try:
            from app.governance.injection_detector import get_injection_detector

            body_bytes = await request.body()
            # Reset the body so downstream handlers can read it again
            request._body = body_bytes  # type: ignore[attr-defined]

            if body_bytes:
                import json as _json

                try:
                    body_json = _json.loads(body_bytes.decode("utf-8", errors="ignore"))
                except (_json.JSONDecodeError, ValueError):
                    body_json = {}

                if isinstance(body_json, dict):
                    detector = get_injection_detector()
                    gov_context = get_governance_context(request)
                    user_id = gov_context.user_context.user_id if gov_context else "anonymous"

                    for field in self._INPUT_FIELDS:
                        value = body_json.get(field)
                        if not isinstance(value, str) or not value:
                            continue

                        result = detector.analyze(value)

                        if result.detected:
                            if result.score >= self.block_score:
                                # Log and block
                                audit_logger = get_audit_logger()
                                await audit_logger.log_async(
                                    action=AuditAction.PERMISSION_DENIED,
                                    user_id=user_id,
                                    level=AuditLevel.WARNING,
                                    status="failure",
                                    metadata={
                                        "path": request.url.path,
                                        "field": field,
                                        "pattern": result.matched_pattern,
                                        "score": result.score,
                                        "reason": "Prompt injection detected",
                                        "request_id": gov_context.request_id if gov_context else None,
                                    },
                                )
                                _inj_logger.warning(
                                    "Prompt injection blocked: user=%s path=%s field=%s score=%.2f pattern=%r",
                                    user_id,
                                    request.url.path,
                                    field,
                                    result.score,
                                    result.matched_pattern,
                                )
                                return JSONResponse(
                                    status_code=400,
                                    content={"detail": "Request blocked: potential prompt injection detected"},
                                )
                            elif result.score >= self.warn_score:
                                _inj_logger.warning(
                                    "Possible prompt injection (warn only): "
                                    "user=%s path=%s field=%s score=%.2f pattern=%r",
                                    user_id,
                                    request.url.path,
                                    field,
                                    result.score,
                                    result.matched_pattern,
                                )

        except Exception as exc:
            _inj_logger.warning("InjectionMiddleware error (non-blocking): %s", exc)

        return await call_next(request)


class AnomalyMiddleware(BaseHTTPMiddleware):
    """Middleware for anomaly detection."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list[str] | None = None,
        block_on_critical: bool = False,
    ) -> None:
        """Initialize anomaly middleware.

        Args:
            app: ASGI application.
            exclude_paths: Paths to exclude from anomaly detection.
            block_on_critical: Whether to block on critical anomalies.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/ready", "/docs"]
        self.block_on_critical = block_on_critical

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with anomaly detection.

        Args:
            request: Incoming request.
            call_next: Next middleware/handler.

        Returns:
            Response from handler.
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        start_time = time.time()

        # Get user context
        gov_context = get_governance_context(request)
        user_id = gov_context.user_context.user_id if gov_context else "anonymous"

        try:
            from app.governance.anomaly_detector import (
                AnomalyBlockedError,
                AnomalySeverity,
                get_anomaly_detector,
                record_event,
            )

            detector = get_anomaly_detector()

            # Check if user is blocked
            if self.block_on_critical and detector.is_blocked(user_id):
                raise AnomalyBlockedError(
                    f"User {user_id} is blocked due to anomalies",
                    user_id=user_id,
                    anomalies=detector.get_anomalies(user_id=user_id, limit=5),
                )

            # Process request
            response = await call_next(request)

            # Record event
            duration_ms = int((time.time() - start_time) * 1000)
            agent_type = self._extract_agent_type(request.url.path)

            # Get content length if available
            content_length = 0
            if hasattr(response, "headers"):
                content_length = int(response.headers.get("content-length", 0))

            anomalies = record_event(
                user_id=user_id,
                agent_type=agent_type or "unknown",
                event_type="request",
                success=response.status_code < 400,
                metadata={
                    "response_time_ms": duration_ms,
                    "status_code": response.status_code,
                    "path": request.url.path,
                    "method": request.method,
                    "output_length": content_length,
                },
            )

            # Block on critical anomalies (for future requests)
            if self.block_on_critical:
                for anomaly in anomalies:
                    if anomaly.severity == AnomalySeverity.CRITICAL:
                        import logging

                        logging.getLogger(__name__).warning(
                            f"Critical anomaly detected for user {user_id}: {anomaly.anomaly_type}"
                        )

            return response

        except Exception as e:
            # Re-raise anomaly blocked errors
            if "AnomalyBlockedError" in type(e).__name__:
                raise
            # Log but don't block on anomaly detector errors
            import logging

            logging.getLogger(__name__).warning(f"Anomaly middleware error: {e}")
            return await call_next(request)

    def _extract_agent_type(self, path: str) -> str | None:
        """Extract agent type from path.

        Args:
            path: Request path.

        Returns:
            Agent type or None.
        """
        agent_patterns = [
            "/chat",
            "/rag",
            "/agent",
            "/langgraph",
            "/research",
            "/servicenow",
            "/helpdesk",
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

        except Exception as e:
            # Handle PII and Anomaly blocked errors
            error_type = type(e).__name__

            if error_type == "PIIBlockedError":
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Request blocked",
                        "message": "Request contains sensitive information that cannot be processed",
                        "severity": str(getattr(e, "severity", "critical")),
                    },
                )

            if error_type == "AnomalyBlockedError":
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Access blocked",
                        "message": "Access has been blocked due to unusual activity",
                        "user_id": getattr(e, "user_id", "unknown"),
                    },
                )

            if error_type == "BudgetExceededError":
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "Budget exceeded",
                        "message": str(e),
                        "budget_type": getattr(e, "budget_type", "unknown"),
                        "current": getattr(e, "current", 0),
                        "limit": getattr(e, "limit", 0),
                    },
                )

            # Re-raise unhandled exceptions
            raise


def setup_governance_middleware(
    app: FastAPI,
    enable_rbac: bool = True,
    enable_rate_limit: bool = True,
    enable_audit: bool = True,
    enable_pii: bool = False,
    enable_anomaly: bool = False,
    enable_injection: bool = True,
    api_key_header: str = "X-API-Key",
    exclude_paths: list[str] | None = None,
    block_on_pii: bool = False,
    block_on_anomaly: bool = False,
) -> None:
    """Set up governance middleware stack on FastAPI app.

    Args:
        app: FastAPI application.
        enable_rbac: Whether to enable RBAC middleware.
        enable_rate_limit: Whether to enable rate limiting.
        enable_audit: Whether to enable audit logging.
        enable_pii: Whether to enable PII detection.
        enable_anomaly: Whether to enable anomaly detection.
        enable_injection: Whether to enable prompt injection detection.
        api_key_header: Header name for API key.
        exclude_paths: Paths to exclude from governance.
        block_on_pii: Whether to block requests with critical PII.
        block_on_anomaly: Whether to block on critical anomalies.

    Note:
        Middleware is applied in reverse order of addition.
        Order will be:
        Exception -> Anomaly -> Injection -> PII -> Audit -> RateLimit -> RBAC
    """
    default_exclude = ["/health", "/ready", "/docs", "/openapi.json", "/redoc"]
    exclude = exclude_paths or default_exclude

    # Add exception handler first (will be outermost)
    app.add_middleware(GovernanceExceptionMiddleware)

    # Add anomaly detection
    if enable_anomaly:
        app.add_middleware(
            AnomalyMiddleware,
            exclude_paths=exclude,
            block_on_critical=block_on_anomaly,
        )

    # Add injection detection (after anomaly, before PII)
    if enable_injection:
        app.add_middleware(
            InjectionMiddleware,
            exclude_paths=exclude,
        )

    # Add PII detection
    if enable_pii:
        app.add_middleware(
            PIIMiddleware,
            exclude_paths=exclude,
            block_on_pii=block_on_pii,
        )

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
