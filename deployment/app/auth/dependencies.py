"""FastAPI Dependencies for Azure AD Authentication.

Provides dependency injection for authentication in FastAPI routes:
- Token validation
- User context injection
- Role-based access control
- Optional vs required authentication

Following Enterprise Development Standards:
- Security Architect: Secure defaults, minimal privilege
- Software Engineer: Clean dependency injection pattern
"""

import os
from functools import wraps
from typing import Annotated, Callable

from fastapi import Depends, Header, HTTPException, Request, status

from app.auth.jwt_handler import JWTHandler, TokenValidationError, get_jwt_handler
from app.auth.user_context import UserContext, UserRole


# Check if auth is enabled (allows graceful degradation in dev)
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"
AUTH_BYPASS_DEV = os.getenv("AUTH_BYPASS_DEV", "false").lower() == "true"


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    request: Request | None = None,
) -> UserContext:
    """Get authenticated user from request.

    This dependency validates the Authorization header and returns
    the authenticated user's context.

    Args:
        authorization: Authorization header (Bearer token).
        request: FastAPI request object.

    Returns:
        UserContext for authenticated user.

    Raises:
        HTTPException: 401 if not authenticated, 403 if invalid token.

    Example:
        >>> @app.get("/protected")
        >>> async def protected_route(
        ...     user: UserContext = Depends(get_current_user)
        ... ):
        ...     return {"email": user.email}
    """
    # Check for bypass in development
    if AUTH_BYPASS_DEV and not AUTH_ENABLED:
        return _get_dev_user(request)

    if not AUTH_ENABLED:
        return UserContext.anonymous()

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        handler = get_jwt_handler()
        return await handler.validate_token(authorization)
    except TokenValidationError as e:
        if e.error_code == "token_expired":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            ) from e
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=e.message,
        ) from e


async def get_current_user_optional(
    authorization: Annotated[str | None, Header()] = None,
    request: Request | None = None,
) -> UserContext:
    """Get user context, allowing anonymous access.

    Similar to get_current_user but returns anonymous context
    instead of raising 401 when no token is provided.

    Args:
        authorization: Authorization header (Bearer token).
        request: FastAPI request object.

    Returns:
        UserContext for user (authenticated or anonymous).

    Example:
        >>> @app.get("/public")
        >>> async def public_route(
        ...     user: UserContext = Depends(get_current_user_optional)
        ... ):
        ...     if user.is_authenticated:
        ...         return {"greeting": f"Hello, {user.display_name}!"}
        ...     return {"greeting": "Hello, guest!"}
    """
    if AUTH_BYPASS_DEV and not AUTH_ENABLED:
        return _get_dev_user(request)

    if not AUTH_ENABLED or not authorization:
        return UserContext.anonymous()

    try:
        handler = get_jwt_handler()
        return await handler.validate_token(authorization)
    except TokenValidationError:
        # Silently return anonymous for optional auth
        return UserContext.anonymous()


def require_role(required_role: UserRole) -> Callable:
    """Create dependency that requires a specific role.

    Args:
        required_role: Role that user must have.

    Returns:
        Dependency function.

    Example:
        >>> @app.post("/admin/settings")
        >>> async def admin_settings(
        ...     user: UserContext = Depends(require_role(UserRole.ADMIN))
        ... ):
        ...     return {"status": "ok"}
    """

    async def role_checker(
        user: Annotated[UserContext, Depends(get_current_user)],
    ) -> UserContext:
        if not user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role.value}' required",
            )
        return user

    return role_checker


def require_any_role(required_roles: list[UserRole]) -> Callable:
    """Create dependency that requires any of the specified roles.

    Args:
        required_roles: List of acceptable roles.

    Returns:
        Dependency function.

    Example:
        >>> @app.get("/operator-data")
        >>> async def operator_data(
        ...     user: UserContext = Depends(
        ...         require_any_role([UserRole.ADMIN, UserRole.OPERATOR])
        ...     )
        ... ):
        ...     return {"data": "sensitive"}
    """

    async def role_checker(
        user: Annotated[UserContext, Depends(get_current_user)],
    ) -> UserContext:
        if not user.has_any_role(required_roles):
            roles_str = ", ".join(r.value for r in required_roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles required: {roles_str}",
            )
        return user

    return role_checker


def require_all_roles(required_roles: list[UserRole]) -> Callable:
    """Create dependency that requires all of the specified roles.

    Args:
        required_roles: List of required roles.

    Returns:
        Dependency function.
    """

    async def role_checker(
        user: Annotated[UserContext, Depends(get_current_user)],
    ) -> UserContext:
        if not user.has_all_roles(required_roles):
            roles_str = ", ".join(r.value for r in required_roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"All roles required: {roles_str}",
            )
        return user

    return role_checker


def require_group(group_name: str) -> Callable:
    """Create dependency that requires membership in an Azure AD group.

    Args:
        group_name: Name of required Azure AD group.

    Returns:
        Dependency function.

    Example:
        >>> @app.get("/team-data")
        >>> async def team_data(
        ...     user: UserContext = Depends(require_group("IT-Support"))
        ... ):
        ...     return {"team": "IT Support"}
    """

    async def group_checker(
        user: Annotated[UserContext, Depends(get_current_user)],
    ) -> UserContext:
        if not user.in_group(group_name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Group membership required: {group_name}",
            )
        return user

    return group_checker


def _get_dev_user(request: Request | None = None) -> UserContext:
    """Get development user context for testing.

    Only used when AUTH_BYPASS_DEV is enabled.

    Args:
        request: FastAPI request (for extracting dev user headers).

    Returns:
        Development UserContext.
    """
    # Allow setting dev user via headers for testing
    if request and request.headers.get("X-Dev-User-Email"):
        return UserContext(
            user_id="dev-user",
            email=request.headers.get("X-Dev-User-Email", "dev@localhost"),
            display_name=request.headers.get("X-Dev-User-Name", "Dev User"),
            roles=[UserRole.ADMIN],  # Full access in dev
            groups=["LangChain-Admins"],
        )

    return UserContext(
        user_id="dev-user",
        email="dev@localhost",
        display_name="Development User",
        roles=[UserRole.ADMIN],
        groups=["LangChain-Admins"],
    )


# Type aliases for cleaner route definitions
CurrentUser = Annotated[UserContext, Depends(get_current_user)]
OptionalUser = Annotated[UserContext, Depends(get_current_user_optional)]
AdminUser = Annotated[UserContext, Depends(require_role(UserRole.ADMIN))]
OperatorUser = Annotated[
    UserContext, Depends(require_any_role([UserRole.ADMIN, UserRole.OPERATOR]))
]


# Decorator for class methods that need user context
def inject_user_context(func: Callable) -> Callable:
    """Decorator to inject user context into agent methods.

    Use this to ensure agent invocations include user context
    for audit logging and RBAC.

    Example:
        >>> class MyAgent:
        ...     @inject_user_context
        ...     async def invoke(self, input: dict, user: UserContext):
        ...         # user is automatically injected
        ...         pass
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # If user not provided, try to get from context
        if "user" not in kwargs or kwargs.get("user") is None:
            kwargs["user"] = UserContext.anonymous()
        return await func(*args, **kwargs)

    return wrapper
