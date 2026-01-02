"""Azure AD Authentication Module for LangChain Agents Platform.

Provides enterprise-grade authentication using Azure AD / Entra ID:
- OAuth 2.0 / OIDC authentication flow
- JWT token validation
- User context injection for agents
- Group-based RBAC mapping

Following Enterprise Development Standards:
- Software Architect: Modular authentication design
- Security Architect: OAuth 2.0, JWT validation, secure token handling
- Data Architect: User context propagation
- Software Engineer: Type-safe, well-documented

Example:
    >>> from app.auth import get_current_user, AzureADAuth
    >>> auth = AzureADAuth()
    >>> user = await auth.validate_token(token)
    >>> print(user.email, user.roles)
"""

from app.auth.azure_ad import AzureADAuth, get_auth_url, get_token_from_code
from app.auth.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_role,
    require_any_role,
)
from app.auth.jwt_handler import JWTHandler, TokenValidationError
from app.auth.user_context import UserContext, UserRole

__all__ = [
    # Main auth class
    "AzureADAuth",
    # Auth URL helpers
    "get_auth_url",
    "get_token_from_code",
    # JWT handling
    "JWTHandler",
    "TokenValidationError",
    # User context
    "UserContext",
    "UserRole",
    # FastAPI dependencies
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "require_any_role",
]
