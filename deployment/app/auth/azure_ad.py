"""Azure AD / Entra ID OAuth 2.0 Authentication.

Provides OAuth 2.0 / OIDC authentication flow using MSAL:
- Authorization URL generation
- Token exchange from authorization code
- Token refresh handling
- Session management

Following Enterprise Development Standards:
- Security Architect: OAuth 2.0 best practices, PKCE support
- Software Engineer: Async-first, comprehensive error handling
"""

import os
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from app.auth.jwt_handler import JWTHandler, TokenValidationError
from app.auth.user_context import UserContext


class AuthenticationError(Exception):
    """Raised when authentication fails.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        error_description: Detailed error description from Azure AD
    """

    def __init__(
        self,
        message: str,
        error_code: str = "authentication_failed",
        error_description: str | None = None,
    ) -> None:
        """Initialize AuthenticationError.

        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code.
            error_description: Detailed description from Azure AD.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.error_description = error_description or message


@dataclass
class TokenResponse:
    """Response from token endpoint.

    Attributes:
        access_token: JWT access token for API calls
        id_token: JWT ID token with user claims
        refresh_token: Token for refreshing access (if available)
        token_type: Token type (usually "Bearer")
        expires_in: Seconds until access token expires
        scope: Granted scopes
    """

    access_token: str
    id_token: str
    refresh_token: str | None
    token_type: str
    expires_in: int
    scope: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenResponse":
        """Create TokenResponse from API response.

        Args:
            data: Response dictionary from token endpoint.

        Returns:
            TokenResponse instance.
        """
        return cls(
            access_token=data["access_token"],
            id_token=data.get("id_token", ""),
            refresh_token=data.get("refresh_token"),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", 3600),
            scope=data.get("scope", ""),
        )


@dataclass
class AuthState:
    """State for OAuth authorization flow.

    Used to prevent CSRF attacks and track flow state.
    """

    state: str
    nonce: str
    redirect_uri: str
    code_verifier: str | None = None  # For PKCE

    @classmethod
    def generate(
        cls,
        redirect_uri: str,
        use_pkce: bool = True,
    ) -> "AuthState":
        """Generate new auth state with random values.

        Args:
            redirect_uri: Callback URI for authorization.
            use_pkce: Whether to use PKCE (recommended).

        Returns:
            AuthState with random state and nonce.
        """
        import base64
        import hashlib

        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)

        code_verifier = None
        if use_pkce:
            code_verifier = secrets.token_urlsafe(64)

        return cls(
            state=state,
            nonce=nonce,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )

    @property
    def code_challenge(self) -> str | None:
        """Generate PKCE code challenge from verifier."""
        if not self.code_verifier:
            return None

        import base64
        import hashlib

        digest = hashlib.sha256(self.code_verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


class AzureADAuth:
    """Azure AD OAuth 2.0 authentication handler.

    Handles the full OAuth 2.0 authorization code flow:
    1. Generate authorization URL
    2. Handle callback with authorization code
    3. Exchange code for tokens
    4. Refresh tokens when needed

    Example:
        >>> auth = AzureADAuth()
        >>> # Step 1: Get auth URL
        >>> auth_url, state = auth.get_authorization_url()
        >>> # Step 2: User visits auth_url, gets redirected with code
        >>> # Step 3: Exchange code for tokens
        >>> tokens = await auth.exchange_code(code, state)
        >>> # Step 4: Validate and get user
        >>> user = await auth.get_user_from_token(tokens.access_token)
    """

    # Azure AD endpoints
    AUTHORITY = "https://login.microsoftonline.com"
    AUTHORIZE_ENDPOINT = "/oauth2/v2.0/authorize"
    TOKEN_ENDPOINT = "/oauth2/v2.0/token"

    # Default scopes
    DEFAULT_SCOPES = ["openid", "profile", "email", "offline_access"]

    def __init__(
        self,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        """Initialize Azure AD auth handler.

        Args:
            tenant_id: Azure AD tenant ID (from env if not provided).
            client_id: Application/client ID (from env if not provided).
            client_secret: Client secret (from env if not provided).
            redirect_uri: OAuth callback URI (from env if not provided).
            scopes: OAuth scopes to request.
        """
        self.tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID", "")
        self.client_id = client_id or os.getenv("AZURE_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("AZURE_CLIENT_SECRET", "")
        self.redirect_uri = redirect_uri or os.getenv(
            "AZURE_REDIRECT_URI",
            "http://localhost:8000/auth/callback",
        )
        self.scopes = scopes or self.DEFAULT_SCOPES

        # JWT handler for token validation
        self._jwt_handler = JWTHandler(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
        )

        # State storage (in production, use Redis or similar)
        self._pending_states: dict[str, AuthState] = {}

    @property
    def authorize_url(self) -> str:
        """Get authorization endpoint URL."""
        return f"{self.AUTHORITY}/{self.tenant_id}{self.AUTHORIZE_ENDPOINT}"

    @property
    def token_url(self) -> str:
        """Get token endpoint URL."""
        return f"{self.AUTHORITY}/{self.tenant_id}{self.TOKEN_ENDPOINT}"

    @property
    def is_configured(self) -> bool:
        """Check if Azure AD is properly configured."""
        return bool(
            self.tenant_id
            and self.client_id
            and self.client_secret
            and self.redirect_uri
        )

    def get_authorization_url(
        self,
        redirect_uri: str | None = None,
        scopes: list[str] | None = None,
        prompt: str | None = None,
        login_hint: str | None = None,
        use_pkce: bool = True,
    ) -> tuple[str, AuthState]:
        """Generate authorization URL for user login.

        Args:
            redirect_uri: Override default redirect URI.
            scopes: Override default scopes.
            prompt: Auth prompt behavior (login, consent, select_account).
            login_hint: Pre-fill email in login form.
            use_pkce: Use PKCE for enhanced security.

        Returns:
            Tuple of (authorization URL, auth state).

        Example:
            >>> auth = AzureADAuth()
            >>> url, state = auth.get_authorization_url()
            >>> # Redirect user to url
            >>> # Store state.state for verification
        """
        redirect = redirect_uri or self.redirect_uri
        auth_state = AuthState.generate(redirect, use_pkce=use_pkce)

        # Store state for verification
        self._pending_states[auth_state.state] = auth_state

        # Build query parameters
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": redirect,
            "scope": " ".join(scopes or self.scopes),
            "state": auth_state.state,
            "nonce": auth_state.nonce,
            "response_mode": "query",
        }

        if prompt:
            params["prompt"] = prompt

        if login_hint:
            params["login_hint"] = login_hint

        if use_pkce and auth_state.code_challenge:
            params["code_challenge"] = auth_state.code_challenge
            params["code_challenge_method"] = "S256"

        url = f"{self.authorize_url}?{urlencode(params)}"
        return url, auth_state

    async def exchange_code(
        self,
        code: str,
        state: str,
    ) -> TokenResponse:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback.
            state: State value from callback (for CSRF verification).

        Returns:
            TokenResponse with access and ID tokens.

        Raises:
            AuthenticationError: If exchange fails or state is invalid.
        """
        # Verify state
        auth_state = self._pending_states.pop(state, None)
        if not auth_state:
            msg = "Invalid or expired state parameter"
            raise AuthenticationError(msg, "invalid_state")

        # Build token request
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": auth_state.redirect_uri,
            "scope": " ".join(self.scopes),
        }

        # Add PKCE verifier if used
        if auth_state.code_verifier:
            data["code_verifier"] = auth_state.code_verifier

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    error_data = response.json()
                    msg = error_data.get("error_description", "Token exchange failed")
                    raise AuthenticationError(
                        msg,
                        error_data.get("error", "token_error"),
                        error_data.get("error_description"),
                    )

                return TokenResponse.from_dict(response.json())

        except httpx.HTTPError as e:
            msg = f"HTTP error during token exchange: {e}"
            raise AuthenticationError(msg, "http_error") from e
        except AuthenticationError:
            raise
        except Exception as e:
            msg = f"Unexpected error during token exchange: {e}"
            raise AuthenticationError(msg, "unknown_error") from e

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous authentication.

        Returns:
            New TokenResponse with fresh tokens.

        Raises:
            AuthenticationError: If refresh fails.
        """
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": " ".join(self.scopes),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    error_data = response.json()
                    msg = error_data.get("error_description", "Token refresh failed")
                    raise AuthenticationError(
                        msg,
                        error_data.get("error", "refresh_error"),
                        error_data.get("error_description"),
                    )

                return TokenResponse.from_dict(response.json())

        except httpx.HTTPError as e:
            msg = f"HTTP error during token refresh: {e}"
            raise AuthenticationError(msg, "http_error") from e
        except AuthenticationError:
            raise
        except Exception as e:
            msg = f"Unexpected error during token refresh: {e}"
            raise AuthenticationError(msg, "unknown_error") from e

    async def get_user_from_token(
        self,
        access_token: str,
    ) -> UserContext:
        """Get user context from access token.

        Args:
            access_token: Valid access token.

        Returns:
            UserContext with user information.

        Raises:
            TokenValidationError: If token is invalid.
        """
        return await self._jwt_handler.validate_token(f"Bearer {access_token}")

    async def authenticate(
        self,
        code: str,
        state: str,
    ) -> tuple[UserContext, TokenResponse]:
        """Complete authentication flow.

        Convenience method that exchanges code and validates user.

        Args:
            code: Authorization code from callback.
            state: State value from callback.

        Returns:
            Tuple of (UserContext, TokenResponse).

        Raises:
            AuthenticationError: If authentication fails.
            TokenValidationError: If token validation fails.
        """
        tokens = await self.exchange_code(code, state)
        user = await self.get_user_from_token(tokens.access_token)
        return user, tokens

    def clear_expired_states(self, max_age_seconds: int = 600) -> int:
        """Clear expired pending states.

        Call periodically to clean up old state entries.

        Args:
            max_age_seconds: Maximum age of state entries.

        Returns:
            Number of entries cleared.
        """
        # In production, use Redis with TTL instead
        # This is a simplified implementation
        count = len(self._pending_states)
        self._pending_states.clear()
        return count


# Module-level convenience functions
_auth_instance: AzureADAuth | None = None


def get_azure_ad_auth() -> AzureADAuth:
    """Get or create singleton AzureADAuth instance.

    Returns:
        Configured AzureADAuth instance.
    """
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = AzureADAuth()
    return _auth_instance


def get_auth_url(
    redirect_uri: str | None = None,
    prompt: str | None = None,
) -> tuple[str, str]:
    """Convenience function to get authorization URL.

    Args:
        redirect_uri: Override default redirect URI.
        prompt: Auth prompt behavior.

    Returns:
        Tuple of (authorization URL, state string).
    """
    auth = get_azure_ad_auth()
    url, auth_state = auth.get_authorization_url(
        redirect_uri=redirect_uri,
        prompt=prompt,
    )
    return url, auth_state.state


async def get_token_from_code(code: str, state: str) -> TokenResponse:
    """Convenience function to exchange code for tokens.

    Args:
        code: Authorization code.
        state: State value for verification.

    Returns:
        TokenResponse with access and ID tokens.
    """
    auth = get_azure_ad_auth()
    return await auth.exchange_code(code, state)
