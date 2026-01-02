"""JWT Token Handler for Azure AD tokens.

Provides secure JWT validation for Azure AD / Entra ID tokens:
- Signature verification using Azure AD public keys
- Issuer and audience validation
- Token expiration checking
- Claims extraction

Following Enterprise Development Standards:
- Security Architect: Cryptographic validation, secure key handling
- Software Engineer: Comprehensive error handling, type safety
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError

from app.auth.user_context import UserContext


class TokenValidationError(Exception):
    """Raised when token validation fails.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code for API responses
    """

    def __init__(self, message: str, error_code: str = "invalid_token") -> None:
        """Initialize TokenValidationError.

        Args:
            message: Human-readable error message.
            error_code: Machine-readable error code.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code


@dataclass
class JWKSCache:
    """Cache for Azure AD JSON Web Key Set (JWKS).

    Caches public keys used to verify JWT signatures.
    Keys are refreshed after TTL expires.
    """

    keys: dict[str, Any]
    fetched_at: float
    ttl_seconds: int = 3600  # 1 hour default

    @property
    def is_expired(self) -> bool:
        """Check if cache has expired."""
        return time.time() - self.fetched_at > self.ttl_seconds


class JWTHandler:
    """Handle JWT validation for Azure AD tokens.

    Validates tokens issued by Azure AD / Entra ID by:
    1. Fetching public keys from Azure AD's JWKS endpoint
    2. Verifying token signature
    3. Validating issuer, audience, and expiration
    4. Extracting user claims

    Example:
        >>> handler = JWTHandler()
        >>> try:
        ...     user = await handler.validate_token(bearer_token)
        ...     print(f"Authenticated: {user.email}")
        ... except TokenValidationError as e:
        ...     print(f"Auth failed: {e.message}")
    """

    # Azure AD endpoints
    AZURE_AD_AUTHORITY = "https://login.microsoftonline.com"
    JWKS_PATH = "/discovery/v2.0/keys"
    OPENID_CONFIG_PATH = "/.well-known/openid-configuration"

    def __init__(
        self,
        tenant_id: str | None = None,
        client_id: str | None = None,
        audience: str | None = None,
        jwks_ttl: int = 3600,
    ) -> None:
        """Initialize JWT handler.

        Args:
            tenant_id: Azure AD tenant ID (from env if not provided).
            client_id: Azure AD client/application ID (from env if not provided).
            audience: Expected token audience (defaults to client_id).
            jwks_ttl: JWKS cache TTL in seconds.
        """
        self.tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID", "")
        self.client_id = client_id or os.getenv("AZURE_CLIENT_ID", "")
        self.audience = audience or self.client_id
        self.jwks_ttl = jwks_ttl

        # Cache for JWKS
        self._jwks_cache: JWKSCache | None = None

        # Expected issuer(s) - Azure AD can issue from multiple
        self._valid_issuers = [
            f"https://login.microsoftonline.com/{self.tenant_id}/v2.0",
            f"https://sts.windows.net/{self.tenant_id}/",
        ]

    @property
    def jwks_url(self) -> str:
        """Get JWKS URL for the tenant."""
        return f"{self.AZURE_AD_AUTHORITY}/{self.tenant_id}{self.JWKS_PATH}"

    @property
    def is_configured(self) -> bool:
        """Check if Azure AD is properly configured."""
        return bool(self.tenant_id and self.client_id)

    async def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch JWKS from Azure AD.

        Returns:
            Dictionary of key ID to key data.

        Raises:
            TokenValidationError: If JWKS cannot be fetched.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.jwks_url)
                response.raise_for_status()
                data = response.json()

                # Convert to dict keyed by kid
                keys = {}
                for key in data.get("keys", []):
                    if "kid" in key:
                        keys[key["kid"]] = key

                return keys
        except httpx.HTTPError as e:
            msg = f"Failed to fetch JWKS: {e}"
            raise TokenValidationError(msg, "jwks_fetch_failed") from e
        except Exception as e:
            msg = f"Unexpected error fetching JWKS: {e}"
            raise TokenValidationError(msg, "jwks_error") from e

    async def _get_signing_key(self, kid: str) -> dict[str, Any]:
        """Get signing key for token verification.

        Args:
            kid: Key ID from token header.

        Returns:
            Key data for signature verification.

        Raises:
            TokenValidationError: If key is not found.
        """
        # Check cache
        if self._jwks_cache is None or self._jwks_cache.is_expired:
            keys = await self._fetch_jwks()
            self._jwks_cache = JWKSCache(
                keys=keys,
                fetched_at=time.time(),
                ttl_seconds=self.jwks_ttl,
            )

        if kid not in self._jwks_cache.keys:
            # Refresh cache once in case key rotated
            keys = await self._fetch_jwks()
            self._jwks_cache = JWKSCache(
                keys=keys,
                fetched_at=time.time(),
                ttl_seconds=self.jwks_ttl,
            )

        if kid not in self._jwks_cache.keys:
            msg = f"Signing key not found: {kid}"
            raise TokenValidationError(msg, "key_not_found")

        return self._jwks_cache.keys[kid]

    def _extract_token(self, authorization: str) -> str:
        """Extract token from Authorization header.

        Args:
            authorization: Full Authorization header value.

        Returns:
            Just the token part (without "Bearer ").

        Raises:
            TokenValidationError: If header format is invalid.
        """
        if not authorization:
            msg = "Authorization header is required"
            raise TokenValidationError(msg, "missing_token")

        parts = authorization.split()
        if len(parts) != 2:
            msg = "Invalid Authorization header format"
            raise TokenValidationError(msg, "invalid_format")

        scheme, token = parts
        if scheme.lower() != "bearer":
            msg = f"Invalid auth scheme: {scheme}"
            raise TokenValidationError(msg, "invalid_scheme")

        return token

    async def validate_token(
        self,
        authorization: str,
        verify_exp: bool = True,
    ) -> UserContext:
        """Validate Azure AD token and return user context.

        Args:
            authorization: Full Authorization header (e.g., "Bearer xyz...").
            verify_exp: Whether to verify token expiration.

        Returns:
            UserContext with user information from token.

        Raises:
            TokenValidationError: If validation fails.

        Example:
            >>> handler = JWTHandler()
            >>> user = await handler.validate_token("Bearer eyJ...")
            >>> print(user.email)
        """
        if not self.is_configured:
            msg = "Azure AD not configured. Set AZURE_TENANT_ID and AZURE_CLIENT_ID."
            raise TokenValidationError(msg, "not_configured")

        # Extract token from header
        token = self._extract_token(authorization)

        try:
            # Decode header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                msg = "Token header missing key ID (kid)"
                raise TokenValidationError(msg, "missing_kid")

            # Get signing key
            key = await self._get_signing_key(kid)

            # Verify and decode token
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self._valid_issuers,
                options={
                    "verify_exp": verify_exp,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

            # Create user context from claims
            return UserContext.from_claims(claims)

        except ExpiredSignatureError as e:
            msg = "Token has expired"
            raise TokenValidationError(msg, "token_expired") from e
        except JWTClaimsError as e:
            msg = f"Invalid token claims: {e}"
            raise TokenValidationError(msg, "invalid_claims") from e
        except JWTError as e:
            msg = f"Token validation failed: {e}"
            raise TokenValidationError(msg, "jwt_error") from e
        except TokenValidationError:
            raise
        except Exception as e:
            msg = f"Unexpected error: {e}"
            raise TokenValidationError(msg, "unknown_error") from e

    def decode_token_unsafe(self, token: str) -> dict[str, Any]:
        """Decode token without validation (for debugging only).

        WARNING: This does NOT verify the token. Only use for
        debugging or logging purposes.

        Args:
            token: JWT token string.

        Returns:
            Decoded claims (unverified).
        """
        return jwt.get_unverified_claims(token)


# Singleton instance for convenience
_jwt_handler: JWTHandler | None = None


def get_jwt_handler() -> JWTHandler:
    """Get or create singleton JWTHandler instance.

    Returns:
        Configured JWTHandler instance.
    """
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler


async def validate_bearer_token(authorization: str) -> UserContext:
    """Convenience function to validate a bearer token.

    Args:
        authorization: Full Authorization header.

    Returns:
        UserContext from validated token.

    Raises:
        TokenValidationError: If validation fails.
    """
    handler = get_jwt_handler()
    return await handler.validate_token(authorization)
