"""Unit tests for Azure AD Authentication module.

Tests cover:
- UserContext creation and methods
- UserRole enum and mappings
- JWT token validation (mocked)
- Azure AD OAuth flow (mocked)
- FastAPI dependencies
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import os

from app.auth.user_context import (
    UserContext,
    UserRole,
    DEFAULT_GROUP_ROLE_MAPPING,
    map_groups_to_roles,
)
from app.auth.jwt_handler import (
    JWTHandler,
    TokenValidationError,
    JWKSCache,
)
from app.auth.azure_ad import (
    AzureADAuth,
    AuthenticationError,
    TokenResponse,
    AuthState,
)


# =============================================================================
# UserContext Tests
# =============================================================================


class TestUserRole:
    """Tests for UserRole enum."""

    def test_role_values(self) -> None:
        """Test that roles have expected string values."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.OPERATOR.value == "operator"
        assert UserRole.USER.value == "user"
        assert UserRole.VIEWER.value == "viewer"
        assert UserRole.ANONYMOUS.value == "anonymous"

    def test_role_comparison(self) -> None:
        """Test role comparison."""
        assert UserRole.ADMIN == UserRole.ADMIN
        assert UserRole.ADMIN != UserRole.USER


class TestUserContext:
    """Tests for UserContext dataclass."""

    def test_create_user_context(self) -> None:
        """Test creating a basic user context."""
        user = UserContext(
            user_id="test-123",
            email="test@example.com",
            display_name="Test User",
            roles=[UserRole.USER],
            groups=["LangChain-Users"],
        )

        assert user.user_id == "test-123"
        assert user.email == "test@example.com"
        assert user.display_name == "Test User"
        assert UserRole.USER in user.roles
        assert user.is_authenticated

    def test_anonymous_user(self) -> None:
        """Test creating anonymous user context."""
        user = UserContext.anonymous()

        assert user.user_id == "anonymous"
        assert not user.is_authenticated
        assert user.primary_role == UserRole.ANONYMOUS

    def test_is_admin(self) -> None:
        """Test is_admin property."""
        admin_user = UserContext(
            user_id="admin-1",
            email="admin@example.com",
            display_name="Admin",
            roles=[UserRole.ADMIN],
        )
        regular_user = UserContext(
            user_id="user-1",
            email="user@example.com",
            display_name="User",
            roles=[UserRole.USER],
        )

        assert admin_user.is_admin
        assert not regular_user.is_admin

    def test_is_operator(self) -> None:
        """Test is_operator property (admin or operator)."""
        admin = UserContext(
            user_id="admin-1",
            email="admin@example.com",
            display_name="Admin",
            roles=[UserRole.ADMIN],
        )
        operator = UserContext(
            user_id="op-1",
            email="op@example.com",
            display_name="Operator",
            roles=[UserRole.OPERATOR],
        )
        user = UserContext(
            user_id="user-1",
            email="user@example.com",
            display_name="User",
            roles=[UserRole.USER],
        )

        assert admin.is_operator
        assert operator.is_operator
        assert not user.is_operator

    def test_primary_role(self) -> None:
        """Test primary role selection based on priority."""
        # Admin takes priority
        multi_role_user = UserContext(
            user_id="multi-1",
            email="multi@example.com",
            display_name="Multi Role",
            roles=[UserRole.USER, UserRole.ADMIN, UserRole.OPERATOR],
        )
        assert multi_role_user.primary_role == UserRole.ADMIN

        # Operator when no admin
        op_user = UserContext(
            user_id="op-1",
            email="op@example.com",
            display_name="Op",
            roles=[UserRole.USER, UserRole.OPERATOR],
        )
        assert op_user.primary_role == UserRole.OPERATOR

    def test_has_role(self) -> None:
        """Test has_role method."""
        user = UserContext(
            user_id="test-1",
            email="test@example.com",
            display_name="Test",
            roles=[UserRole.USER, UserRole.OPERATOR],
        )

        assert user.has_role(UserRole.USER)
        assert user.has_role(UserRole.OPERATOR)
        assert not user.has_role(UserRole.ADMIN)

    def test_has_any_role(self) -> None:
        """Test has_any_role method."""
        user = UserContext(
            user_id="test-1",
            email="test@example.com",
            display_name="Test",
            roles=[UserRole.USER],
        )

        assert user.has_any_role([UserRole.USER, UserRole.ADMIN])
        assert user.has_any_role([UserRole.USER])
        assert not user.has_any_role([UserRole.ADMIN, UserRole.OPERATOR])

    def test_has_all_roles(self) -> None:
        """Test has_all_roles method."""
        user = UserContext(
            user_id="test-1",
            email="test@example.com",
            display_name="Test",
            roles=[UserRole.USER, UserRole.OPERATOR],
        )

        assert user.has_all_roles([UserRole.USER])
        assert user.has_all_roles([UserRole.USER, UserRole.OPERATOR])
        assert not user.has_all_roles([UserRole.USER, UserRole.ADMIN])

    def test_in_group(self) -> None:
        """Test in_group method."""
        user = UserContext(
            user_id="test-1",
            email="test@example.com",
            display_name="Test",
            roles=[UserRole.USER],
            groups=["IT-Support", "LangChain-Users"],
        )

        assert user.in_group("IT-Support")
        assert user.in_group("LangChain-Users")
        assert not user.in_group("Admin-Group")

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        user = UserContext(
            user_id="test-1",
            email="test@example.com",
            display_name="Test User",
            roles=[UserRole.USER],
            groups=["Group1"],
            department="IT",
        )

        d = user.to_dict()
        assert d["user_id"] == "test-1"
        assert d["email"] == "test@example.com"
        assert d["roles"] == ["user"]
        assert d["is_authenticated"] is True
        assert d["department"] == "IT"

    def test_to_audit_dict(self) -> None:
        """Test to_audit_dict for minimal audit logging."""
        user = UserContext(
            user_id="test-1",
            email="test@example.com",
            display_name="Test User",
            roles=[UserRole.USER],
        )

        d = user.to_audit_dict()
        assert d["user_id"] == "test-1"
        assert d["email"] == "test@example.com"
        assert d["primary_role"] == "user"
        assert "display_name" not in d  # Minimal

    def test_from_claims(self) -> None:
        """Test creating UserContext from JWT claims."""
        claims = {
            "oid": "user-abc-123",
            "email": "john.doe@company.com",
            "name": "John Doe",
            "groups": ["LangChain-Users", "LangChain-Operators"],
            "department": "IT",
            "jobTitle": "Developer",
            "tid": "tenant-xyz",
            "exp": 1735689600,
        }

        user = UserContext.from_claims(claims)

        assert user.user_id == "user-abc-123"
        assert user.email == "john.doe@company.com"
        assert user.display_name == "John Doe"
        assert UserRole.USER in user.roles
        assert UserRole.OPERATOR in user.roles
        assert user.department == "IT"
        assert user.job_title == "Developer"
        assert user.tenant_id == "tenant-xyz"

    def test_from_claims_fallback_fields(self) -> None:
        """Test from_claims with fallback fields."""
        claims = {
            "sub": "sub-123",
            "preferred_username": "jane@company.com",
            "given_name": "Jane",
        }

        user = UserContext.from_claims(claims)

        assert user.user_id == "sub-123"
        assert user.email == "jane@company.com"
        assert user.display_name == "Jane"

    def test_from_claims_default_user_role(self) -> None:
        """Test from_claims defaults to USER role when no matching groups."""
        claims = {
            "oid": "user-123",
            "email": "user@company.com",
            "name": "User",
            "groups": ["Unknown-Group"],
        }

        user = UserContext.from_claims(claims)
        assert UserRole.USER in user.roles

    def test_frozen_dataclass(self) -> None:
        """Test that UserContext is immutable (frozen)."""
        user = UserContext(
            user_id="test-1",
            email="test@example.com",
            display_name="Test",
            roles=[UserRole.USER],
        )

        with pytest.raises(AttributeError):
            user.user_id = "new-id"  # type: ignore


class TestMapGroupsToRoles:
    """Tests for group-to-role mapping function."""

    def test_default_mapping(self) -> None:
        """Test default group-to-role mapping."""
        groups = ["LangChain-Admins", "LangChain-Users"]
        roles = map_groups_to_roles(groups)

        assert UserRole.ADMIN in roles
        assert UserRole.USER in roles

    def test_custom_mapping(self) -> None:
        """Test custom group-to-role mapping."""
        custom_mapping = {
            "MyAdmins": UserRole.ADMIN,
            "MyUsers": UserRole.USER,
        }
        groups = ["MyAdmins"]
        roles = map_groups_to_roles(groups, custom_mapping)

        assert roles == [UserRole.ADMIN]

    def test_unknown_groups_default_to_user(self) -> None:
        """Test that unknown groups default to USER role."""
        groups = ["Unknown-Group", "Another-Unknown"]
        roles = map_groups_to_roles(groups)

        assert roles == [UserRole.USER]

    def test_empty_groups(self) -> None:
        """Test empty groups list."""
        roles = map_groups_to_roles([])
        assert roles == [UserRole.USER]


# =============================================================================
# JWT Handler Tests
# =============================================================================


class TestJWKSCache:
    """Tests for JWKS cache."""

    def test_cache_not_expired(self) -> None:
        """Test cache is not expired when fresh."""
        import time

        cache = JWKSCache(
            keys={"key1": {"kty": "RSA"}},
            fetched_at=time.time(),
            ttl_seconds=3600,
        )

        assert not cache.is_expired

    def test_cache_expired(self) -> None:
        """Test cache is expired after TTL."""
        import time

        cache = JWKSCache(
            keys={"key1": {"kty": "RSA"}},
            fetched_at=time.time() - 4000,  # 4000 seconds ago
            ttl_seconds=3600,
        )

        assert cache.is_expired


class TestJWTHandler:
    """Tests for JWT token handler."""

    def test_is_configured_when_set(self) -> None:
        """Test is_configured when env vars are set."""
        handler = JWTHandler(
            tenant_id="test-tenant",
            client_id="test-client",
        )
        assert handler.is_configured

    def test_is_configured_when_not_set(self) -> None:
        """Test is_configured when env vars are not set."""
        handler = JWTHandler(tenant_id="", client_id="")
        assert not handler.is_configured

    def test_jwks_url(self) -> None:
        """Test JWKS URL construction."""
        handler = JWTHandler(tenant_id="test-tenant")
        expected = "https://login.microsoftonline.com/test-tenant/discovery/v2.0/keys"
        assert handler.jwks_url == expected

    def test_extract_token_valid(self) -> None:
        """Test extracting token from Authorization header."""
        handler = JWTHandler(tenant_id="test", client_id="test")
        token = handler._extract_token("Bearer abc123")
        assert token == "abc123"

    def test_extract_token_missing(self) -> None:
        """Test extracting token when header is missing."""
        handler = JWTHandler(tenant_id="test", client_id="test")
        with pytest.raises(TokenValidationError) as exc:
            handler._extract_token("")
        assert exc.value.error_code == "missing_token"

    def test_extract_token_invalid_scheme(self) -> None:
        """Test extracting token with wrong scheme."""
        handler = JWTHandler(tenant_id="test", client_id="test")
        with pytest.raises(TokenValidationError) as exc:
            handler._extract_token("Basic abc123")
        assert exc.value.error_code == "invalid_scheme"

    def test_extract_token_invalid_format(self) -> None:
        """Test extracting token with invalid format."""
        handler = JWTHandler(tenant_id="test", client_id="test")
        with pytest.raises(TokenValidationError) as exc:
            handler._extract_token("Bearer")
        assert exc.value.error_code == "invalid_format"


# =============================================================================
# Azure AD Auth Tests
# =============================================================================


class TestAuthState:
    """Tests for AuthState."""

    def test_generate_state(self) -> None:
        """Test generating auth state."""
        state = AuthState.generate("http://localhost:8000/callback")

        assert len(state.state) > 0
        assert len(state.nonce) > 0
        assert state.redirect_uri == "http://localhost:8000/callback"

    def test_generate_with_pkce(self) -> None:
        """Test generating state with PKCE."""
        state = AuthState.generate("http://localhost:8000/callback", use_pkce=True)

        assert state.code_verifier is not None
        assert state.code_challenge is not None

    def test_generate_without_pkce(self) -> None:
        """Test generating state without PKCE."""
        state = AuthState.generate("http://localhost:8000/callback", use_pkce=False)

        assert state.code_verifier is None
        assert state.code_challenge is None


class TestTokenResponse:
    """Tests for TokenResponse."""

    def test_from_dict(self) -> None:
        """Test creating TokenResponse from dictionary."""
        data = {
            "access_token": "access123",
            "id_token": "id456",
            "refresh_token": "refresh789",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "openid profile email",
        }

        response = TokenResponse.from_dict(data)

        assert response.access_token == "access123"
        assert response.id_token == "id456"
        assert response.refresh_token == "refresh789"
        assert response.token_type == "Bearer"
        assert response.expires_in == 3600

    def test_from_dict_minimal(self) -> None:
        """Test creating TokenResponse with minimal fields."""
        data = {"access_token": "access123"}

        response = TokenResponse.from_dict(data)

        assert response.access_token == "access123"
        assert response.refresh_token is None
        assert response.token_type == "Bearer"


class TestAzureADAuth:
    """Tests for AzureADAuth."""

    def test_is_configured(self) -> None:
        """Test is_configured check."""
        auth = AzureADAuth(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )
        assert auth.is_configured

    def test_is_not_configured(self) -> None:
        """Test is_configured when missing values."""
        auth = AzureADAuth(
            tenant_id="",
            client_id="client",
            client_secret="secret",
        )
        assert not auth.is_configured

    def test_authorize_url(self) -> None:
        """Test authorization URL construction."""
        auth = AzureADAuth(tenant_id="test-tenant")
        expected = "https://login.microsoftonline.com/test-tenant/oauth2/v2.0/authorize"
        assert auth.authorize_url == expected

    def test_token_url(self) -> None:
        """Test token URL construction."""
        auth = AzureADAuth(tenant_id="test-tenant")
        expected = "https://login.microsoftonline.com/test-tenant/oauth2/v2.0/token"
        assert auth.token_url == expected

    def test_get_authorization_url(self) -> None:
        """Test generating authorization URL."""
        auth = AzureADAuth(
            tenant_id="test-tenant",
            client_id="test-client",
            redirect_uri="http://localhost:8000/callback",
        )

        url, state = auth.get_authorization_url()

        assert "test-tenant" in url
        assert "test-client" in url
        assert "response_type=code" in url
        assert state.state is not None

    def test_get_authorization_url_with_options(self) -> None:
        """Test authorization URL with custom options."""
        auth = AzureADAuth(
            tenant_id="test-tenant",
            client_id="test-client",
        )

        url, _ = auth.get_authorization_url(
            prompt="consent",
            login_hint="user@example.com",
        )

        assert "prompt=consent" in url
        assert "login_hint=user%40example.com" in url

    @pytest.mark.asyncio
    async def test_exchange_code_invalid_state(self) -> None:
        """Test code exchange with invalid state."""
        auth = AzureADAuth(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="secret",
        )

        with pytest.raises(AuthenticationError) as exc:
            await auth.exchange_code("code123", "invalid-state")

        assert exc.value.error_code == "invalid_state"


# =============================================================================
# Integration-like tests with mocks
# =============================================================================


class TestJWTHandlerWithMocks:
    """Integration tests for JWT handler with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_validate_token_not_configured(self) -> None:
        """Test validation fails when not configured."""
        handler = JWTHandler(tenant_id="", client_id="")

        with pytest.raises(TokenValidationError) as exc:
            await handler.validate_token("Bearer token123")

        assert exc.value.error_code == "not_configured"


class TestAzureADAuthWithMocks:
    """Integration tests for Azure AD auth with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_exchange_code_success(self) -> None:
        """Test successful code exchange with mocked response."""
        auth = AzureADAuth(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Generate state first
        _, state = auth.get_authorization_url()

        # Mock the HTTP response
        mock_response = {
            "access_token": "access123",
            "id_token": "id456",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = MagicMock(
                status_code=200,
                json=lambda: mock_response,
            )

            tokens = await auth.exchange_code("code123", state.state)

            assert tokens.access_token == "access123"
            assert tokens.id_token == "id456"

    @pytest.mark.asyncio
    async def test_exchange_code_error_response(self) -> None:
        """Test code exchange with error response."""
        auth = AzureADAuth(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret",
        )

        _, state = auth.get_authorization_url()

        error_response = {
            "error": "invalid_grant",
            "error_description": "Code has expired",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = MagicMock(
                status_code=400,
                json=lambda: error_response,
            )

            with pytest.raises(AuthenticationError) as exc:
                await auth.exchange_code("expired-code", state.state)

            assert exc.value.error_code == "invalid_grant"


# =============================================================================
# Dependencies Tests
# =============================================================================


class TestAuthDependencies:
    """Tests for FastAPI auth dependencies."""

    @pytest.mark.asyncio
    async def test_get_current_user_no_auth(self) -> None:
        """Test get_current_user when auth is disabled."""
        # Temporarily disable auth
        with patch.dict(os.environ, {"AUTH_ENABLED": "false", "AUTH_BYPASS_DEV": "false"}):
            # Need to reload the module to pick up env change
            from app.auth import dependencies
            import importlib
            importlib.reload(dependencies)

            user = await dependencies.get_current_user_optional()
            assert user.user_id == "anonymous"

    @pytest.mark.asyncio
    async def test_dev_user_bypass(self) -> None:
        """Test development user bypass."""
        with patch.dict(os.environ, {"AUTH_ENABLED": "false", "AUTH_BYPASS_DEV": "true"}):
            from app.auth import dependencies
            import importlib
            importlib.reload(dependencies)

            # Create mock request with dev headers
            mock_request = MagicMock()
            mock_request.headers.get = lambda key, default=None: {
                "X-Dev-User-Email": "dev@localhost",
                "X-Dev-User-Name": "Dev User",
            }.get(key, default)

            user = await dependencies.get_current_user_optional(
                authorization=None,
                request=mock_request,
            )

            # Should get dev user with admin role
            assert user.is_admin
