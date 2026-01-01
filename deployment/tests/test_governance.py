"""Unit tests for the governance framework.

Tests cover:
- Role-Based Access Control (RBAC)
- Audit logging
- Rate limiting
- Approval workflows
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# RBAC tests
from app.governance.rbac import (
    Permission,
    PermissionDeniedError,
    RBACConfig,
    RBACManager,
    Role,
    ROLE_PERMISSIONS,
    UserContext,
    check_permission,
    get_permissions_for_role,
    get_rbac_manager,
    require_permission,
    reset_rbac_manager,
)

# Audit logging tests
from app.governance.audit_logger import (
    AuditAction,
    AuditConfig,
    AuditEntry,
    AuditLevel,
    AuditLogger,
    audit_agent_response,
    get_audit_logger,
    reset_audit_logger,
)

# Rate limiting tests
from app.governance.rate_limiter import (
    InMemoryRateLimiter,
    RateLimitConfig,
    RateLimitExceededError,
    RateLimiter,
    RateLimitResult,
    RateLimitRule,
    TokenBucket,
    check_rate_limit,
    get_rate_limiter,
    require_rate_limit,
    reset_rate_limiter,
)

# Approval workflow tests
from app.governance.approval_workflow import (
    ActionType,
    ApprovalLevel,
    ApprovalRequest,
    ApprovalRejectedError,
    ApprovalRequiredError,
    ApprovalResponse,
    ApprovalStatus,
    ApprovalWorkflowConfig,
    ApprovalWorkflowManager,
    get_approval_level,
    get_approval_manager,
    request_approval,
    requires_approval,
    reset_approval_manager,
)


# ==================== RBAC Tests ====================


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self) -> None:
        """Test role enum values."""
        assert Role.ADMIN.value == "admin"
        assert Role.OPERATOR.value == "operator"
        assert Role.USER.value == "user"
        assert Role.VIEWER.value == "viewer"
        assert Role.SERVICE.value == "service"

    def test_role_from_string(self) -> None:
        """Test creating role from string."""
        assert Role("admin") == Role.ADMIN
        assert Role("user") == Role.USER


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self) -> None:
        """Test permission enum values."""
        assert Permission.AGENT_INVOKE.value == "agent:invoke"
        assert Permission.APPROVE_L1.value == "agent:approve:l1"
        assert Permission.ALL.value == "*"


class TestUserContext:
    """Tests for UserContext."""

    def test_user_context_creation(self) -> None:
        """Test creating a user context."""
        ctx = UserContext(user_id="user123", role=Role.USER)
        assert ctx.user_id == "user123"
        assert ctx.role == Role.USER
        assert ctx.permissions == set()

    def test_has_permission_explicit(self) -> None:
        """Test explicit permission checking."""
        ctx = UserContext(
            user_id="user123",
            role=Role.VIEWER,
            permissions={Permission.AGENT_INVOKE},
        )
        assert ctx.has_permission(Permission.AGENT_INVOKE)
        assert not ctx.has_permission(Permission.APPROVE_L1)

    def test_has_permission_role_based(self) -> None:
        """Test role-based permission checking."""
        ctx = UserContext(user_id="user123", role=Role.USER)
        assert ctx.has_permission(Permission.AGENT_INVOKE)
        assert ctx.has_permission(Permission.AGENT_READ)
        assert not ctx.has_permission(Permission.APPROVE_L1)

    def test_admin_has_all_permissions(self) -> None:
        """Test that admin has all permissions."""
        ctx = UserContext(user_id="admin", role=Role.ADMIN)
        assert ctx.has_permission(Permission.AGENT_INVOKE)
        assert ctx.has_permission(Permission.APPROVE_L1)
        assert ctx.has_permission(Permission.APPROVE_L2)
        assert ctx.has_permission(Permission.APPROVE_L3)
        assert ctx.has_permission(Permission.SYSTEM_ADMIN)

    def test_wildcard_permission(self) -> None:
        """Test wildcard permission grants everything."""
        ctx = UserContext(
            user_id="super",
            role=Role.USER,
            permissions={Permission.ALL},
        )
        assert ctx.has_permission(Permission.SYSTEM_ADMIN)


class TestRBACManager:
    """Tests for RBACManager."""

    def setup_method(self) -> None:
        """Reset global state before each test."""
        reset_rbac_manager()

    def test_get_user_context_no_api_key(self) -> None:
        """Test getting user context without API key."""
        manager = RBACManager()
        ctx = manager.get_user_context()
        assert ctx.user_id == "anonymous"
        assert ctx.role == Role.VIEWER

    def test_get_user_context_with_admin_key(self) -> None:
        """Test getting user context with admin API key."""
        manager = RBACManager()
        ctx = manager.get_user_context(api_key="sk-admin-test-12345")
        assert ctx.role == Role.ADMIN

    def test_get_user_context_with_operator_key(self) -> None:
        """Test getting user context with operator API key."""
        manager = RBACManager()
        ctx = manager.get_user_context(api_key="sk-operator-test-12345")
        assert ctx.role == Role.OPERATOR

    def test_get_user_context_with_service_key(self) -> None:
        """Test getting user context with service API key."""
        manager = RBACManager()
        ctx = manager.get_user_context(api_key="sk-service-bot-12345")
        assert ctx.role == Role.SERVICE

    def test_rbac_disabled(self) -> None:
        """Test that disabled RBAC returns admin context."""
        config = RBACConfig(enabled=False)
        manager = RBACManager(config)
        ctx = manager.get_user_context()
        assert ctx.role == Role.ADMIN

    def test_check_permission(self) -> None:
        """Test permission checking through manager."""
        manager = RBACManager()
        ctx = UserContext(user_id="test", role=Role.USER)
        assert manager.check_permission(ctx, Permission.AGENT_INVOKE)
        assert not manager.check_permission(ctx, Permission.APPROVE_L1)

    def test_require_permission_success(self) -> None:
        """Test require_permission with valid permission."""
        manager = RBACManager()
        ctx = UserContext(user_id="test", role=Role.USER)
        manager.require_permission(ctx, Permission.AGENT_INVOKE)  # Should not raise

    def test_require_permission_denied(self) -> None:
        """Test require_permission raises when denied."""
        manager = RBACManager()
        ctx = UserContext(user_id="test", role=Role.USER)
        with pytest.raises(PermissionDeniedError) as exc:
            manager.require_permission(ctx, Permission.APPROVE_L1)
        assert exc.value.user_id == "test"
        assert exc.value.permission == Permission.APPROVE_L1

    def test_api_key_caching(self) -> None:
        """Test that API key contexts are cached."""
        manager = RBACManager()
        ctx1 = manager.get_user_context(api_key="sk-test-12345")
        ctx2 = manager.get_user_context(api_key="sk-test-12345")
        assert ctx1 is ctx2

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        manager = RBACManager()
        manager.get_user_context(api_key="sk-test-12345")
        manager.clear_cache()
        assert len(manager._api_key_cache) == 0


class TestRBACConvenienceFunctions:
    """Tests for RBAC convenience functions."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_rbac_manager()

    def test_check_permission_function(self) -> None:
        """Test check_permission convenience function."""
        result = check_permission("sk-admin-test", Permission.SYSTEM_ADMIN)
        assert result is True

    def test_require_permission_function(self) -> None:
        """Test require_permission convenience function."""
        ctx = require_permission("sk-admin-test", Permission.SYSTEM_ADMIN)
        assert ctx.role == Role.ADMIN

    def test_get_permissions_for_role(self) -> None:
        """Test getting permissions for a role."""
        perms = get_permissions_for_role(Role.OPERATOR)
        assert Permission.AGENT_INVOKE in perms
        assert Permission.APPROVE_L1 in perms
        assert Permission.SYSTEM_ADMIN not in perms


# ==================== Audit Logger Tests ====================


class TestAuditEntry:
    """Tests for AuditEntry."""

    def test_audit_entry_creation(self) -> None:
        """Test creating an audit entry."""
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = AuditEntry(
            timestamp=timestamp,
            action=AuditAction.AGENT_INVOKE,
            user_id="user123",
            agent_type="helpdesk",
        )
        assert entry.action == AuditAction.AGENT_INVOKE
        assert entry.user_id == "user123"
        assert entry.timestamp is not None

    def test_audit_entry_to_json(self) -> None:
        """Test converting entry to JSON."""
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = AuditEntry(
            timestamp=timestamp,
            action=AuditAction.AGENT_INVOKE,
            user_id="user123",
        )
        json_str = entry.to_json()
        assert "agent:invoke" in json_str
        assert "user123" in json_str

    def test_audit_entry_from_json(self) -> None:
        """Test creating entry from JSON."""
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = AuditEntry(
            timestamp=timestamp,
            action=AuditAction.AGENT_INVOKE,
            user_id="user123",
        )
        json_str = entry.to_json()
        restored = AuditEntry.from_json(json_str)
        assert restored.action == entry.action
        assert restored.user_id == entry.user_id


class TestAuditLogger:
    """Tests for AuditLogger."""

    def setup_method(self) -> None:
        """Reset global state and create temp dir."""
        reset_audit_logger()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audit_logger_creation(self) -> None:
        """Test creating an audit logger."""
        config = AuditConfig(log_path=os.path.join(self.temp_dir, "audit.jsonl"))
        logger = AuditLogger(config)
        assert logger is not None

    def test_log_entry(self) -> None:
        """Test logging an entry."""
        config = AuditConfig(log_path=os.path.join(self.temp_dir, "audit.jsonl"))
        logger = AuditLogger(config)

        entry = logger.log(
            action=AuditAction.AGENT_INVOKE,
            user_id="test_user",
            agent_type="helpdesk",
        )

        assert entry.action == AuditAction.AGENT_INVOKE
        assert entry.user_id == "test_user"

        # Check file was written
        log_path = os.path.join(self.temp_dir, "audit.jsonl")
        assert os.path.exists(log_path)

    def test_log_with_privacy_hashing(self) -> None:
        """Test that inputs are hashed for privacy."""
        config = AuditConfig(
            log_path=os.path.join(self.temp_dir, "audit.jsonl"),
            log_inputs=False,  # Hash by default
            log_outputs=False,  # Hash by default
        )
        logger = AuditLogger(config)

        entry = logger.log(
            action=AuditAction.AGENT_INVOKE,
            user_id="test_user",
            input_text="my secret query",
            output_text="my secret response",
        )

        # Check that hashes are set, not plain text
        assert entry.input_hash is not None
        assert entry.output_hash is not None
        assert entry.input_hash != "my secret query"
        assert entry.output_hash != "my secret response"

    @pytest.mark.asyncio
    async def test_log_async(self) -> None:
        """Test async logging."""
        config = AuditConfig(log_path=os.path.join(self.temp_dir, "audit.jsonl"))
        logger = AuditLogger(config)

        entry = await logger.log_async(
            action=AuditAction.SYSTEM_START,
            user_id="async_user",
        )

        assert entry.user_id == "async_user"

    def test_query_entries(self) -> None:
        """Test querying audit entries."""
        config = AuditConfig(log_path=os.path.join(self.temp_dir, "audit.jsonl"))
        logger = AuditLogger(config)

        # Log some entries
        logger.log(action=AuditAction.AGENT_INVOKE, user_id="user1")
        logger.log(action=AuditAction.SYSTEM_START, user_id="user2")
        logger.log(action=AuditAction.AGENT_INVOKE, user_id="user1")

        # Query by user
        entries = logger.query(user_id="user1")
        assert len(entries) == 2

        # Query by action
        entries = logger.query(action=AuditAction.SYSTEM_START)
        assert len(entries) == 1

    def test_export_entries(self) -> None:
        """Test exporting audit entries."""
        config = AuditConfig(log_path=os.path.join(self.temp_dir, "audit.jsonl"))
        logger = AuditLogger(config)

        logger.log(action=AuditAction.AGENT_INVOKE, user_id="user1")
        logger.log(action=AuditAction.SYSTEM_START, user_id="user2")

        export_path = os.path.join(self.temp_dir, "export.jsonl")
        count = logger.export(export_path)

        assert count == 2
        assert os.path.exists(export_path)


class TestAuditConvenienceFunctions:
    """Tests for audit convenience functions."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_audit_logger()

    def test_audit_agent_response(self) -> None:
        """Test audit_agent_response convenience function."""
        entry = audit_agent_response(
            user_id="test_user",
            agent_type="helpdesk",
            input_text="Help me",
            output_text="Sure!",
            duration_ms=100,
        )

        assert entry.user_id == "test_user"
        assert entry.agent_type == "helpdesk"
        assert entry.duration_ms == 100


# ==================== Rate Limiter Tests ====================


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_token_bucket_creation(self) -> None:
        """Test creating a token bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.capacity == 10
        assert bucket.tokens == 10.0

    def test_consume_tokens(self) -> None:
        """Test consuming tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        success, remaining = bucket.consume(5)
        assert success is True
        assert remaining >= 4.9  # Allow for small refill during test

        success, remaining = bucket.consume(5)
        assert success is True
        assert remaining < 1.0  # Should be nearly empty

        success, remaining = bucket.consume(1)
        assert success is False

    def test_token_refill(self) -> None:
        """Test that tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 per second

        bucket.consume(10)  # Empty the bucket
        assert bucket.tokens == 0.0

        # Simulate time passing
        import time
        time.sleep(0.1)  # 100ms

        # Force update by consuming 0
        bucket.consume(0)
        assert bucket.tokens >= 0.9  # Should have ~1 token

    def test_time_until_available(self) -> None:
        """Test calculating time until tokens available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.consume(10)  # Empty bucket

        wait_time = bucket.time_until_available(5)
        assert wait_time >= 4.9  # Should be ~5 seconds


class TestRateLimitRule:
    """Tests for RateLimitRule."""

    def test_rule_creation(self) -> None:
        """Test creating a rate limit rule."""
        rule = RateLimitRule(scope="user", limit=100, window_seconds=60)
        assert rule.scope == "user"
        assert rule.limit == 100

    def test_effective_burst(self) -> None:
        """Test effective burst calculation."""
        rule = RateLimitRule(scope="user", limit=100)
        assert rule.effective_burst == 100

        rule_with_burst = RateLimitRule(scope="user", limit=100, burst_limit=150)
        assert rule_with_burst.effective_burst == 150


class TestInMemoryRateLimiter:
    """Tests for InMemoryRateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self) -> None:
        """Test that rate limiter allows requests within limit."""
        config = RateLimitConfig(enabled=True, default_limit=10, burst_multiplier=1.0)
        limiter = InMemoryRateLimiter(config)

        result = await limiter.check("user1")
        assert result.allowed is True
        assert result.remaining == 9

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess(self) -> None:
        """Test that rate limiter blocks excess requests."""
        config = RateLimitConfig(enabled=True, default_limit=3, burst_multiplier=1.0)
        limiter = InMemoryRateLimiter(config)

        # Exhaust limit
        for _ in range(3):
            result = await limiter.check("user1")
            assert result.allowed is True

        # Next should be blocked
        result = await limiter.check("user1")
        assert result.allowed is False
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_rate_limiter_disabled(self) -> None:
        """Test that disabled rate limiter allows all."""
        config = RateLimitConfig(enabled=False)
        limiter = InMemoryRateLimiter(config)

        for _ in range(100):
            result = await limiter.check("user1")
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self) -> None:
        """Test resetting rate limit for a key."""
        config = RateLimitConfig(enabled=True, default_limit=3, burst_multiplier=1.0)
        limiter = InMemoryRateLimiter(config)

        # Exhaust limit
        for _ in range(3):
            await limiter.check("user1")

        # Reset
        await limiter.reset("user1")

        # Should be allowed again
        result = await limiter.check("user1")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_rate_limiter_per_scope(self) -> None:
        """Test that different scopes have separate limits."""
        config = RateLimitConfig(enabled=True, default_limit=2, burst_multiplier=1.0)
        limiter = InMemoryRateLimiter(config)

        # Exhaust scope A
        await limiter.check("user1", scope="A")
        await limiter.check("user1", scope="A")
        result = await limiter.check("user1", scope="A")
        assert result.allowed is False

        # Scope B should still work
        result = await limiter.check("user1", scope="B")
        assert result.allowed is True


class TestRateLimiter:
    """Tests for unified RateLimiter."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_rate_limiter()

    @pytest.mark.asyncio
    async def test_check_user(self) -> None:
        """Test checking rate limit for user."""
        limiter = RateLimiter()
        result = await limiter.check_user("user123")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_agent(self) -> None:
        """Test checking rate limit for agent."""
        limiter = RateLimiter()
        result = await limiter.check_agent("user123", "helpdesk")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_global(self) -> None:
        """Test checking global rate limit."""
        limiter = RateLimiter()
        result = await limiter.check_global()
        assert result.allowed is True


class TestRateLimitConvenienceFunctions:
    """Tests for rate limit convenience functions."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_rate_limiter()

    @pytest.mark.asyncio
    async def test_check_rate_limit(self) -> None:
        """Test check_rate_limit function."""
        result = await check_rate_limit("user123", "helpdesk")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_require_rate_limit_success(self) -> None:
        """Test require_rate_limit when allowed."""
        result = await require_rate_limit("user123")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_require_rate_limit_exceeded(self) -> None:
        """Test require_rate_limit when exceeded."""
        # Create limited config
        config = RateLimitConfig(enabled=True, default_limit=1, burst_multiplier=1.0)
        limiter = RateLimiter(config)

        # Monkeypatch the global getter
        with patch("app.governance.rate_limiter.get_rate_limiter", return_value=limiter):
            # First should pass
            await require_rate_limit("user123")

            # Second should fail
            with pytest.raises(RateLimitExceededError):
                await require_rate_limit("user123")


# ==================== Approval Workflow Tests ====================


class TestApprovalLevel:
    """Tests for ApprovalLevel enum."""

    def test_approval_level_values(self) -> None:
        """Test approval level values."""
        assert ApprovalLevel.L1.value == "l1"
        assert ApprovalLevel.L2.value == "l2"
        assert ApprovalLevel.L3.value == "l3"


class TestApprovalRequest:
    """Tests for ApprovalRequest."""

    def test_request_creation(self) -> None:
        """Test creating an approval request."""
        request = ApprovalRequest(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
        )
        assert request.status == ApprovalStatus.PENDING
        assert request.level == ApprovalLevel.L1
        assert request.id is not None

    def test_request_expiration(self) -> None:
        """Test request expiration check."""
        request = ApprovalRequest(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
        )
        assert not request.is_expired()

        # Set to past
        past = datetime.now(timezone.utc) - timedelta(hours=48)
        request.expires_at = past.isoformat()
        assert request.is_expired()

    def test_request_to_dict(self) -> None:
        """Test converting request to dict."""
        request = ApprovalRequest(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
        )
        d = request.to_dict()
        assert d["action_type"] == "servicenow:create_incident"
        assert d["requester_id"] == "user123"


class TestApprovalWorkflowManager:
    """Tests for ApprovalWorkflowManager."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_approval_manager()

    def test_create_request(self) -> None:
        """Test creating an approval request."""
        manager = ApprovalWorkflowManager()
        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={"description": "Test incident"},
        )
        assert request.status == ApprovalStatus.PENDING
        assert request.requester_id == "user123"

    def test_get_request(self) -> None:
        """Test getting a request by ID."""
        manager = ApprovalWorkflowManager()
        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={},
        )

        fetched = manager.get_request(request.id)
        assert fetched is not None
        assert fetched.id == request.id

    def test_list_pending(self) -> None:
        """Test listing pending requests."""
        manager = ApprovalWorkflowManager()

        manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user1",
            agent_type="servicenow",
            action_details={},
        )
        manager.create_request(
            action_type=ActionType.PASSWORD_RESET,
            requester_id="user2",
            agent_type="helpdesk",
            action_details={},
        )

        pending = manager.list_pending()
        assert len(pending) == 2

        # Filter by level
        l2_pending = manager.list_pending(level=ApprovalLevel.L2)
        assert len(l2_pending) == 1

    def test_can_approve_admin(self) -> None:
        """Test that admin can approve any level."""
        manager = ApprovalWorkflowManager()
        admin_ctx = UserContext(user_id="admin", role=Role.ADMIN)

        for action_type in ActionType:
            request = ApprovalRequest(
                action_type=action_type,
                requester_id="user123",
                agent_type="test",
            )
            assert manager.can_approve(admin_ctx, request)

    def test_can_approve_operator(self) -> None:
        """Test operator approval permissions."""
        manager = ApprovalWorkflowManager()
        operator_ctx = UserContext(user_id="operator", role=Role.OPERATOR)

        # L1 should pass
        l1_request = ApprovalRequest(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="test",
            level=ApprovalLevel.L1,
        )
        assert manager.can_approve(operator_ctx, l1_request)

        # L3 should fail
        l3_request = ApprovalRequest(
            action_type=ActionType.SYSTEM_RESTART,
            requester_id="user123",
            agent_type="test",
            level=ApprovalLevel.L3,
        )
        assert not manager.can_approve(operator_ctx, l3_request)

    def test_approve_request(self) -> None:
        """Test approving a request."""
        manager = ApprovalWorkflowManager()
        admin_ctx = UserContext(user_id="admin", role=Role.ADMIN)

        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={},
        )

        response = manager.approve(request.id, admin_ctx, reason="Looks good")

        assert response.approved is True
        assert response.approver_id == "admin"

        # Check request was updated
        updated = manager.get_request(request.id)
        assert updated.status == ApprovalStatus.APPROVED

    def test_reject_request(self) -> None:
        """Test rejecting a request."""
        manager = ApprovalWorkflowManager()
        admin_ctx = UserContext(user_id="admin", role=Role.ADMIN)

        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={},
        )

        response = manager.reject(request.id, admin_ctx, reason="Not appropriate")

        assert response.approved is False
        assert response.reason == "Not appropriate"

        updated = manager.get_request(request.id)
        assert updated.status == ApprovalStatus.REJECTED

    def test_reject_requires_reason(self) -> None:
        """Test that rejection requires a reason when configured."""
        config = ApprovalWorkflowConfig(require_reason_on_reject=True)
        manager = ApprovalWorkflowManager(config)
        admin_ctx = UserContext(user_id="admin", role=Role.ADMIN)

        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={},
        )

        with pytest.raises(ValueError) as exc:
            manager.reject(request.id, admin_ctx)
        assert "reason is required" in str(exc.value)

    def test_cancel_request(self) -> None:
        """Test cancelling a request."""
        manager = ApprovalWorkflowManager()

        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={},
        )

        result = manager.cancel(request.id, "user123")
        assert result is True

        updated = manager.get_request(request.id)
        assert updated.status == ApprovalStatus.CANCELLED

    def test_cancel_wrong_user(self) -> None:
        """Test that only requester can cancel."""
        manager = ApprovalWorkflowManager()

        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={},
        )

        with pytest.raises(PermissionError):
            manager.cancel(request.id, "other_user")

    def test_max_pending_limit(self) -> None:
        """Test maximum pending requests limit."""
        config = ApprovalWorkflowConfig(max_pending_per_user=2)
        manager = ApprovalWorkflowManager(config)

        # Create max requests
        for _ in range(2):
            manager.create_request(
                action_type=ActionType.CREATE_INCIDENT,
                requester_id="user123",
                agent_type="servicenow",
                action_details={},
            )

        # Third should fail
        with pytest.raises(ValueError) as exc:
            manager.create_request(
                action_type=ActionType.CREATE_INCIDENT,
                requester_id="user123",
                agent_type="servicenow",
                action_details={},
            )
        assert "exceeded" in str(exc.value)

    def test_callback_notification(self) -> None:
        """Test that callbacks are notified."""
        manager = ApprovalWorkflowManager()
        admin_ctx = UserContext(user_id="admin", role=Role.ADMIN)

        callback_called = []

        def my_callback(req: ApprovalRequest, resp: ApprovalResponse) -> None:
            callback_called.append((req.id, resp.approved))

        manager.register_callback(my_callback)

        request = manager.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={},
        )

        manager.approve(request.id, admin_ctx)

        assert len(callback_called) == 1
        assert callback_called[0][1] is True


class TestApprovalConvenienceFunctions:
    """Tests for approval convenience functions."""

    def setup_method(self) -> None:
        """Reset global state."""
        reset_approval_manager()

    def test_requires_approval(self) -> None:
        """Test requires_approval function."""
        assert requires_approval(ActionType.SYSTEM_RESTART)
        assert requires_approval(ActionType.CREATE_INCIDENT)

    def test_get_approval_level(self) -> None:
        """Test get_approval_level function."""
        assert get_approval_level(ActionType.CREATE_INCIDENT) == ApprovalLevel.L1
        assert get_approval_level(ActionType.PASSWORD_RESET) == ApprovalLevel.L2
        assert get_approval_level(ActionType.SYSTEM_RESTART) == ApprovalLevel.L3

    @pytest.mark.asyncio
    async def test_request_approval_no_wait(self) -> None:
        """Test request_approval without waiting."""
        request = await request_approval(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id="user123",
            agent_type="servicenow",
            action_details={"test": "data"},
            wait=False,
        )

        assert request.status == ApprovalStatus.PENDING


# ==================== Integration Tests ====================


class TestGovernanceIntegration:
    """Integration tests for governance framework."""

    def setup_method(self) -> None:
        """Reset all global state."""
        reset_rbac_manager()
        reset_audit_logger()
        reset_rate_limiter()
        reset_approval_manager()

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Test a complete governance workflow."""
        # 1. Get user context
        rbac = get_rbac_manager()
        user_ctx = rbac.get_user_context(api_key="sk-operator-john-12345")
        assert user_ctx.role == Role.OPERATOR

        # 2. Check rate limit
        result = await check_rate_limit(user_ctx.user_id, "servicenow")
        assert result.allowed is True

        # 3. Log the request
        audit = get_audit_logger()
        entry = audit.log(
            action=AuditAction.AGENT_INVOKE,
            user_id=user_ctx.user_id,
            agent_type="servicenow",
        )
        assert entry is not None

        # 4. Create approval request
        approval_mgr = get_approval_manager()
        request = approval_mgr.create_request(
            action_type=ActionType.CREATE_INCIDENT,
            requester_id=user_ctx.user_id,
            agent_type="servicenow",
            action_details={"description": "Test"},
        )
        assert request.status == ApprovalStatus.PENDING

        # 5. Approve (operator can approve L1)
        response = approval_mgr.approve(request.id, user_ctx)
        assert response.approved is True
