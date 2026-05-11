"""Governance framework for enterprise IT agents.

This module provides a comprehensive governance layer including:
- Role-Based Access Control (RBAC)
- Audit logging for compliance
- Rate limiting
- Approval workflows for sensitive actions
- PII detection and masking
- Token usage cost tracking
- Anomaly detection
- Prompt injection / jailbreak detection
- FastAPI middleware integration

Usage:
    from app.governance import (
        # RBAC
        Role, Permission, UserContext, RBACManager,
        get_rbac_manager, check_permission, require_permission,

        # Audit
        AuditLogger, AuditAction, AuditLevel,
        get_audit_logger, audit_agent_response,

        # Rate limiting
        RateLimiter, RateLimitResult, RateLimitRule,
        get_rate_limiter, check_rate_limit,

        # Approval workflow
        ApprovalWorkflowManager, ApprovalRequest, ApprovalLevel,
        get_approval_manager, request_approval,

        # PII detection
        PIIDetector, PIIType, PIIMatch, PIIConfig,
        get_pii_detector, detect_pii, mask_pii,

        # Injection detection
        InjectionDetector, InjectionResult,
        get_injection_detector, detect_injection,

        # Cost tracking
        CostTracker, TokenUsage, CostConfig,
        get_cost_tracker, track_usage, get_usage_summary,

        # Anomaly detection
        AnomalyDetector, Anomaly, AnomalyType, AnomalyConfig,
        get_anomaly_detector, record_event, check_for_anomalies,

        # Middleware
        setup_governance_middleware, get_user_context,
    )

    # Set up middleware on FastAPI app
    setup_governance_middleware(app)

    # Use in routes
    @app.post("/api/agents/invoke")
    async def invoke_agent(request: Request):
        user = get_user_context(request)
        if not user.has_permission(Permission.AGENT_INVOKE):
            raise HTTPException(403, "Permission denied")
        ...
"""

# RBAC exports
# Anomaly detection exports
from app.governance.anomaly_detector import (
    Anomaly,
    AnomalyBlockedError,
    AnomalyConfig,
    AnomalyDetector,
    AnomalySeverity,
    AnomalyType,
    ContentConfig,
    ErrorConfig,
    Event,
    RateConfig,
    check_for_anomalies,
    get_anomaly_detector,
    record_event,
    reset_anomaly_detector,
)

# Approval workflow exports
from app.governance.approval_workflow import (
    ActionType,
    ApprovalLevel,
    ApprovalRejectedError,
    ApprovalRequest,
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

# Audit logging exports
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

# Cost tracking exports
from app.governance.cost_tracker import (
    BudgetConfig,
    BudgetExceededError,
    CostConfig,
    CostTracker,
    ModelPricing,
    ModelProvider,
    TokenUsage,
    UsageSummary,
    get_cost_tracker,
    get_usage_summary,
    reset_cost_tracker,
    track_usage,
)

# Injection detection exports
from app.governance.injection_detector import (
    InjectionDetector,
    InjectionResult,
    detect_injection,
    get_injection_detector,
    reset_injection_detector,
)

# Middleware exports
from app.governance.middleware import (
    AnomalyMiddleware,
    AuditMiddleware,
    GovernanceContext,
    GovernanceExceptionMiddleware,
    InjectionMiddleware,
    PIIMiddleware,
    RateLimitMiddleware,
    RBACMiddleware,
    create_permission_dependency,
    create_role_dependency,
    get_governance_context,
    get_user_context,
    require_admin,
    require_agent_invoke,
    require_audit_read,
    require_operator,
    require_user,
    setup_governance_middleware,
)

# PII detection exports
from app.governance.pii_detector import (
    PIIAnalysisResult,
    PIIBlockedError,
    PIIConfig,
    PIIDetector,
    PIIMatch,
    PIISeverity,
    PIIType,
    check_for_pii,
    detect_pii,
    get_pii_detector,
    mask_pii,
    reset_pii_detector,
)

# Rate limiting exports
from app.governance.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitExceededError,
    RateLimitResult,
    RateLimitRule,
    check_rate_limit,
    get_rate_limiter,
    require_rate_limit,
    reset_rate_limiter,
)
from app.governance.rbac import (
    ROLE_PERMISSIONS,
    Permission,
    PermissionDeniedError,
    RBACConfig,
    RBACManager,
    Role,
    UserContext,
    check_permission,
    get_permissions_for_role,
    get_rbac_manager,
    require_permission,
    reset_rbac_manager,
)

__all__ = [
    # RBAC
    "Role",
    "Permission",
    "UserContext",
    "RBACConfig",
    "RBACManager",
    "PermissionDeniedError",
    "ROLE_PERMISSIONS",
    "get_rbac_manager",
    "reset_rbac_manager",
    "check_permission",
    "require_permission",
    "get_permissions_for_role",
    # Audit
    "AuditAction",
    "AuditLevel",
    "AuditEntry",
    "AuditConfig",
    "AuditLogger",
    "get_audit_logger",
    "reset_audit_logger",
    "audit_agent_response",
    # Rate limiting
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitRule",
    "RateLimiter",
    "RateLimitExceededError",
    "get_rate_limiter",
    "reset_rate_limiter",
    "check_rate_limit",
    "require_rate_limit",
    # Approval workflow
    "ActionType",
    "ApprovalLevel",
    "ApprovalStatus",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalWorkflowConfig",
    "ApprovalWorkflowManager",
    "ApprovalRequiredError",
    "ApprovalRejectedError",
    "get_approval_manager",
    "reset_approval_manager",
    "requires_approval",
    "get_approval_level",
    "request_approval",
    # Middleware
    "GovernanceContext",
    "RBACMiddleware",
    "RateLimitMiddleware",
    "AuditMiddleware",
    "PIIMiddleware",
    "InjectionMiddleware",
    "AnomalyMiddleware",
    "GovernanceExceptionMiddleware",
    "setup_governance_middleware",
    "get_governance_context",
    "get_user_context",
    "create_permission_dependency",
    "create_role_dependency",
    "require_admin",
    "require_operator",
    "require_user",
    "require_agent_invoke",
    "require_audit_read",
    # PII detection
    "PIIType",
    "PIISeverity",
    "PIIMatch",
    "PIIAnalysisResult",
    "PIIConfig",
    "PIIDetector",
    "PIIBlockedError",
    "get_pii_detector",
    "reset_pii_detector",
    "detect_pii",
    "mask_pii",
    "check_for_pii",
    # Injection detection
    "InjectionDetector",
    "InjectionResult",
    "get_injection_detector",
    "reset_injection_detector",
    "detect_injection",
    # Cost tracking
    "ModelProvider",
    "ModelPricing",
    "TokenUsage",
    "UsageSummary",
    "BudgetConfig",
    "CostConfig",
    "CostTracker",
    "BudgetExceededError",
    "get_cost_tracker",
    "reset_cost_tracker",
    "track_usage",
    "get_usage_summary",
    # Anomaly detection
    "AnomalyType",
    "AnomalySeverity",
    "Event",
    "Anomaly",
    "RateConfig",
    "ErrorConfig",
    "ContentConfig",
    "AnomalyConfig",
    "AnomalyDetector",
    "AnomalyBlockedError",
    "get_anomaly_detector",
    "reset_anomaly_detector",
    "record_event",
    "check_for_anomalies",
]
