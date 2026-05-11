"""Approval workflow for enterprise IT agents.

Provides:
- Multi-level approval workflows
- Approval request management
- Integration with HITL patterns
- Configurable approval rules
- Approval audit trail
"""

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from app.governance.rbac import Permission, Role, UserContext


class ApprovalLevel(str, Enum):
    """Approval levels for different action types."""

    L1 = "l1"  # Standard actions - single approver
    L2 = "l2"  # Elevated actions - senior approver
    L3 = "l3"  # Critical actions - admin approval


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ActionType(str, Enum):
    """Types of actions that may require approval."""

    # ServiceNow actions
    CREATE_INCIDENT = "servicenow:create_incident"
    UPDATE_INCIDENT = "servicenow:update_incident"
    CLOSE_INCIDENT = "servicenow:close_incident"
    CREATE_CHANGE = "servicenow:create_change"
    APPROVE_CHANGE = "servicenow:approve_change"

    # IT Support actions
    PASSWORD_RESET = "it:password_reset"
    ACCESS_GRANT = "it:access_grant"
    ACCESS_REVOKE = "it:access_revoke"
    SYSTEM_RESTART = "it:system_restart"

    # Document actions
    DOCUMENT_DELETE = "doc:delete"
    DOCUMENT_SHARE = "doc:share"

    # Agent actions
    AGENT_CONFIG_CHANGE = "agent:config_change"
    BULK_OPERATION = "agent:bulk_operation"


# Action to approval level mapping
ACTION_APPROVAL_LEVELS: dict[ActionType, ApprovalLevel] = {
    # L1 - Standard operations
    ActionType.CREATE_INCIDENT: ApprovalLevel.L1,
    ActionType.UPDATE_INCIDENT: ApprovalLevel.L1,
    ActionType.DOCUMENT_SHARE: ApprovalLevel.L1,
    # L2 - Elevated operations
    ActionType.CLOSE_INCIDENT: ApprovalLevel.L2,
    ActionType.PASSWORD_RESET: ApprovalLevel.L2,
    ActionType.ACCESS_GRANT: ApprovalLevel.L2,
    ActionType.CREATE_CHANGE: ApprovalLevel.L2,
    # L3 - Critical operations
    ActionType.ACCESS_REVOKE: ApprovalLevel.L3,
    ActionType.SYSTEM_RESTART: ApprovalLevel.L3,
    ActionType.APPROVE_CHANGE: ApprovalLevel.L3,
    ActionType.DOCUMENT_DELETE: ApprovalLevel.L3,
    ActionType.AGENT_CONFIG_CHANGE: ApprovalLevel.L3,
    ActionType.BULK_OPERATION: ApprovalLevel.L3,
}

# Permission required to approve at each level
APPROVAL_PERMISSIONS: dict[ApprovalLevel, Permission] = {
    ApprovalLevel.L1: Permission.APPROVE_L1,
    ApprovalLevel.L2: Permission.APPROVE_L2,
    ApprovalLevel.L3: Permission.APPROVE_L3,
}


@dataclass
class ApprovalRequest:
    """A request for approval of an action.

    Attributes:
        id: Unique request identifier
        action_type: Type of action requiring approval
        level: Approval level required
        requester_id: User ID of the requester
        agent_type: Agent requesting approval
        action_details: Details of the action to be approved
        status: Current status of the request
        created_at: Request creation timestamp
        expires_at: Request expiration timestamp
        approver_id: User ID of the approver (if approved/rejected)
        approved_at: Approval timestamp
        rejection_reason: Reason for rejection (if rejected)
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    action_type: ActionType = ActionType.CREATE_INCIDENT
    level: ApprovalLevel = ApprovalLevel.L1
    requester_id: str = ""
    agent_type: str = ""
    action_details: dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str = ""
    approver_id: str | None = None
    approved_at: str | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default expiration if not provided."""
        if not self.expires_at:
            expiry = datetime.now(timezone.utc) + timedelta(hours=24)
            self.expires_at = expiry.isoformat()

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > expiry

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "action_type": self.action_type.value,
            "level": self.level.value,
            "requester_id": self.requester_id,
            "agent_type": self.agent_type,
            "action_details": self.action_details,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "approver_id": self.approver_id,
            "approved_at": self.approved_at,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }


@dataclass
class ApprovalResponse:
    """Response to an approval request.

    Attributes:
        request_id: ID of the approval request
        approved: Whether the request was approved
        approver_id: User ID of the approver
        reason: Reason for approval/rejection
        modifications: Any modifications to the original action
    """

    request_id: str
    approved: bool
    approver_id: str
    reason: str | None = None
    modifications: dict[str, Any] | None = None


@dataclass
class ApprovalWorkflowConfig:
    """Configuration for approval workflows.

    Attributes:
        enabled: Whether approval workflows are enabled
        auto_approve_l1: Whether to auto-approve L1 actions
        default_expiry_hours: Default expiry time for requests
        require_reason_on_reject: Whether to require a reason for rejection
        notify_on_pending: Whether to send notifications for pending requests
        max_pending_per_user: Maximum pending requests per user
    """

    enabled: bool = True
    auto_approve_l1: bool = False
    default_expiry_hours: int = 24
    require_reason_on_reject: bool = True
    notify_on_pending: bool = True
    max_pending_per_user: int = 10

    @classmethod
    def from_env(cls) -> "ApprovalWorkflowConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("APPROVAL_WORKFLOW_ENABLED", "true").lower() == "true",
            auto_approve_l1=os.getenv("APPROVAL_AUTO_APPROVE_L1", "false").lower() == "true",
            default_expiry_hours=int(os.getenv("APPROVAL_EXPIRY_HOURS", "24")),
            require_reason_on_reject=os.getenv("APPROVAL_REQUIRE_REJECT_REASON", "true").lower() == "true",
            notify_on_pending=os.getenv("APPROVAL_NOTIFY_PENDING", "true").lower() == "true",
            max_pending_per_user=int(os.getenv("APPROVAL_MAX_PENDING", "10")),
        )


# Type for approval callbacks
ApprovalCallback = Callable[[ApprovalRequest, ApprovalResponse], None]


class ApprovalWorkflowManager:
    """Manages approval workflows for agent actions.

    Provides:
    - Request creation and tracking
    - Approval/rejection handling
    - Expiration management
    - Callback notifications
    """

    def __init__(self, config: ApprovalWorkflowConfig | None = None) -> None:
        """Initialize approval workflow manager.

        Args:
            config: Workflow configuration.
        """
        self.config = config or ApprovalWorkflowConfig.from_env()
        self._requests: dict[str, ApprovalRequest] = {}
        self._callbacks: list[ApprovalCallback] = []
        self._user_pending_count: dict[str, int] = {}

    def register_callback(self, callback: ApprovalCallback) -> None:
        """Register a callback for approval events.

        Args:
            callback: Callback function to invoke on approval events.
        """
        self._callbacks.append(callback)

    def _notify_callbacks(
        self,
        request: ApprovalRequest,
        response: ApprovalResponse,
    ) -> None:
        """Notify all registered callbacks.

        Args:
            request: The approval request.
            response: The approval response.
        """
        for callback in self._callbacks:
            try:
                callback(request, response)
            except Exception:
                pass  # Don't let callback failures affect workflow

    def create_request(
        self,
        action_type: ActionType,
        requester_id: str,
        agent_type: str,
        action_details: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        """Create an approval request.

        Args:
            action_type: Type of action requiring approval.
            requester_id: User ID of the requester.
            agent_type: Agent type requesting approval.
            action_details: Details of the action.
            metadata: Additional metadata.

        Returns:
            Created approval request.

        Raises:
            ValueError: If max pending requests exceeded.
        """
        # Check pending limit
        current_pending = self._user_pending_count.get(requester_id, 0)
        if current_pending >= self.config.max_pending_per_user:
            msg = f"Maximum pending requests ({self.config.max_pending_per_user}) exceeded"
            raise ValueError(msg)

        level = ACTION_APPROVAL_LEVELS.get(action_type, ApprovalLevel.L1)

        request = ApprovalRequest(
            action_type=action_type,
            level=level,
            requester_id=requester_id,
            agent_type=agent_type,
            action_details=action_details,
            metadata=metadata or {},
        )

        self._requests[request.id] = request
        self._user_pending_count[requester_id] = current_pending + 1

        return request

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID.

        Args:
            request_id: Request identifier.

        Returns:
            Approval request or None if not found.
        """
        request = self._requests.get(request_id)
        if request and request.is_expired() and request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.EXPIRED
        return request

    def list_pending(
        self,
        requester_id: str | None = None,
        level: ApprovalLevel | None = None,
    ) -> list[ApprovalRequest]:
        """List pending approval requests.

        Args:
            requester_id: Filter by requester (optional).
            level: Filter by approval level (optional).

        Returns:
            List of pending requests.
        """
        pending = []
        for request in self._requests.values():
            # Check expiration
            if request.is_expired() and request.status == ApprovalStatus.PENDING:
                request.status = ApprovalStatus.EXPIRED
                continue

            if request.status != ApprovalStatus.PENDING:
                continue

            if requester_id and request.requester_id != requester_id:
                continue

            if level and request.level != level:
                continue

            pending.append(request)

        return pending

    def can_approve(
        self,
        user_context: UserContext,
        request: ApprovalRequest,
    ) -> bool:
        """Check if a user can approve a request.

        Args:
            user_context: User context of the approver.
            request: Request to check.

        Returns:
            True if user can approve.
        """
        # Admin can approve anything
        if user_context.role == Role.ADMIN:
            return True

        # Check level-specific permission
        required_permission = APPROVAL_PERMISSIONS.get(request.level)
        if required_permission:
            return user_context.has_permission(required_permission)

        return False

    def approve(
        self,
        request_id: str,
        approver: UserContext,
        reason: str | None = None,
        modifications: dict[str, Any] | None = None,
    ) -> ApprovalResponse:
        """Approve a request.

        Args:
            request_id: Request identifier.
            approver: User context of the approver.
            reason: Reason for approval.
            modifications: Any modifications to the action.

        Returns:
            Approval response.

        Raises:
            ValueError: If request not found, expired, or already processed.
            PermissionError: If user cannot approve.
        """
        request = self.get_request(request_id)
        if not request:
            msg = f"Request {request_id} not found"
            raise ValueError(msg)

        if request.status != ApprovalStatus.PENDING:
            msg = f"Request {request_id} is not pending (status: {request.status.value})"
            raise ValueError(msg)

        if not self.can_approve(approver, request):
            msg = f"User {approver.user_id} cannot approve {request.level.value} requests"
            raise PermissionError(msg)

        # Update request
        request.status = ApprovalStatus.APPROVED
        request.approver_id = approver.user_id
        request.approved_at = datetime.now(timezone.utc).isoformat()

        # Update pending count
        self._user_pending_count[request.requester_id] = max(
            0, self._user_pending_count.get(request.requester_id, 1) - 1
        )

        response = ApprovalResponse(
            request_id=request_id,
            approved=True,
            approver_id=approver.user_id,
            reason=reason,
            modifications=modifications,
        )

        self._notify_callbacks(request, response)
        return response

    def reject(
        self,
        request_id: str,
        approver: UserContext,
        reason: str | None = None,
    ) -> ApprovalResponse:
        """Reject a request.

        Args:
            request_id: Request identifier.
            approver: User context of the approver.
            reason: Reason for rejection.

        Returns:
            Approval response.

        Raises:
            ValueError: If request not found, expired, or already processed.
            PermissionError: If user cannot reject.
        """
        request = self.get_request(request_id)
        if not request:
            msg = f"Request {request_id} not found"
            raise ValueError(msg)

        if request.status != ApprovalStatus.PENDING:
            msg = f"Request {request_id} is not pending (status: {request.status.value})"
            raise ValueError(msg)

        if not self.can_approve(approver, request):
            msg = f"User {approver.user_id} cannot reject {request.level.value} requests"
            raise PermissionError(msg)

        if self.config.require_reason_on_reject and not reason:
            msg = "Rejection reason is required"
            raise ValueError(msg)

        # Update request
        request.status = ApprovalStatus.REJECTED
        request.approver_id = approver.user_id
        request.approved_at = datetime.now(timezone.utc).isoformat()
        request.rejection_reason = reason

        # Update pending count
        self._user_pending_count[request.requester_id] = max(
            0, self._user_pending_count.get(request.requester_id, 1) - 1
        )

        response = ApprovalResponse(
            request_id=request_id,
            approved=False,
            approver_id=approver.user_id,
            reason=reason,
        )

        self._notify_callbacks(request, response)
        return response

    def cancel(self, request_id: str, requester_id: str) -> bool:
        """Cancel a pending request.

        Args:
            request_id: Request identifier.
            requester_id: User ID of the original requester.

        Returns:
            True if cancelled successfully.

        Raises:
            ValueError: If request not found or not pending.
            PermissionError: If user is not the original requester.
        """
        request = self.get_request(request_id)
        if not request:
            msg = f"Request {request_id} not found"
            raise ValueError(msg)

        if request.status != ApprovalStatus.PENDING:
            msg = f"Request {request_id} is not pending"
            raise ValueError(msg)

        if request.requester_id != requester_id:
            msg = "Only the original requester can cancel"
            raise PermissionError(msg)

        request.status = ApprovalStatus.CANCELLED

        # Update pending count
        self._user_pending_count[request.requester_id] = max(
            0, self._user_pending_count.get(request.requester_id, 1) - 1
        )

        return True

    def cleanup_expired(self) -> int:
        """Clean up expired requests.

        Returns:
            Number of requests cleaned up.
        """
        expired_ids = []
        for request_id, request in self._requests.items():
            if request.is_expired() and request.status == ApprovalStatus.PENDING:
                request.status = ApprovalStatus.EXPIRED
                expired_ids.append(request_id)

        return len(expired_ids)

    async def wait_for_approval(
        self,
        request_id: str,
        timeout_seconds: float = 300.0,
        poll_interval: float = 1.0,
    ) -> ApprovalRequest:
        """Wait for a request to be approved or rejected.

        Args:
            request_id: Request identifier.
            timeout_seconds: Maximum wait time.
            poll_interval: Polling interval.

        Returns:
            Updated approval request.

        Raises:
            TimeoutError: If approval times out.
            ValueError: If request not found.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            request = self.get_request(request_id)
            if not request:
                msg = f"Request {request_id} not found"
                raise ValueError(msg)

            if request.status != ApprovalStatus.PENDING:
                return request

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout_seconds:
                msg = f"Approval request {request_id} timed out"
                raise TimeoutError(msg)

            await asyncio.sleep(poll_interval)


# Global approval workflow manager instance
_approval_manager: ApprovalWorkflowManager | None = None


def get_approval_manager() -> ApprovalWorkflowManager:
    """Get or create the global approval workflow manager."""
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalWorkflowManager()
    return _approval_manager


def reset_approval_manager() -> None:
    """Reset the global approval workflow manager."""
    global _approval_manager
    _approval_manager = None


# Convenience functions


def requires_approval(action_type: ActionType) -> bool:
    """Check if an action type requires approval.

    Args:
        action_type: Action type to check.

    Returns:
        True if approval is required.
    """
    manager = get_approval_manager()
    if not manager.config.enabled:
        return False

    level = ACTION_APPROVAL_LEVELS.get(action_type)
    if not level:
        return False

    # Auto-approve L1 if configured
    if level == ApprovalLevel.L1 and manager.config.auto_approve_l1:
        return False

    return True


def get_approval_level(action_type: ActionType) -> ApprovalLevel | None:
    """Get the approval level for an action type.

    Args:
        action_type: Action type.

    Returns:
        Approval level or None if no approval needed.
    """
    return ACTION_APPROVAL_LEVELS.get(action_type)


async def request_approval(
    action_type: ActionType,
    requester_id: str,
    agent_type: str,
    action_details: dict[str, Any],
    wait: bool = False,
    timeout_seconds: float = 300.0,
) -> ApprovalRequest:
    """Request approval for an action.

    Args:
        action_type: Type of action.
        requester_id: User ID of the requester.
        agent_type: Agent requesting approval.
        action_details: Details of the action.
        wait: Whether to wait for approval.
        timeout_seconds: Wait timeout (if wait=True).

    Returns:
        Approval request (with final status if wait=True).
    """
    manager = get_approval_manager()
    request = manager.create_request(
        action_type=action_type,
        requester_id=requester_id,
        agent_type=agent_type,
        action_details=action_details,
    )

    if wait:
        return await manager.wait_for_approval(request.id, timeout_seconds)

    return request


class ApprovalRequiredError(Exception):
    """Raised when an action requires approval."""

    def __init__(
        self,
        message: str,
        request: ApprovalRequest,
    ) -> None:
        """Initialize error.

        Args:
            message: Error message.
            request: The approval request that was created.
        """
        super().__init__(message)
        self.request = request


class ApprovalRejectedError(Exception):
    """Raised when an approval request is rejected."""

    def __init__(
        self,
        message: str,
        request: ApprovalRequest,
    ) -> None:
        """Initialize error.

        Args:
            message: Error message.
            request: The rejected approval request.
        """
        super().__init__(message)
        self.request = request
        self.reason = request.rejection_reason
