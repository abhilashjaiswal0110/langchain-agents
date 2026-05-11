"""Escalation Handler for cross-domain and human escalations.

Manages escalation workflows:
- Cross-domain escalations when request spans multiple domains
- Human-in-the-loop for sensitive or complex issues
- Priority-based escalation routing
- Escalation tracking and resolution

Following Enterprise Development Standards:
- Software Architect: HITL integration, workflow management
- Security Architect: Sensitive data handling, audit trails
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langsmith import traceable


class EscalationLevel(str, Enum):
    """Escalation priority levels."""

    LOW = "low"  # General questions, non-urgent
    MEDIUM = "medium"  # Standard support, moderate urgency
    HIGH = "high"  # Important issues, needs quick resolution
    CRITICAL = "critical"  # Security incidents, business critical


class EscalationReason(str, Enum):
    """Reasons for escalation."""

    CROSS_DOMAIN = "cross_domain"  # Request spans multiple domains
    HUMAN_REQUESTED = "human_requested"  # User requested human help
    AGENT_UNCERTAIN = "agent_uncertain"  # Agent unsure how to proceed
    SENSITIVE_DATA = "sensitive_data"  # Involves sensitive information
    SECURITY_INCIDENT = "security_incident"  # Security-related issue
    APPROVAL_REQUIRED = "approval_required"  # Action needs approval
    COMPLEX_ISSUE = "complex_issue"  # Too complex for automation
    REPEATED_FAILURE = "repeated_failure"  # Multiple failed attempts


class EscalationStatus(str, Enum):
    """Status of an escalation."""

    PENDING = "pending"  # Waiting for assignment
    ASSIGNED = "assigned"  # Assigned to handler
    IN_PROGRESS = "in_progress"  # Being worked on
    RESOLVED = "resolved"  # Successfully resolved
    CLOSED = "closed"  # Closed without resolution


@dataclass
class EscalationRequest:
    """Request for escalation.

    Attributes:
        id: Unique escalation identifier
        level: Priority level
        reason: Reason for escalation
        domains_involved: List of domains involved
        user_context: User who initiated the request
        messages: Conversation history
        summary: Brief summary of the issue
        metadata: Additional context
        created_at: When escalation was created
        status: Current status
        assigned_to: Handler assigned (if any)
        resolution: Resolution details (if resolved)
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    level: EscalationLevel = EscalationLevel.MEDIUM
    reason: EscalationReason = EscalationReason.AGENT_UNCERTAIN
    domains_involved: list[str] = field(default_factory=list)
    user_context: dict[str, Any] = field(default_factory=dict)
    messages: list[BaseMessage] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: EscalationStatus = EscalationStatus.PENDING
    assigned_to: str | None = None
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "level": self.level.value,
            "reason": self.reason.value,
            "domains_involved": self.domains_involved,
            "user_context": self.user_context,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "resolution": self.resolution,
        }


@dataclass
class EscalationResponse:
    """Response from escalation handling.

    Attributes:
        request_id: ID of the escalation request
        success: Whether escalation was successful
        message: Message to user
        next_steps: Suggested next steps
        estimated_response: Estimated time for response
    """

    request_id: str
    success: bool
    message: str
    next_steps: list[str] = field(default_factory=list)
    estimated_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "message": self.message,
            "next_steps": self.next_steps,
            "estimated_response": self.estimated_response,
        }


class EscalationHandler:
    """Handle escalations for the IT Supervisor.

    Manages:
    - Creating escalation requests
    - Routing to appropriate handlers
    - Tracking escalation status
    - Notifying relevant parties

    Example:
        >>> handler = EscalationHandler()
        >>> request = handler.create_escalation(
        ...     level=EscalationLevel.HIGH,
        ...     reason=EscalationReason.SECURITY_INCIDENT,
        ...     summary="Potential phishing attack detected",
        ...     user_context=user.to_dict(),
        ... )
        >>> response = await handler.process_escalation(request)
    """

    # Level to response time mapping
    RESPONSE_TIMES = {
        EscalationLevel.CRITICAL: "15 minutes",
        EscalationLevel.HIGH: "1 hour",
        EscalationLevel.MEDIUM: "4 hours",
        EscalationLevel.LOW: "24 hours",
    }

    # Reason to level mapping (default levels)
    REASON_LEVELS = {
        EscalationReason.SECURITY_INCIDENT: EscalationLevel.CRITICAL,
        EscalationReason.SENSITIVE_DATA: EscalationLevel.HIGH,
        EscalationReason.APPROVAL_REQUIRED: EscalationLevel.MEDIUM,
        EscalationReason.CROSS_DOMAIN: EscalationLevel.MEDIUM,
        EscalationReason.HUMAN_REQUESTED: EscalationLevel.MEDIUM,
        EscalationReason.COMPLEX_ISSUE: EscalationLevel.MEDIUM,
        EscalationReason.AGENT_UNCERTAIN: EscalationLevel.LOW,
        EscalationReason.REPEATED_FAILURE: EscalationLevel.LOW,
    }

    def __init__(self) -> None:
        """Initialize escalation handler."""
        # In-memory storage (replace with database in production)
        self._pending_escalations: dict[str, EscalationRequest] = {}
        self._resolved_escalations: dict[str, EscalationRequest] = {}

    def create_escalation(
        self,
        reason: EscalationReason,
        summary: str,
        user_context: dict[str, Any] | None = None,
        messages: list[BaseMessage] | None = None,
        domains_involved: list[str] | None = None,
        level: EscalationLevel | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EscalationRequest:
        """Create a new escalation request.

        Args:
            reason: Reason for escalation.
            summary: Brief summary of the issue.
            user_context: User context dictionary.
            messages: Conversation history.
            domains_involved: List of domains involved.
            level: Override automatic level assignment.
            metadata: Additional metadata.

        Returns:
            EscalationRequest instance.
        """
        # Determine level if not provided
        if level is None:
            level = self.REASON_LEVELS.get(reason, EscalationLevel.MEDIUM)

        request = EscalationRequest(
            level=level,
            reason=reason,
            summary=summary,
            user_context=user_context or {},
            messages=messages or [],
            domains_involved=domains_involved or [],
            metadata=metadata or {},
        )

        self._pending_escalations[request.id] = request

        return request

    @traceable(name="process_escalation")
    async def process_escalation(
        self,
        request: EscalationRequest,
    ) -> EscalationResponse:
        """Process an escalation request.

        Args:
            request: Escalation request to process.

        Returns:
            EscalationResponse with next steps.
        """
        # Update status
        request.status = EscalationStatus.PENDING

        # Generate response based on level
        response_time = self.RESPONSE_TIMES.get(request.level, "24 hours")

        # Build user message
        message = self._build_escalation_message(request)

        # Build next steps
        next_steps = self._build_next_steps(request)

        return EscalationResponse(
            request_id=request.id,
            success=True,
            message=message,
            next_steps=next_steps,
            estimated_response=response_time,
        )

    def _build_escalation_message(self, request: EscalationRequest) -> str:
        """Build user-facing escalation message.

        Args:
            request: Escalation request.

        Returns:
            Message to display to user.
        """
        level_messages = {
            EscalationLevel.CRITICAL: (
                "I've escalated this as a CRITICAL issue. A specialist will contact you within 15 minutes."
            ),
            EscalationLevel.HIGH: (
                "I've escalated this as a HIGH priority issue. A specialist will contact you within 1 hour."
            ),
            EscalationLevel.MEDIUM: ("I've escalated this to our support team. You should hear back within 4 hours."),
            EscalationLevel.LOW: ("I've logged your request for follow-up. Our team will respond within 24 hours."),
        }

        base_message = level_messages.get(request.level, level_messages[EscalationLevel.MEDIUM])

        if request.reason == EscalationReason.SECURITY_INCIDENT:
            base_message += " Please do not share any passwords or sensitive information."

        if request.domains_involved:
            domains_str = ", ".join(request.domains_involved)
            base_message += f" This involves the following teams: {domains_str}."

        return f"Escalation Reference: #{request.id}\n\n{base_message}"

    def _build_next_steps(self, request: EscalationRequest) -> list[str]:
        """Build next steps for user.

        Args:
            request: Escalation request.

        Returns:
            List of next steps.
        """
        steps = []

        if request.level in (EscalationLevel.CRITICAL, EscalationLevel.HIGH):
            steps.append("A specialist will contact you directly")
            steps.append("Please keep your phone/email accessible")
        else:
            steps.append("You'll receive an email confirmation shortly")
            steps.append("You can continue using the chat for other questions")

        if request.reason == EscalationReason.SECURITY_INCIDENT:
            steps.append("Do not click any suspicious links")
            steps.append("Do not share your passwords with anyone")

        if request.reason == EscalationReason.APPROVAL_REQUIRED:
            steps.append("Your request has been sent for approval")
            steps.append("You'll be notified once approved")

        steps.append(f"Reference your escalation ID #{request.id} for follow-up")

        return steps

    def get_escalation(self, escalation_id: str) -> EscalationRequest | None:
        """Get escalation by ID.

        Args:
            escalation_id: Escalation ID.

        Returns:
            EscalationRequest or None if not found.
        """
        if escalation_id in self._pending_escalations:
            return self._pending_escalations[escalation_id]
        return self._resolved_escalations.get(escalation_id)

    def resolve_escalation(
        self,
        escalation_id: str,
        resolution: str,
        resolved_by: str,
    ) -> bool:
        """Mark an escalation as resolved.

        Args:
            escalation_id: Escalation ID.
            resolution: Resolution description.
            resolved_by: Handler who resolved it.

        Returns:
            True if resolved, False if not found.
        """
        request = self._pending_escalations.pop(escalation_id, None)
        if request is None:
            return False

        request.status = EscalationStatus.RESOLVED
        request.resolution = resolution
        request.assigned_to = resolved_by

        self._resolved_escalations[escalation_id] = request

        return True

    def get_pending_escalations(
        self,
        level: EscalationLevel | None = None,
    ) -> list[EscalationRequest]:
        """Get all pending escalations.

        Args:
            level: Optional filter by level.

        Returns:
            List of pending escalation requests.
        """
        escalations = list(self._pending_escalations.values())

        if level:
            escalations = [e for e in escalations if e.level == level]

        # Sort by level (critical first) then creation time
        level_priority = {
            EscalationLevel.CRITICAL: 0,
            EscalationLevel.HIGH: 1,
            EscalationLevel.MEDIUM: 2,
            EscalationLevel.LOW: 3,
        }

        escalations.sort(key=lambda e: (level_priority.get(e.level, 4), e.created_at))

        return escalations

    def get_escalation_stats(self) -> dict[str, Any]:
        """Get escalation statistics.

        Returns:
            Dictionary with escalation stats.
        """
        pending = list(self._pending_escalations.values())
        resolved = list(self._resolved_escalations.values())

        by_level = {}
        for level in EscalationLevel:
            by_level[level.value] = len([e for e in pending if e.level == level])

        by_reason = {}
        for reason in EscalationReason:
            by_reason[reason.value] = len([e for e in pending if e.reason == reason])

        return {
            "total_pending": len(pending),
            "total_resolved": len(resolved),
            "by_level": by_level,
            "by_reason": by_reason,
        }


# Singleton instance
_handler: EscalationHandler | None = None


def get_escalation_handler() -> EscalationHandler:
    """Get or create escalation handler singleton.

    Returns:
        EscalationHandler instance.
    """
    global _handler
    if _handler is None:
        _handler = EscalationHandler()
    return _handler
