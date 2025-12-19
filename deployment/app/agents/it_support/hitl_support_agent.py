"""Human-in-the-Loop IT Support Agent.

This agent handles IT support with human oversight:
- Ticket creation and management
- Escalation workflows
- Human approval for sensitive actions
- Resolution tracking

Following Enterprise Development Standards:
- Software Architect: Interrupt-based HITL pattern
- Security Architect: Approval gates for sensitive ops
- Data Architect: Ticket state management
- Software Engineer: Robust error handling
"""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.agent_base import BaseAgent, AgentConfig
from app.agents.base.tools import tool_error_handler


class ITSupportState(BaseModel):
    """State schema for the HITL IT Support Agent."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation history"
    )
    session_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    ticket_id: str | None = Field(default=None, description="Current ticket ID")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Ticket priority"
    )
    category: str = Field(default="", description="Issue category")
    status: Literal["new", "in_progress", "pending_approval", "escalated", "resolved"] = Field(
        default="new",
        description="Ticket status"
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether action requires human approval"
    )
    approved_by: str | None = Field(default=None, description="Approver name")
    resolution: str | None = Field(default=None, description="Resolution details")


# Simulated ticket store
_tickets: dict[str, dict[str, Any]] = {}

# Actions requiring approval
SENSITIVE_ACTIONS = [
    "password_reset",
    "access_grant",
    "system_change",
    "data_deletion",
    "admin_access",
]


@tool
@tool_error_handler
def create_ticket(
    description: str,
    category: str,
    priority: str = "medium",
    user_email: str = ""
) -> str:
    """Create a new IT support ticket.

    Args:
        description: Issue description
        category: Issue category
        priority: Priority level (low/medium/high/critical)
        user_email: User's email address

    Returns:
        Ticket creation confirmation with ID
    """
    ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"

    _tickets[ticket_id] = {
        "id": ticket_id,
        "description": description,
        "category": category,
        "priority": priority,
        "user_email": user_email,
        "status": "new",
        "created_at": datetime.now().isoformat(),
        "history": [{"action": "created", "timestamp": datetime.now().isoformat()}],
    }

    return (
        f"Ticket created successfully!\n\n"
        f"Ticket ID: {ticket_id}\n"
        f"Category: {category}\n"
        f"Priority: {priority}\n"
        f"Status: New\n\n"
        f"Description: {description}"
    )


@tool
@tool_error_handler
def get_ticket_status(ticket_id: str) -> str:
    """Get the current status of a ticket.

    Args:
        ticket_id: Ticket ID to check

    Returns:
        Ticket status and details
    """
    if ticket_id not in _tickets:
        return f"Ticket {ticket_id} not found."

    ticket = _tickets[ticket_id]
    return (
        f"Ticket: {ticket_id}\n"
        f"Status: {ticket['status']}\n"
        f"Priority: {ticket['priority']}\n"
        f"Category: {ticket['category']}\n"
        f"Created: {ticket['created_at']}\n\n"
        f"Description: {ticket['description']}"
    )


@tool
@tool_error_handler
def update_ticket(ticket_id: str, status: str, notes: str = "") -> str:
    """Update a ticket's status.

    Args:
        ticket_id: Ticket ID to update
        status: New status
        notes: Update notes

    Returns:
        Update confirmation
    """
    if ticket_id not in _tickets:
        return f"Ticket {ticket_id} not found."

    _tickets[ticket_id]["status"] = status
    _tickets[ticket_id]["history"].append({
        "action": f"status_changed_to_{status}",
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    })

    return f"Ticket {ticket_id} updated to status: {status}"


@tool
@tool_error_handler
def check_action_approval(action: str) -> str:
    """Check if an action requires human approval.

    Args:
        action: Action to check

    Returns:
        Approval requirement status
    """
    requires_approval = any(
        sensitive in action.lower()
        for sensitive in SENSITIVE_ACTIONS
    )

    if requires_approval:
        return (
            f"Action '{action}' REQUIRES HUMAN APPROVAL\n"
            f"Reason: This is a sensitive action that could affect security or data.\n"
            f"Next step: Request approval from supervisor before proceeding."
        )
    else:
        return f"Action '{action}' can proceed without additional approval."


@tool
@tool_error_handler
def escalate_ticket(ticket_id: str, reason: str, escalation_level: str = "L2") -> str:
    """Escalate a ticket to higher support level.

    Args:
        ticket_id: Ticket to escalate
        reason: Reason for escalation
        escalation_level: Target level (L2/L3/Management)

    Returns:
        Escalation confirmation
    """
    if ticket_id not in _tickets:
        return f"Ticket {ticket_id} not found."

    _tickets[ticket_id]["status"] = "escalated"
    _tickets[ticket_id]["escalation_level"] = escalation_level
    _tickets[ticket_id]["history"].append({
        "action": f"escalated_to_{escalation_level}",
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    })

    return (
        f"Ticket {ticket_id} escalated to {escalation_level}\n"
        f"Reason: {reason}\n"
        f"A specialist will review this ticket shortly."
    )


@tool
@tool_error_handler
def resolve_ticket(ticket_id: str, resolution: str) -> str:
    """Resolve a ticket with resolution details.

    Args:
        ticket_id: Ticket to resolve
        resolution: Resolution description

    Returns:
        Resolution confirmation
    """
    if ticket_id not in _tickets:
        return f"Ticket {ticket_id} not found."

    _tickets[ticket_id]["status"] = "resolved"
    _tickets[ticket_id]["resolution"] = resolution
    _tickets[ticket_id]["resolved_at"] = datetime.now().isoformat()
    _tickets[ticket_id]["history"].append({
        "action": "resolved",
        "resolution": resolution,
        "timestamp": datetime.now().isoformat(),
    })

    return (
        f"Ticket {ticket_id} has been RESOLVED\n\n"
        f"Resolution: {resolution}\n"
        f"Resolved at: {_tickets[ticket_id]['resolved_at']}"
    )


class HITLSupportAgent(BaseAgent):
    """Human-in-the-Loop IT Support Agent.

    Features:
    - Ticket management
    - Sensitive action approval gates
    - Escalation workflow
    - Resolution tracking

    Example:
        >>> agent = HITLSupportAgent()
        >>> result = agent.handle_issue("I can't access my email")
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the HITL Support Agent."""
        super().__init__(config)

        self.register_tools([
            create_ticket,
            get_ticket_status,
            update_ticket,
            check_action_approval,
            escalate_ticket,
            resolve_ticket,
        ])

    def _get_system_prompt(self) -> str:
        """Get the support agent's system prompt."""
        return """You are an IT Support Agent with human oversight capabilities.
Your role is to help resolve IT issues while ensuring sensitive actions
get proper approval.

## Your Capabilities:
1. **Tickets**: Create, update, and resolve tickets
2. **Approval Check**: Verify if actions need approval
3. **Escalation**: Escalate complex issues
4. **Resolution**: Document resolutions

## Important Guidelines:

### Actions Requiring Approval:
- Password resets
- Access grants/revocations
- System configuration changes
- Data deletion
- Admin access requests

For these actions, ALWAYS:
1. Check if approval is needed (check_action_approval)
2. If yes, pause and request human approval
3. Only proceed after approval is granted

### Process:
1. Understand the user's issue
2. Create a ticket if not already created
3. Check if proposed actions need approval
4. For sensitive actions, pause for approval
5. Execute approved actions
6. Document resolution

### Communication:
- Be empathetic and professional
- Explain what actions you're taking
- Be clear about why approval is needed
- Provide regular status updates

Always prioritize security over convenience."""

    def _build_graph(self) -> StateGraph:
        """Build the HITL support agent's workflow graph."""

        def triage_issue(state: ITSupportState) -> dict:
            """Initial triage of the issue."""
            system_prompt = SystemMessage(content=self._get_system_prompt())
            messages = [system_prompt] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def approval_gate(state: ITSupportState) -> dict:
            """Human approval checkpoint."""
            # Check if we need approval
            requires_approval = getattr(state, "requires_approval", False)
            if not requires_approval:
                return {}

            ticket_id = getattr(state, "ticket_id", None)
            approval = interrupt({
                "type": "approval_request",
                "ticket_id": ticket_id,
                "action": "Sensitive action requires supervisor approval",
                "prompt": "Please approve or deny this action. "
                         "Reply 'approve' or 'deny' with optional notes."
            })

            if approval.lower().startswith("approve"):
                return {
                    "approved_by": "supervisor",
                    "requires_approval": False,
                    "status": "in_progress",
                }
            else:
                return {
                    "status": "pending_approval",
                    "messages": [AIMessage(content=f"Action denied: {approval}")],
                }

        def should_continue(state: ITSupportState) -> str:
            messages = list(state.messages)
            if not messages:
                return "triage"

            last_message = messages[-1]

            # Check for tool calls
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                # Check if any tool call is for sensitive action
                for tool_call in last_message.tool_calls:
                    if any(s in tool_call.get("name", "").lower() for s in ["reset", "grant", "delete"]):
                        return "approval"
                return "tools"

            return "end"

        def check_approval_needed(state: ITSupportState) -> str:
            requires_approval = getattr(state, "requires_approval", False)
            approved_by = getattr(state, "approved_by", None)
            if requires_approval and not approved_by:
                return "approval"
            return "continue"

        graph = StateGraph(ITSupportState)

        graph.add_node("triage", triage_issue)
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_node("approval", approval_gate)

        graph.add_edge(START, "triage")
        graph.add_conditional_edges(
            "triage",
            should_continue,
            {"tools": "tools", "approval": "approval", "end": END}
        )
        graph.add_edge("tools", "triage")
        graph.add_conditional_edges(
            "approval",
            check_approval_needed,
            {"approval": "approval", "continue": "triage"}
        )

        return graph

    @traceable(name="support_handle_issue")
    def handle_issue(
        self,
        description: str,
        user_email: str = "",
        priority: str = "medium",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle an IT support issue.

        Args:
            description: Issue description
            user_email: User's email
            priority: Issue priority
            session_id: Optional session ID

        Returns:
            Support interaction result
        """
        return self.invoke(
            message=f"I need help with: {description}",
            session_id=session_id,
            priority=priority,
            user_id=user_email,
        )
