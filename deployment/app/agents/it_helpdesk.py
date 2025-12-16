"""IT Helpdesk Agent with LangGraph for conversational IT support.

This agent handles common IT support tasks like password resets,
software troubleshooting, hardware issues, and knowledge base searches.
"""

import os
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel


# =============================================================================
# Agent State
# =============================================================================

class AgentState(BaseModel):
    """State for IT Helpdesk Agent."""

    messages: Annotated[list, add_messages]
    ticket_id: str | None = None
    user_email: str | None = None
    issue_category: str | None = None
    resolution_status: str = "open"


# =============================================================================
# IT Helpdesk Tools
# =============================================================================

# Simulated knowledge base for IT support
IT_KNOWLEDGE_BASE = {
    "password_reset": {
        "title": "Password Reset Guide",
        "content": """To reset your password:
1. Go to the Self-Service Password Portal at https://password.company.com
2. Click 'Forgot Password'
3. Enter your employee ID or email
4. Answer your security questions or use MFA
5. Create a new password following the policy:
   - Minimum 12 characters
   - At least one uppercase, lowercase, number, and special character
   - Cannot reuse last 10 passwords
If you're locked out, contact IT Support at ext. 5555.""",
        "category": "access"
    },
    "vpn_setup": {
        "title": "VPN Setup Instructions",
        "content": """To set up VPN access:
1. Download the GlobalProtect client from https://software.company.com/vpn
2. Install and launch the application
3. Enter portal address: vpn.company.com
4. Sign in with your AD credentials
5. Complete MFA verification
6. Select your region for optimal performance
Troubleshooting:
- If connection fails, check your internet connection
- Ensure your firewall allows GlobalProtect
- Clear cache if certificate errors occur""",
        "category": "network"
    },
    "email_outlook": {
        "title": "Outlook Email Configuration",
        "content": """To configure Outlook:
1. Open Outlook and go to File > Account Settings
2. Click 'New' to add an account
3. Enter your company email address
4. Outlook will auto-discover settings
5. Complete authentication with MFA
Common issues:
- Sync problems: Check internet connection
- Calendar not updating: Clear offline items
- Large mailbox: Archive old emails""",
        "category": "email"
    },
    "software_install": {
        "title": "Software Installation Guide",
        "content": """To install approved software:
1. Open the Software Center from Start Menu
2. Browse or search for the application
3. Click 'Install' and wait for completion
4. Restart if prompted
For software not in Software Center:
- Submit a request through ServiceNow
- Include business justification
- Manager approval may be required
Standard approval time: 24-48 hours""",
        "category": "software"
    },
    "printer_setup": {
        "title": "Printer Setup Guide",
        "content": """To add a network printer:
1. Go to Settings > Devices > Printers & Scanners
2. Click 'Add a printer or scanner'
3. Select your floor's printer from the list
4. If not listed, click 'The printer I want isn't listed'
5. Enter the printer path: \\\\printserver\\PrinterName
Common printer names by floor:
- Floor 1: PRINTER-F1-HP
- Floor 2: PRINTER-F2-XEROX
- Floor 3: PRINTER-F3-HP""",
        "category": "hardware"
    },
    "teams_issues": {
        "title": "Microsoft Teams Troubleshooting",
        "content": """Common Teams issues and solutions:
1. Audio/Video not working:
   - Check device settings in Teams
   - Update audio/video drivers
   - Restart Teams
2. Screen sharing not working:
   - Grant screen recording permission
   - Disable GPU hardware acceleration
3. Teams running slow:
   - Clear Teams cache
   - Disable animations
   - Check network bandwidth""",
        "category": "software"
    }
}

# Simulated ticket database
TICKET_DATABASE: dict[str, dict[str, Any]] = {}


@tool
def search_knowledge_base(query: str, category: str | None = None) -> str:
    """Search the IT knowledge base for solutions to common problems.

    Args:
        query: Search query describing the issue.
        category: Optional category filter (access, network, email, software, hardware).

    Returns:
        Relevant knowledge base articles.
    """
    query_lower = query.lower()
    results = []

    for key, article in IT_KNOWLEDGE_BASE.items():
        # Filter by category if specified
        if category and article["category"] != category.lower():
            continue

        # Simple keyword matching
        if any(word in key for word in query_lower.split()):
            results.append(f"**{article['title']}**\n{article['content']}")
        elif any(word in article["content"].lower() for word in query_lower.split()):
            results.append(f"**{article['title']}**\n{article['content']}")

    if results:
        return "\n\n---\n\n".join(results[:2])  # Return top 2 matches
    return "No matching articles found. Please provide more details about your issue."


@tool
def create_support_ticket(
    title: str,
    description: str,
    priority: Literal["low", "medium", "high", "critical"] = "medium",
    category: str = "general",
    user_email: str | None = None,
) -> str:
    """Create a new IT support ticket.

    Args:
        title: Brief title of the issue.
        description: Detailed description of the problem.
        priority: Ticket priority (low, medium, high, critical).
        category: Issue category (hardware, software, network, access, email).
        user_email: User's email address for updates.

    Returns:
        Ticket confirmation with ID and details.
    """
    ticket_id = f"INC{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"

    ticket = {
        "id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "category": category,
        "user_email": user_email,
        "status": "new",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "assigned_to": None,
        "resolution": None,
    }

    TICKET_DATABASE[ticket_id] = ticket

    return f"""Ticket created successfully!

**Ticket ID:** {ticket_id}
**Title:** {title}
**Priority:** {priority.upper()}
**Category:** {category}
**Status:** New

You will receive updates at {user_email or 'your registered email'}.
Expected response time based on priority:
- Critical: 1 hour
- High: 4 hours
- Medium: 8 hours
- Low: 24 hours"""


@tool
def check_ticket_status(ticket_id: str) -> str:
    """Check the status of an existing support ticket.

    Args:
        ticket_id: The ticket ID to check (e.g., INC20241215ABC123).

    Returns:
        Current ticket status and details.
    """
    ticket = TICKET_DATABASE.get(ticket_id)

    if not ticket:
        # Simulate some existing tickets
        return f"""Ticket {ticket_id} not found in current session.

If this is an existing ticket, please verify:
1. The ticket ID is correct (format: INC + date + 6 characters)
2. The ticket was created in ServiceNow

To check tickets in ServiceNow:
- Go to https://servicenow.company.com
- Navigate to My Tickets
- Search by ticket ID"""

    return f"""**Ticket Status: {ticket_id}**

**Title:** {ticket['title']}
**Status:** {ticket['status'].upper()}
**Priority:** {ticket['priority'].upper()}
**Category:** {ticket['category']}
**Created:** {ticket['created_at']}
**Last Updated:** {ticket['updated_at']}
**Assigned To:** {ticket['assigned_to'] or 'Pending assignment'}
**Resolution:** {ticket['resolution'] or 'In progress'}"""


@tool
def check_system_status(system: str | None = None) -> str:
    """Check the status of IT systems and services.

    Args:
        system: Specific system to check (email, vpn, network, erp, crm).

    Returns:
        Current system status and any known issues.
    """
    # Simulated system status
    systems = {
        "email": {"status": "operational", "message": "All email services running normally"},
        "vpn": {"status": "operational", "message": "VPN services available in all regions"},
        "network": {"status": "operational", "message": "Network connectivity normal"},
        "erp": {"status": "degraded", "message": "SAP experiencing slow response times. Team investigating."},
        "crm": {"status": "operational", "message": "Salesforce running normally"},
        "teams": {"status": "operational", "message": "Microsoft Teams fully operational"},
        "sharepoint": {"status": "maintenance", "message": "Scheduled maintenance 2-4 AM UTC"},
    }

    if system:
        sys_status = systems.get(system.lower())
        if sys_status:
            status_emoji = {"operational": "🟢", "degraded": "🟡", "maintenance": "🔵", "outage": "🔴"}
            emoji = status_emoji.get(sys_status["status"], "⚪")
            return f"{emoji} **{system.upper()}**: {sys_status['status'].upper()}\n{sys_status['message']}"
        return f"System '{system}' not found. Available systems: {', '.join(systems.keys())}"

    # Return all systems status
    report = ["**IT Systems Status Dashboard**\n"]
    status_emoji = {"operational": "🟢", "degraded": "🟡", "maintenance": "🔵", "outage": "🔴"}

    for name, info in systems.items():
        emoji = status_emoji.get(info["status"], "⚪")
        report.append(f"{emoji} **{name.upper()}**: {info['status']} - {info['message']}")

    return "\n".join(report)


@tool
def initiate_password_reset(employee_id: str, reset_method: Literal["email", "sms", "security_questions"] = "email") -> str:
    """Initiate a password reset for a user.

    Args:
        employee_id: Employee ID or email address.
        reset_method: Method to receive reset link (email, sms, security_questions).

    Returns:
        Confirmation of password reset initiation.
    """
    return f"""Password reset initiated for {employee_id}

**Reset Method:** {reset_method.replace('_', ' ').title()}

Next steps:
1. {"Check your registered email for the reset link" if reset_method == "email" else "Check your registered phone for SMS code" if reset_method == "sms" else "You will be prompted to answer your security questions"}
2. The reset link/code is valid for 15 minutes
3. Create a new password following the password policy
4. You will be logged out of all active sessions

If you don't receive the reset {"email" if reset_method == "email" else "SMS"} within 5 minutes:
- Check your spam/junk folder
- Verify your contact information is up to date
- Contact IT Support at ext. 5555"""


@tool
def request_software(
    software_name: str,
    business_justification: str,
    urgency: Literal["standard", "urgent"] = "standard",
) -> str:
    """Request software installation or license.

    Args:
        software_name: Name of the software to request.
        business_justification: Business reason for the software.
        urgency: Request urgency (standard: 48hrs, urgent: 24hrs).

    Returns:
        Software request confirmation.
    """
    request_id = f"SWR{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    return f"""Software request submitted successfully!

**Request ID:** {request_id}
**Software:** {software_name}
**Urgency:** {urgency.title()}
**Justification:** {business_justification}

**Approval Workflow:**
1. Manager approval required
2. IT Security review (for new software)
3. License procurement (if needed)
4. Installation scheduling

**Expected Timeline:**
- Standard: 2-3 business days
- Urgent: 24 hours (requires additional approval)

You will receive email updates on the request status."""


@tool
def escalate_to_human(reason: str, preferred_contact: Literal["phone", "email", "chat"] = "chat") -> str:
    """Escalate the issue to a human IT support agent.

    Args:
        reason: Reason for escalation.
        preferred_contact: Preferred contact method.

    Returns:
        Escalation confirmation and next steps.
    """
    escalation_id = f"ESC{datetime.now().strftime('%H%M%S')}"

    contact_info = {
        "phone": "Call IT Support at ext. 5555 (Mon-Fri 8AM-6PM)",
        "email": "Email it-support@company.com (Response within 4 hours)",
        "chat": "Live chat available at https://support.company.com/chat",
    }

    return f"""Issue escalated to human support.

**Escalation ID:** {escalation_id}
**Reason:** {reason}
**Preferred Contact:** {preferred_contact.title()}

**Next Steps:**
{contact_info[preferred_contact]}

A support specialist will review your case and contact you shortly.
Please have the following ready:
- Your employee ID
- Description of the issue
- Any error messages or screenshots
- Steps you've already tried"""


# =============================================================================
# IT Helpdesk Agent Class
# =============================================================================

class ITHelpdeskAgent:
    """IT Helpdesk Agent with conversation memory and LangGraph workflow."""

    SYSTEM_PROMPT = """You are an expert IT Support Agent for a large enterprise organization. Your role is to help employees with technical issues, answer IT-related questions, and guide them through troubleshooting steps.

**Your Capabilities:**
1. Search the IT knowledge base for solutions
2. Create and track support tickets
3. Check IT system status
4. Initiate password resets
5. Process software requests
6. Escalate complex issues to human agents

**Guidelines:**
- Always be professional, patient, and helpful
- Start by understanding the user's issue clearly
- Search the knowledge base before creating tickets
- Provide step-by-step instructions when possible
- Ask clarifying questions if the issue is unclear
- Create tickets for issues that require hands-on support
- Escalate to human agents when you cannot resolve the issue

**Response Format:**
- Use clear, concise language
- Format steps as numbered lists
- Highlight important information
- Provide relevant links when available
- Always confirm understanding before taking actions

Remember: Your goal is to resolve issues efficiently while providing excellent user experience."""

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
        model_name: str | None = None,
        temperature: float = 0,
    ) -> None:
        """Initialize IT Helpdesk Agent.

        Args:
            model_provider: LLM provider to use.
            model_name: Specific model name (uses default if not specified).
            temperature: LLM temperature setting.
        """
        self.model_provider = model_provider
        self.temperature = temperature

        # Initialize LLM
        self.llm = self._get_llm(model_provider, model_name, temperature)

        # Define tools
        self.tools = [
            search_knowledge_base,
            create_support_ticket,
            check_ticket_status,
            check_system_status,
            initiate_password_reset,
            request_software,
            escalate_to_human,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize memory (for conversation persistence)
        self.memory = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()

    def _get_llm(
        self,
        provider: str,
        model_name: str | None,
        temperature: float,
    ) -> ChatOpenAI | ChatAnthropic:
        """Get LLM instance based on provider."""
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

        if provider == "auto":
            if has_anthropic:
                provider = "anthropic"
            elif has_openai:
                provider = "openai"
            else:
                raise ValueError("No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

        if provider == "anthropic":
            return ChatAnthropic(
                model=model_name or "claude-sonnet-4-20250514",
                temperature=temperature,
            )
        else:
            return ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=temperature,
            )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create graph with state
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))

        # Add edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        # Compile with memory
        return graph.compile(checkpointer=self.memory)

    def _agent_node(self, state: AgentState) -> dict:
        """Process messages and decide on actions."""
        messages = state.messages

        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + list(messages)

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    @traceable(name="it_helpdesk_chat", tags=["it-support", "helpdesk"])
    def chat(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a chat message.

        Args:
            message: User's message.
            thread_id: Conversation thread ID for memory.

        Returns:
            Response with answer and metadata.
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        # Extract the last AI message
        last_message = result["messages"][-1]

        return {
            "response": last_message.content,
            "thread_id": thread_id,
            "tool_calls": getattr(last_message, "tool_calls", []),
        }

    async def achat(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of chat.

        Args:
            message: User's message.
            thread_id: Conversation thread ID for memory.

        Returns:
            Response with answer and metadata.
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph asynchronously
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        # Extract the last AI message
        last_message = result["messages"][-1]

        return {
            "response": last_message.content,
            "thread_id": thread_id,
            "tool_calls": getattr(last_message, "tool_calls", []),
        }

    def get_conversation_history(self, thread_id: str) -> list[dict]:
        """Get conversation history for a thread.

        Args:
            thread_id: The thread ID to retrieve.

        Returns:
            List of messages in the conversation.
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = self.graph.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {
                        "role": "assistant" if isinstance(m, AIMessage) else "user" if isinstance(m, HumanMessage) else "system",
                        "content": m.content,
                    }
                    for m in messages
                    if not isinstance(m, SystemMessage)
                ]
        except Exception:
            pass

        return []


# Global instance
it_helpdesk_agent = ITHelpdeskAgent(model_provider="auto")
