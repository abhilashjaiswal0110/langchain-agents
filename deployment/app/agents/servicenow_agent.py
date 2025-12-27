"""ServiceNow Integration Agent for ITSM operations.

This agent provides integration with ServiceNow for ticket management,
change requests, and CMDB operations.
"""

import os
import uuid
from datetime import datetime, timedelta
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

class ServiceNowState(BaseModel):
    """State for ServiceNow Agent."""

    messages: Annotated[list, add_messages]
    current_ticket: str | None = None
    user_info: dict | None = None


# =============================================================================
# Simulated ServiceNow Data
# =============================================================================

# Simulated incidents database
INCIDENTS_DB: dict[str, dict[str, Any]] = {
    "INC0010001": {
        "number": "INC0010001",
        "short_description": "Email not syncing on mobile device",
        "description": "User reports that emails are not syncing on their iPhone since Monday.",
        "state": "In Progress",
        "priority": "3 - Moderate",
        "assigned_to": "John Smith",
        "assignment_group": "Email Support",
        "caller": "jane.doe@company.com",
        "created": "2024-12-13 09:00:00",
        "updated": "2024-12-14 14:30:00",
        "category": "Software",
        "subcategory": "Email",
        "comments": [
            {"user": "John Smith", "text": "Investigating sync settings", "time": "2024-12-14 14:30:00"}
        ]
    },
    "INC0010002": {
        "number": "INC0010002",
        "short_description": "VPN connection dropping frequently",
        "description": "VPN disconnects every 10-15 minutes requiring reconnection.",
        "state": "New",
        "priority": "2 - High",
        "assigned_to": None,
        "assignment_group": "Network Support",
        "caller": "bob.johnson@company.com",
        "created": "2024-12-15 08:00:00",
        "updated": "2024-12-15 08:00:00",
        "category": "Network",
        "subcategory": "VPN",
        "comments": []
    }
}

# Simulated change requests
CHANGE_REQUESTS_DB: dict[str, dict[str, Any]] = {
    "CHG0001234": {
        "number": "CHG0001234",
        "short_description": "Windows Server 2019 Security Patches",
        "description": "Monthly security patch deployment for all Windows Server 2019 instances.",
        "state": "Scheduled",
        "type": "Standard",
        "risk": "Low",
        "planned_start": "2024-12-18 02:00:00",
        "planned_end": "2024-12-18 06:00:00",
        "assigned_to": "DevOps Team",
        "approval_status": "Approved",
        "impact": "Low - Automated patching with auto-restart"
    }
}

# Simulated CMDB data
CMDB_DB: dict[str, dict[str, Any]] = {
    "SRV001": {
        "name": "PROD-WEB-01",
        "class": "Server",
        "os": "Windows Server 2019",
        "ip": "10.0.1.100",
        "location": "DC-East",
        "status": "Operational",
        "owner": "Web Team"
    },
    "SRV002": {
        "name": "PROD-DB-01",
        "class": "Server",
        "os": "Linux RHEL 8",
        "ip": "10.0.1.101",
        "location": "DC-East",
        "status": "Operational",
        "owner": "Database Team"
    },
    "APP001": {
        "name": "SAP-ERP",
        "class": "Application",
        "version": "S/4HANA 2023",
        "status": "Operational",
        "owner": "ERP Team",
        "dependencies": ["PROD-DB-01", "PROD-APP-01"]
    }
}


# =============================================================================
# ServiceNow Tools
# =============================================================================

@tool
def search_incidents(
    query: str | None = None,
    state: str | None = None,
    priority: str | None = None,
    assigned_to: str | None = None,
    limit: int = 5,
) -> str:
    """Search for incidents in ServiceNow.

    Args:
        query: Search query for incident description.
        state: Filter by state (New, In Progress, Resolved, Closed).
        priority: Filter by priority (1-Critical, 2-High, 3-Moderate, 4-Low).
        assigned_to: Filter by assignee name.
        limit: Maximum number of results to return.

    Returns:
        List of matching incidents.
    """
    results = []

    for inc_id, incident in INCIDENTS_DB.items():
        # Apply filters
        if state and incident["state"].lower() != state.lower():
            continue
        if priority and priority not in incident["priority"]:
            continue
        if assigned_to and incident["assigned_to"] and assigned_to.lower() not in incident["assigned_to"].lower():
            continue
        if query:
            query_lower = query.lower()
            if query_lower not in incident["short_description"].lower() and query_lower not in incident["description"].lower():
                continue

        results.append(incident)

        if len(results) >= limit:
            break

    if not results:
        return "No incidents found matching the criteria."

    output = [f"**Found {len(results)} incident(s):**\n"]
    for inc in results:
        output.append(f"""
**{inc['number']}** - {inc['short_description']}
- State: {inc['state']}
- Priority: {inc['priority']}
- Assigned to: {inc['assigned_to'] or 'Unassigned'}
- Created: {inc['created']}
""")

    return "\n".join(output)


@tool
def get_incident_details(incident_number: str) -> str:
    """Get detailed information about a specific incident.

    Args:
        incident_number: The incident number (e.g., INC0010001).

    Returns:
        Detailed incident information.
    """
    incident = INCIDENTS_DB.get(incident_number.upper())

    if not incident:
        return f"Incident {incident_number} not found. Please verify the incident number."

    comments_text = "\n".join([
        f"  - [{c['time']}] {c['user']}: {c['text']}"
        for c in incident.get("comments", [])
    ]) or "  No comments yet"

    return f"""**Incident Details: {incident['number']}**

**Short Description:** {incident['short_description']}
**Description:** {incident['description']}

**Status Information:**
- State: {incident['state']}
- Priority: {incident['priority']}
- Category: {incident['category']} / {incident['subcategory']}

**Assignment:**
- Assigned To: {incident['assigned_to'] or 'Unassigned'}
- Assignment Group: {incident['assignment_group']}

**Caller:** {incident['caller']}

**Timestamps:**
- Created: {incident['created']}
- Last Updated: {incident['updated']}

**Work Notes/Comments:**
{comments_text}"""


@tool
def create_incident(
    short_description: str,
    description: str,
    category: str,
    subcategory: str,
    priority: Literal["1", "2", "3", "4"] = "3",
    caller_email: str | None = None,
) -> str:
    """Create a new incident in ServiceNow.

    Args:
        short_description: Brief title of the incident.
        description: Detailed description of the issue.
        category: Main category (Hardware, Software, Network, Access).
        subcategory: Subcategory within the main category.
        priority: Priority level (1-Critical, 2-High, 3-Moderate, 4-Low).
        caller_email: Email of the user reporting the issue.

    Returns:
        Created incident details.
    """
    incident_number = f"INC{str(uuid.uuid4().int)[:7]}"

    priority_map = {
        "1": "1 - Critical",
        "2": "2 - High",
        "3": "3 - Moderate",
        "4": "4 - Low"
    }

    incident = {
        "number": incident_number,
        "short_description": short_description,
        "description": description,
        "state": "New",
        "priority": priority_map.get(priority, "3 - Moderate"),
        "assigned_to": None,
        "assignment_group": f"{category} Support",
        "caller": caller_email or "unknown",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": category,
        "subcategory": subcategory,
        "comments": []
    }

    INCIDENTS_DB[incident_number] = incident

    return f"""**Incident Created Successfully**

**Incident Number:** {incident_number}
**Short Description:** {short_description}
**Priority:** {priority_map.get(priority, '3 - Moderate')}
**Category:** {category} / {subcategory}
**State:** New

The incident has been submitted and will be assigned to the {category} Support team.
You will receive email updates at {caller_email or 'your registered email'}.

**SLA Information:**
- Critical (P1): 1 hour response, 4 hour resolution
- High (P2): 4 hour response, 8 hour resolution
- Moderate (P3): 8 hour response, 24 hour resolution
- Low (P4): 24 hour response, 72 hour resolution"""


@tool
def update_incident(
    incident_number: str,
    work_notes: str | None = None,
    state: str | None = None,
    assigned_to: str | None = None,
) -> str:
    """Update an existing incident in ServiceNow.

    Args:
        incident_number: The incident number to update.
        work_notes: Notes to add to the incident.
        state: New state (In Progress, On Hold, Resolved, Closed).
        assigned_to: New assignee name.

    Returns:
        Updated incident confirmation.
    """
    incident = INCIDENTS_DB.get(incident_number.upper())

    if not incident:
        return f"Incident {incident_number} not found."

    updates = []

    if state:
        incident["state"] = state
        updates.append(f"State changed to: {state}")

    if assigned_to:
        incident["assigned_to"] = assigned_to
        updates.append(f"Assigned to: {assigned_to}")

    if work_notes:
        incident["comments"].append({
            "user": "IT Support Agent",
            "text": work_notes,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        updates.append("Work notes added")

    incident["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""**Incident {incident_number} Updated**

Updates applied:
{chr(10).join('- ' + u for u in updates)}

Current State: {incident['state']}
Last Updated: {incident['updated']}"""


@tool
def get_change_requests(
    state: str | None = None,
    upcoming_days: int = 7,
) -> str:
    """Get upcoming or recent change requests.

    Args:
        state: Filter by state (Scheduled, Implement, Review, Closed).
        upcoming_days: Number of days to look ahead for scheduled changes.

    Returns:
        List of change requests.
    """
    results = []

    for chg_id, change in CHANGE_REQUESTS_DB.items():
        if state and change["state"].lower() != state.lower():
            continue
        results.append(change)

    if not results:
        return "No change requests found matching the criteria."

    output = [f"**Change Requests ({len(results)} found):**\n"]

    for chg in results:
        output.append(f"""
**{chg['number']}** - {chg['short_description']}
- Type: {chg['type']} | Risk: {chg['risk']}
- State: {chg['state']} | Approval: {chg['approval_status']}
- Planned: {chg['planned_start']} to {chg['planned_end']}
- Impact: {chg['impact']}
""")

    return "\n".join(output)


@tool
def search_cmdb(
    query: str | None = None,
    ci_class: str | None = None,
    status: str | None = None,
) -> str:
    """Search the Configuration Management Database (CMDB).

    Args:
        query: Search query for CI name or description.
        ci_class: Filter by class (Server, Application, Network Device).
        status: Filter by status (Operational, Maintenance, Retired).

    Returns:
        List of matching configuration items.
    """
    results = []

    for ci_id, ci in CMDB_DB.items():
        if ci_class and ci["class"].lower() != ci_class.lower():
            continue
        if status and ci["status"].lower() != status.lower():
            continue
        if query and query.lower() not in ci["name"].lower():
            continue
        results.append((ci_id, ci))

    if not results:
        return "No configuration items found matching the criteria."

    output = [f"**CMDB Search Results ({len(results)} items):**\n"]

    for ci_id, ci in results:
        if ci["class"] == "Server":
            output.append(f"""
**{ci['name']}** ({ci_id})
- Class: {ci['class']}
- OS: {ci['os']}
- IP: {ci['ip']}
- Location: {ci['location']}
- Status: {ci['status']}
- Owner: {ci['owner']}
""")
        else:
            deps = ", ".join(ci.get("dependencies", [])) or "None"
            output.append(f"""
**{ci['name']}** ({ci_id})
- Class: {ci['class']}
- Version: {ci.get('version', 'N/A')}
- Status: {ci['status']}
- Owner: {ci['owner']}
- Dependencies: {deps}
""")

    return "\n".join(output)


@tool
def get_my_tickets(user_email: str) -> str:
    """Get all tickets for a specific user.

    Args:
        user_email: User's email address.

    Returns:
        List of user's incidents.
    """
    results = []

    for inc_id, incident in INCIDENTS_DB.items():
        if incident["caller"].lower() == user_email.lower():
            results.append(incident)

    if not results:
        return f"No tickets found for {user_email}."

    output = [f"**Tickets for {user_email}:**\n"]

    for inc in results:
        output.append(f"""
**{inc['number']}** - {inc['short_description']}
- State: {inc['state']} | Priority: {inc['priority']}
- Updated: {inc['updated']}
""")

    return "\n".join(output)


# =============================================================================
# ServiceNow Agent Class
# =============================================================================

class ServiceNowAgent:
    """ServiceNow Agent for ITSM operations with conversation memory."""

    SYSTEM_PROMPT = """You are a ServiceNow ITSM Agent specialized in helping users interact with the ServiceNow platform. You have access to incident management, change management, and CMDB functions.

**Your Capabilities:**
1. Search and retrieve incidents
2. Create new incidents with proper categorization
3. Update existing incidents with work notes
4. View upcoming change requests
5. Search the CMDB for configuration items
6. Track user's tickets

**Best Practices:**
- Always verify incident numbers before taking action
- Use appropriate priority levels based on business impact
- Provide clear categorization for new incidents
- Include relevant details when creating incidents
- Check CMDB for related configuration items when investigating issues

**Response Guidelines:**
- Confirm actions before making changes
- Provide ticket numbers for reference
- Explain SLA timelines when relevant
- Suggest related actions when appropriate

You are integrated with the ServiceNow instance and can perform real-time operations."""

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
        model_name: str | None = None,
        temperature: float = 0,
    ) -> None:
        """Initialize ServiceNow Agent."""
        self.model_provider = model_provider
        self.temperature = temperature

        # Initialize LLM
        self.llm = self._get_llm(model_provider, model_name, temperature)

        # Define tools
        self.tools = [
            search_incidents,
            get_incident_details,
            create_incident,
            update_incident,
            get_change_requests,
            search_cmdb,
            get_my_tickets,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize memory
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
                raise ValueError("No LLM API key found.")

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
        graph = StateGraph(ServiceNowState)

        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile(checkpointer=self.memory)

    def _agent_node(self, state: ServiceNowState) -> dict:
        """Process messages and decide on actions."""
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + list(messages)

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: ServiceNowState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    @traceable(name="servicenow_chat", tags=["servicenow", "itsm"])
    def chat(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a chat message."""
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

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
        """Async version of chat."""
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}

        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        last_message = result["messages"][-1]

        return {
            "response": last_message.content,
            "thread_id": thread_id,
            "tool_calls": getattr(last_message, "tool_calls", []),
        }


# NOTE: Global instance removed to avoid instantiation before .env is loaded
# Instances are now created lazily by ConversationManager
