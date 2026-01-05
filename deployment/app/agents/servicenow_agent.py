"""ServiceNow Integration Agent for ITSM operations.

This agent provides integration with ServiceNow for ticket management,
change requests, and CMDB operations. Supports both live API calls
and simulation mode for development/testing.
"""

import asyncio
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
# ServiceNow Configuration
# =============================================================================

def get_servicenow_config() -> dict[str, Any]:
    """Get ServiceNow configuration from environment.

    Returns:
        Configuration dictionary with instance, credentials, and mode.
    """
    return {
        "instance": os.getenv("SERVICENOW_INSTANCE", ""),
        "username": os.getenv("SERVICENOW_USERNAME", ""),
        "password": os.getenv("SERVICENOW_PASSWORD", ""),
        "mode": os.getenv("SERVICENOW_MODE", "simulation"),
        "timeout": int(os.getenv("SERVICENOW_TIMEOUT", "30")),
        "verify_ssl": os.getenv("SERVICENOW_VERIFY_SSL", "true").lower() == "true",
    }


def is_live_mode() -> bool:
    """Check if ServiceNow is configured for live mode.

    Returns:
        True if live mode is enabled and credentials are configured.
    """
    config = get_servicenow_config()
    is_live = config["mode"] == "live"
    is_configured = bool(config["instance"] and config["username"] and config["password"])
    return is_live and is_configured


def get_base_url() -> str:
    """Get ServiceNow base URL.

    Returns:
        Base URL for ServiceNow API.
    """
    instance = os.getenv("SERVICENOW_INSTANCE", "")
    return f"https://{instance}.service-now.com"


# =============================================================================
# HTTP Client for ServiceNow API
# =============================================================================

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class ServiceNowAPI:
    """Synchronous wrapper for ServiceNow REST API calls."""

    def __init__(self) -> None:
        """Initialize API client."""
        self.config = get_servicenow_config()
        self._client: httpx.Client | None = None

    def _get_client(self) -> "httpx.Client":
        """Get or create HTTP client."""
        if not HTTPX_AVAILABLE:
            msg = "httpx package required for live mode. Install: pip install httpx"
            raise RuntimeError(msg)
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config["timeout"],
                verify=self.config["verify_ssl"],
            )
        return self._client

    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make authenticated request to ServiceNow.

        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            json: Request body.
            params: Query parameters.

        Returns:
            Response JSON.

        Raises:
            RuntimeError: If request fails.
        """
        client = self._get_client()
        url = f"{get_base_url()}{endpoint}"

        try:
            response = client.request(
                method=method,
                url=url,
                json=json,
                params=params,
                auth=(self.config["username"], self.config["password"]),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            msg = f"ServiceNow API error: {e.response.status_code} - {e.response.text}"
            raise RuntimeError(msg) from e
        except httpx.RequestError as e:
            msg = f"ServiceNow connection error: {e}"
            raise RuntimeError(msg) from e

    def get_incidents(
        self,
        query: str | None = None,
        state: str | None = None,
        priority: str | None = None,
        assigned_to: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for incidents in ServiceNow.

        Args:
            query: Text search in short_description or description.
            state: Filter by incident state.
            priority: Filter by priority.
            assigned_to: Filter by assignee.
            limit: Maximum results.

        Returns:
            List of incident records.
        """
        query_parts = []

        if query:
            query_parts.append(f"short_descriptionLIKE{query}^ORdescriptionLIKE{query}")
        if state:
            # Map friendly state names to ServiceNow state values
            state_map = {
                "new": "1",
                "in progress": "2",
                "on hold": "3",
                "resolved": "6",
                "closed": "7",
            }
            state_value = state_map.get(state.lower(), state)
            query_parts.append(f"state={state_value}")
        if priority:
            query_parts.append(f"priority={priority}")
        if assigned_to:
            query_parts.append(f"assigned_to.nameLIKE{assigned_to}")

        params = {
            "sysparm_query": "^".join(query_parts) if query_parts else "",
            "sysparm_limit": limit,
            "sysparm_display_value": "true",
        }

        result = self._request("GET", "/api/now/table/incident", params=params)
        return result.get("result", [])

    def get_incident(self, incident_number: str) -> dict[str, Any] | None:
        """Get incident by number.

        Args:
            incident_number: Incident number (e.g., INC0010001).

        Returns:
            Incident record or None if not found.
        """
        params = {
            "sysparm_query": f"number={incident_number}",
            "sysparm_limit": 1,
            "sysparm_display_value": "true",
        }

        result = self._request("GET", "/api/now/table/incident", params=params)
        incidents = result.get("result", [])
        return incidents[0] if incidents else None

    def create_incident(
        self,
        short_description: str,
        description: str,
        category: str,
        subcategory: str,
        priority: str = "3",
        caller_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new incident.

        Args:
            short_description: Brief title.
            description: Detailed description.
            category: Incident category.
            subcategory: Subcategory.
            priority: Priority level (1-4).
            caller_id: Caller's sys_id or email.
            **kwargs: Additional fields.

        Returns:
            Created incident record.
        """
        data = {
            "short_description": short_description,
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "priority": priority,
            **kwargs,
        }

        if caller_id:
            data["caller_id"] = caller_id

        result = self._request("POST", "/api/now/table/incident", json=data)
        return result.get("result", {})

    def update_incident(
        self,
        incident_number: str,
        work_notes: str | None = None,
        state: str | None = None,
        assigned_to: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update an existing incident.

        Args:
            incident_number: Incident number.
            work_notes: Notes to add.
            state: New state.
            assigned_to: New assignee.
            **kwargs: Additional fields.

        Returns:
            Updated incident record.
        """
        # First get the sys_id
        incident = self.get_incident(incident_number)
        if not incident:
            return {"error": f"Incident {incident_number} not found"}

        sys_id = incident.get("sys_id")
        if not sys_id:
            return {"error": "Could not get incident sys_id"}

        data: dict[str, Any] = {**kwargs}

        if work_notes:
            data["work_notes"] = work_notes
        if state:
            state_map = {
                "new": "1",
                "in progress": "2",
                "on hold": "3",
                "resolved": "6",
                "closed": "7",
            }
            data["state"] = state_map.get(state.lower(), state)
        if assigned_to:
            data["assigned_to"] = assigned_to

        result = self._request("PATCH", f"/api/now/table/incident/{sys_id}", json=data)
        return result.get("result", {})

    def get_change_requests(
        self,
        state: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get change requests.

        Args:
            state: Filter by state.
            limit: Maximum results.

        Returns:
            List of change request records.
        """
        query_parts = []

        if state:
            query_parts.append(f"state={state}")

        params = {
            "sysparm_query": "^".join(query_parts) if query_parts else "",
            "sysparm_limit": limit,
            "sysparm_display_value": "true",
        }

        result = self._request("GET", "/api/now/table/change_request", params=params)
        return result.get("result", [])

    def search_cmdb(
        self,
        query: str | None = None,
        ci_class: str = "cmdb_ci",
        status: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search CMDB for configuration items.

        Args:
            query: Search query for CI name.
            ci_class: CI class table (e.g., cmdb_ci_server).
            status: Filter by status.
            limit: Maximum results.

        Returns:
            List of CI records.
        """
        query_parts = []

        if query:
            query_parts.append(f"nameLIKE{query}")
        if status:
            query_parts.append(f"install_status={status}")

        # Map friendly class names to table names
        class_map = {
            "server": "cmdb_ci_server",
            "application": "cmdb_ci_appl",
            "network device": "cmdb_ci_netgear",
            "computer": "cmdb_ci_computer",
        }
        table = class_map.get(ci_class.lower(), ci_class)

        params = {
            "sysparm_query": "^".join(query_parts) if query_parts else "",
            "sysparm_limit": limit,
            "sysparm_display_value": "true",
        }

        result = self._request("GET", f"/api/now/table/{table}", params=params)
        return result.get("result", [])

    def get_user_incidents(
        self,
        user_email: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get incidents for a specific user.

        Args:
            user_email: User's email address.
            limit: Maximum results.

        Returns:
            List of incident records.
        """
        params = {
            "sysparm_query": f"caller_id.email={user_email}",
            "sysparm_limit": limit,
            "sysparm_display_value": "true",
        }

        result = self._request("GET", "/api/now/table/incident", params=params)
        return result.get("result", [])

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


# Global API client instance
_api_client: ServiceNowAPI | None = None


def get_api_client() -> ServiceNowAPI:
    """Get or create the API client.

    Returns:
        ServiceNowAPI instance.
    """
    global _api_client
    if _api_client is None:
        _api_client = ServiceNowAPI()
    return _api_client


# =============================================================================
# Agent State
# =============================================================================

class ServiceNowState(BaseModel):
    """State for ServiceNow Agent."""

    messages: Annotated[list, add_messages]
    current_ticket: str | None = None
    user_info: dict | None = None


# =============================================================================
# Simulated ServiceNow Data (Fallback for simulation mode)
# =============================================================================

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
    if is_live_mode():
        try:
            api = get_api_client()
            incidents = api.get_incidents(
                query=query,
                state=state,
                priority=priority,
                assigned_to=assigned_to,
                limit=limit,
            )

            if not incidents:
                return "No incidents found matching the criteria."

            output = [f"**Found {len(incidents)} incident(s) [LIVE DATA]:**\n"]
            for inc in incidents:
                output.append(f"""
**{inc.get('number', 'N/A')}** - {inc.get('short_description', 'No description')}
- State: {inc.get('state', 'Unknown')}
- Priority: {inc.get('priority', 'Unknown')}
- Assigned to: {inc.get('assigned_to', 'Unassigned') or 'Unassigned'}
- Created: {inc.get('sys_created_on', 'Unknown')}
""")
            return "\n".join(output)

        except Exception as e:
            return f"Error searching incidents: {e}"

    # Simulation mode - use mock data
    results = []

    for inc_id, incident in INCIDENTS_DB.items():
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

    output = [f"**Found {len(results)} incident(s) [SIMULATION]:**\n"]
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
    if is_live_mode():
        try:
            api = get_api_client()
            incident = api.get_incident(incident_number.upper())

            if not incident:
                return f"Incident {incident_number} not found. Please verify the incident number."

            return f"""**Incident Details: {incident.get('number')}** [LIVE DATA]

**Short Description:** {incident.get('short_description', 'N/A')}
**Description:** {incident.get('description', 'N/A')}

**Status Information:**
- State: {incident.get('state', 'Unknown')}
- Priority: {incident.get('priority', 'Unknown')}
- Category: {incident.get('category', 'N/A')} / {incident.get('subcategory', 'N/A')}

**Assignment:**
- Assigned To: {incident.get('assigned_to', 'Unassigned') or 'Unassigned'}
- Assignment Group: {incident.get('assignment_group', 'N/A')}

**Caller:** {incident.get('caller_id', 'Unknown')}

**Timestamps:**
- Created: {incident.get('sys_created_on', 'Unknown')}
- Last Updated: {incident.get('sys_updated_on', 'Unknown')}

**Work Notes:**
{incident.get('work_notes', 'No work notes')}"""

        except Exception as e:
            return f"Error getting incident details: {e}"

    # Simulation mode
    incident = INCIDENTS_DB.get(incident_number.upper())

    if not incident:
        return f"Incident {incident_number} not found. Please verify the incident number."

    comments_text = "\n".join([
        f"  - [{c['time']}] {c['user']}: {c['text']}"
        for c in incident.get("comments", [])
    ]) or "  No comments yet"

    return f"""**Incident Details: {incident['number']}** [SIMULATION]

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
    priority_map = {
        "1": "1 - Critical",
        "2": "2 - High",
        "3": "3 - Moderate",
        "4": "4 - Low"
    }

    if is_live_mode():
        try:
            api = get_api_client()
            result = api.create_incident(
                short_description=short_description,
                description=description,
                category=category,
                subcategory=subcategory,
                priority=priority,
                caller_id=caller_email,
            )

            if "error" in result:
                return f"Error creating incident: {result['error']}"

            return f"""**Incident Created Successfully** [LIVE DATA]

**Incident Number:** {result.get('number', 'N/A')}
**Short Description:** {short_description}
**Priority:** {priority_map.get(priority, '3 - Moderate')}
**Category:** {category} / {subcategory}
**State:** {result.get('state', 'New')}

The incident has been submitted to ServiceNow.
You will receive email updates at {caller_email or 'your registered email'}.

**SLA Information:**
- Critical (P1): 1 hour response, 4 hour resolution
- High (P2): 4 hour response, 8 hour resolution
- Moderate (P3): 8 hour response, 24 hour resolution
- Low (P4): 24 hour response, 72 hour resolution"""

        except Exception as e:
            return f"Error creating incident: {e}"

    # Simulation mode
    incident_number = f"INC{str(uuid.uuid4().int)[:7]}"

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

    return f"""**Incident Created Successfully** [SIMULATION]

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
    if is_live_mode():
        try:
            api = get_api_client()
            result = api.update_incident(
                incident_number=incident_number.upper(),
                work_notes=work_notes,
                state=state,
                assigned_to=assigned_to,
            )

            if "error" in result:
                return f"Error updating incident: {result['error']}"

            updates = []
            if state:
                updates.append(f"State changed to: {state}")
            if assigned_to:
                updates.append(f"Assigned to: {assigned_to}")
            if work_notes:
                updates.append("Work notes added")

            return f"""**Incident {incident_number} Updated** [LIVE DATA]

Updates applied:
{chr(10).join('- ' + u for u in updates)}

Current State: {result.get('state', 'Unknown')}
Last Updated: {result.get('sys_updated_on', 'Unknown')}"""

        except Exception as e:
            return f"Error updating incident: {e}"

    # Simulation mode
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

    return f"""**Incident {incident_number} Updated** [SIMULATION]

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
    if is_live_mode():
        try:
            api = get_api_client()
            changes = api.get_change_requests(state=state, limit=10)

            if not changes:
                return "No change requests found matching the criteria."

            output = [f"**Change Requests ({len(changes)} found) [LIVE DATA]:**\n"]

            for chg in changes:
                output.append(f"""
**{chg.get('number', 'N/A')}** - {chg.get('short_description', 'No description')}
- Type: {chg.get('type', 'N/A')} | Risk: {chg.get('risk', 'N/A')}
- State: {chg.get('state', 'Unknown')} | Approval: {chg.get('approval', 'Unknown')}
- Planned: {chg.get('start_date', 'TBD')} to {chg.get('end_date', 'TBD')}
""")

            return "\n".join(output)

        except Exception as e:
            return f"Error getting change requests: {e}"

    # Simulation mode
    results = []

    for chg_id, change in CHANGE_REQUESTS_DB.items():
        if state and change["state"].lower() != state.lower():
            continue
        results.append(change)

    if not results:
        return "No change requests found matching the criteria."

    output = [f"**Change Requests ({len(results)} found) [SIMULATION]:**\n"]

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
    if is_live_mode():
        try:
            api = get_api_client()
            cis = api.search_cmdb(
                query=query,
                ci_class=ci_class or "cmdb_ci",
                status=status,
                limit=10,
            )

            if not cis:
                return "No configuration items found matching the criteria."

            output = [f"**CMDB Search Results ({len(cis)} items) [LIVE DATA]:**\n"]

            for ci in cis:
                output.append(f"""
**{ci.get('name', 'N/A')}** ({ci.get('sys_id', 'N/A')[:8]}...)
- Class: {ci.get('sys_class_name', 'Unknown')}
- Status: {ci.get('install_status', 'Unknown')}
- Owner: {ci.get('owned_by', 'Unknown')}
""")

            return "\n".join(output)

        except Exception as e:
            return f"Error searching CMDB: {e}"

    # Simulation mode
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

    output = [f"**CMDB Search Results ({len(results)} items) [SIMULATION]:**\n"]

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
    if is_live_mode():
        try:
            api = get_api_client()
            incidents = api.get_user_incidents(user_email, limit=10)

            if not incidents:
                return f"No tickets found for {user_email}."

            output = [f"**Tickets for {user_email} [LIVE DATA]:**\n"]

            for inc in incidents:
                output.append(f"""
**{inc.get('number', 'N/A')}** - {inc.get('short_description', 'No description')}
- State: {inc.get('state', 'Unknown')} | Priority: {inc.get('priority', 'Unknown')}
- Updated: {inc.get('sys_updated_on', 'Unknown')}
""")

            return "\n".join(output)

        except Exception as e:
            return f"Error getting user tickets: {e}"

    # Simulation mode
    results = []

    for inc_id, incident in INCIDENTS_DB.items():
        if incident["caller"].lower() == user_email.lower():
            results.append(incident)

    if not results:
        return f"No tickets found for {user_email}."

    output = [f"**Tickets for {user_email} [SIMULATION]:**\n"]

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

    def _get_system_prompt(self) -> str:
        """Generate system prompt with current mode information.

        Returns:
            System prompt string.
        """
        mode = "LIVE" if is_live_mode() else "SIMULATION"
        config = get_servicenow_config()
        instance_info = f"Instance: {config['instance']}" if config['instance'] else "Instance: Not configured"

        return f"""You are a ServiceNow ITSM Agent specialized in helping users interact with the ServiceNow platform. You have access to incident management, change management, and CMDB functions.

**Current Mode:** {mode}
**{instance_info}**

{"**NOTE:** You are connected to a LIVE ServiceNow instance. All operations will affect real data." if mode == "LIVE" else "**NOTE:** Running in simulation mode with sample data. Configure SERVICENOW_MODE=live for real API calls."}

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

You are integrated with the ServiceNow platform and can perform real-time operations."""

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic", "auto"] = "auto",
        model_name: str | None = None,
        temperature: float = 0,
    ) -> None:
        """Initialize ServiceNow Agent.

        Args:
            model_provider: LLM provider to use.
            model_name: Specific model name.
            temperature: LLM temperature.
        """
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
        """Get LLM instance based on provider.

        Args:
            provider: LLM provider name.
            model_name: Specific model name.
            temperature: LLM temperature.

        Returns:
            LLM instance.

        Raises:
            ValueError: If no API key is found.
        """
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

        if provider == "auto":
            if has_openai:
                provider = "openai"
            elif has_anthropic:
                provider = "anthropic"
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
        """Build the LangGraph workflow.

        Returns:
            Compiled state graph.
        """
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
        """Process messages and decide on actions.

        Args:
            state: Current agent state.

        Returns:
            Updated state with response.
        """
        messages = state.messages

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self._get_system_prompt())] + list(messages)

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: ServiceNowState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end.

        Args:
            state: Current agent state.

        Returns:
            "continue" to call tools, "end" to finish.
        """
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
        """Process a chat message.

        Args:
            message: User message.
            thread_id: Conversation thread ID.

        Returns:
            Response dictionary with response, thread_id, and tool_calls.
        """
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
        """Async version of chat.

        Args:
            message: User message.
            thread_id: Conversation thread ID.

        Returns:
            Response dictionary with response, thread_id, and tool_calls.
        """
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
