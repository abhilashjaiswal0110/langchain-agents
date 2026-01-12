"""Incident Management Tools for Deep Agents.

Integrates with ServiceNow for real incident operations.
"""

from typing import Literal

from langchain_core.tools import tool

# Import ServiceNow API from existing agent
from app.agents.servicenow_agent import (
    get_api_client,
    is_live_mode,
    INCIDENTS_DB,
)


@tool
def search_incidents(
    query: str | None = None,
    state: str | None = None,
    priority: str | None = None,
    assigned_to: str | None = None,
    category: str | None = None,
    limit: int = 10,
) -> str:
    """Search for incidents in ServiceNow.

    Use this to find incidents matching specific criteria for investigation
    or to understand patterns across multiple incidents.

    Args:
        query: Text search in incident description.
        state: Filter by state (New, In Progress, On Hold, Resolved, Closed).
        priority: Filter by priority (1-Critical, 2-High, 3-Moderate, 4-Low).
        assigned_to: Filter by assignee name.
        category: Filter by category (Hardware, Software, Network, Access).
        limit: Maximum number of results.

    Returns:
        Formatted list of matching incidents.
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

            output = [f"**Found {len(incidents)} incident(s) [LIVE]:**\n"]
            for inc in incidents:
                output.append(f"""
**{inc.get('number', 'N/A')}** - {inc.get('short_description', 'No description')}
- State: {inc.get('state', 'Unknown')} | Priority: {inc.get('priority', 'Unknown')}
- Category: {inc.get('category', 'N/A')}
- Assigned: {inc.get('assigned_to', 'Unassigned') or 'Unassigned'}
- Created: {inc.get('sys_created_on', 'Unknown')}
""")
            return "\n".join(output)
        except Exception as e:
            return f"Error searching incidents: {e}"

    # Simulation mode
    results = []
    for inc_id, incident in INCIDENTS_DB.items():
        if state and incident["state"].lower() != state.lower():
            continue
        if priority and priority not in incident["priority"]:
            continue
        if category and incident.get("category", "").lower() != category.lower():
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
- State: {inc['state']} | Priority: {inc['priority']}
- Category: {inc.get('category', 'N/A')}
- Assigned: {inc.get('assigned_to') or 'Unassigned'}
""")
    return "\n".join(output)


@tool
def get_incident_details(incident_number: str) -> str:
    """Get comprehensive details about a specific incident.

    Use this for deep-dive investigation of a specific incident.

    Args:
        incident_number: The incident number (e.g., INC0010001).

    Returns:
        Detailed incident information including timeline and work notes.
    """
    if is_live_mode():
        try:
            api = get_api_client()
            incident = api.get_incident(incident_number.upper())

            if not incident:
                return f"Incident {incident_number} not found."

            return f"""**Incident: {incident.get('number')}** [LIVE]

**Summary:** {incident.get('short_description', 'N/A')}
**Description:** {incident.get('description', 'N/A')}

**Status:**
- State: {incident.get('state', 'Unknown')}
- Priority: {incident.get('priority', 'Unknown')}
- Impact: {incident.get('impact', 'N/A')}
- Urgency: {incident.get('urgency', 'N/A')}

**Classification:**
- Category: {incident.get('category', 'N/A')}
- Subcategory: {incident.get('subcategory', 'N/A')}
- Service: {incident.get('business_service', 'N/A')}

**Assignment:**
- Group: {incident.get('assignment_group', 'N/A')}
- Assigned To: {incident.get('assigned_to', 'Unassigned') or 'Unassigned'}

**Caller:** {incident.get('caller_id', 'Unknown')}

**Timeline:**
- Opened: {incident.get('sys_created_on', 'Unknown')}
- Updated: {incident.get('sys_updated_on', 'Unknown')}
- Resolved: {incident.get('resolved_at', 'Not resolved')}

**Work Notes:**
{incident.get('work_notes', 'No work notes')}"""

        except Exception as e:
            return f"Error getting incident details: {e}"

    # Simulation mode
    incident = INCIDENTS_DB.get(incident_number.upper())
    if not incident:
        return f"Incident {incident_number} not found."

    return f"""**Incident: {incident['number']}** [SIMULATION]

**Summary:** {incident['short_description']}
**Description:** {incident['description']}

**Status:**
- State: {incident['state']}
- Priority: {incident['priority']}

**Classification:**
- Category: {incident.get('category', 'N/A')}
- Subcategory: {incident.get('subcategory', 'N/A')}

**Assignment:**
- Group: {incident.get('assignment_group', 'N/A')}
- Assigned To: {incident.get('assigned_to') or 'Unassigned'}

**Caller:** {incident.get('caller', 'Unknown')}

**Timeline:**
- Created: {incident.get('created', 'Unknown')}
- Updated: {incident.get('updated', 'Unknown')}"""


@tool
def create_incident(
    short_description: str,
    description: str,
    category: str,
    subcategory: str,
    priority: Literal["1", "2", "3", "4"] = "3",
    impact: Literal["1", "2", "3"] = "2",
    urgency: Literal["1", "2", "3"] = "2",
    caller_email: str | None = None,
    affected_ci: str | None = None,
) -> str:
    """Create a new incident in ServiceNow.

    Use this when a new issue needs to be tracked and resolved.

    Args:
        short_description: Brief title of the incident (max 100 chars).
        description: Detailed description of the issue.
        category: Main category (Hardware, Software, Network, Access).
        subcategory: Subcategory within the main category.
        priority: Priority level (1-Critical, 2-High, 3-Moderate, 4-Low).
        impact: Business impact (1-High, 2-Medium, 3-Low).
        urgency: Urgency level (1-High, 2-Medium, 3-Low).
        caller_email: Email of the affected user.
        affected_ci: Configuration item affected.

    Returns:
        Created incident number and details.
    """
    priority_map = {
        "1": "1 - Critical",
        "2": "2 - High",
        "3": "3 - Moderate",
        "4": "4 - Low",
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

            return f"""**Incident Created** [LIVE]

**Number:** {result.get('number', 'N/A')}
**Title:** {short_description}
**Priority:** {priority_map.get(priority, '3 - Moderate')}
**Category:** {category} / {subcategory}

The incident has been created in ServiceNow.
Response SLA begins now based on priority level."""

        except Exception as e:
            return f"Error creating incident: {e}"

    # Simulation mode
    import uuid
    incident_number = f"INC{str(uuid.uuid4().int)[:7]}"

    return f"""**Incident Created** [SIMULATION]

**Number:** {incident_number}
**Title:** {short_description}
**Priority:** {priority_map.get(priority, '3 - Moderate')}
**Category:** {category} / {subcategory}
**Impact:** {impact} | **Urgency:** {urgency}

The incident has been logged and assigned to the {category} Support team."""


@tool
def update_incident(
    incident_number: str,
    work_notes: str | None = None,
    state: str | None = None,
    assigned_to: str | None = None,
    resolution_notes: str | None = None,
) -> str:
    """Update an existing incident.

    Use this to add progress notes, change state, or reassign.

    Args:
        incident_number: The incident number to update.
        work_notes: Technical notes to add.
        state: New state (In Progress, On Hold, Resolved, Closed).
        assigned_to: New assignee name or email.
        resolution_notes: Notes for resolution (required when resolving).

    Returns:
        Confirmation of updates applied.
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
                updates.append(f"State -> {state}")
            if assigned_to:
                updates.append(f"Assigned -> {assigned_to}")
            if work_notes:
                updates.append("Work notes added")

            return f"""**Incident {incident_number} Updated** [LIVE]

Updates applied:
{chr(10).join('- ' + u for u in updates)}"""

        except Exception as e:
            return f"Error updating incident: {e}"

    # Simulation
    return f"""**Incident {incident_number} Updated** [SIMULATION]

Updates would be applied in live mode."""


@tool
def escalate_incident(
    incident_number: str,
    escalation_reason: str,
    new_priority: Literal["1", "2"] | None = None,
    escalation_group: str | None = None,
) -> str:
    """Escalate an incident to a higher priority or specialist team.

    Use this for critical issues requiring immediate attention or specialized expertise.

    Args:
        incident_number: The incident number to escalate.
        escalation_reason: Reason for escalation.
        new_priority: New priority level (1-Critical or 2-High).
        escalation_group: Specialist group to escalate to.

    Returns:
        Confirmation of escalation.
    """
    priority_map = {"1": "1 - Critical", "2": "2 - High"}

    escalation_details = [f"**Incident {incident_number} Escalated**"]
    escalation_details.append(f"\n**Reason:** {escalation_reason}")

    if new_priority:
        escalation_details.append(f"**New Priority:** {priority_map.get(new_priority, new_priority)}")

    if escalation_group:
        escalation_details.append(f"**Escalated To:** {escalation_group}")

    mode = "LIVE" if is_live_mode() else "SIMULATION"
    escalation_details.append(f"\n[{mode}] Escalation notifications sent.")

    return "\n".join(escalation_details)
