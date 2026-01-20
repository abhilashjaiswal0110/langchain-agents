"""Problem Management Tools for Deep Agents.

Tools for root cause analysis and problem record management.
"""

import uuid
from typing import Literal

from langchain_core.tools import tool

from app.agents.servicenow_agent import is_live_mode


# Simulated problem database
PROBLEMS_DB = {
    "PRB0000001": {
        "number": "PRB0000001",
        "short_description": "Recurring VPN connectivity issues",
        "description": "Multiple incidents reported for VPN disconnections affecting remote workers.",
        "state": "Under Investigation",
        "priority": "2 - High",
        "root_cause": None,
        "workaround": "Reconnect to VPN and clear DNS cache",
        "related_incidents": ["INC0010002", "INC0010005", "INC0010008"],
        "affected_ci": "VPN-GATEWAY-01",
        "assigned_to": "Network Team",
        "created": "2024-12-10",
    },
    "PRB0000002": {
        "number": "PRB0000002",
        "short_description": "Email sync failures on mobile devices",
        "description": "Pattern of email sync issues identified across iOS devices.",
        "state": "Known Error",
        "priority": "3 - Moderate",
        "root_cause": "ActiveSync timeout configuration too aggressive",
        "workaround": "Remove and re-add email account",
        "related_incidents": ["INC0010001"],
        "affected_ci": "EXCHANGE-01",
        "assigned_to": "Email Support",
        "created": "2024-12-08",
    },
}


@tool
def search_problems(
    query: str | None = None,
    state: str | None = None,
    priority: str | None = None,
    has_root_cause: bool | None = None,
    limit: int = 10,
) -> str:
    """Search for problem records in ServiceNow.

    Use this to find existing problems that might be related to
    current incidents or to identify patterns.

    Args:
        query: Text search in problem description.
        state: Filter by state (New, Under Investigation, Known Error, Closed).
        priority: Filter by priority (1-Critical, 2-High, 3-Moderate, 4-Low).
        has_root_cause: Filter by whether root cause is identified.
        limit: Maximum number of results.

    Returns:
        List of matching problem records.
    """
    results = []

    for prb_id, problem in PROBLEMS_DB.items():
        if state and problem["state"].lower() != state.lower():
            continue
        if priority and priority not in problem["priority"]:
            continue
        if has_root_cause is not None:
            has_rca = problem["root_cause"] is not None
            if has_root_cause != has_rca:
                continue
        if query:
            query_lower = query.lower()
            if query_lower not in problem["short_description"].lower() and query_lower not in problem["description"].lower():
                continue

        results.append(problem)
        if len(results) >= limit:
            break

    mode = "LIVE" if is_live_mode() else "SIMULATION"

    if not results:
        return f"No problem records found matching the criteria. [{mode}]"

    output = [f"**Found {len(results)} problem record(s) [{mode}]:**\n"]
    for prb in results:
        rca_status = "RCA Identified" if prb["root_cause"] else "Under Investigation"
        output.append(f"""
**{prb['number']}** - {prb['short_description']}
- State: {prb['state']} | Priority: {prb['priority']}
- Root Cause: {rca_status}
- Related Incidents: {len(prb['related_incidents'])}
""")

    return "\n".join(output)


@tool
def get_problem_details(problem_number: str) -> str:
    """Get comprehensive details about a specific problem record.

    Use this for detailed investigation of a problem including
    root cause analysis and related incidents.

    Args:
        problem_number: The problem number (e.g., PRB0000001).

    Returns:
        Detailed problem record information.
    """
    problem = PROBLEMS_DB.get(problem_number.upper())

    mode = "LIVE" if is_live_mode() else "SIMULATION"

    if not problem:
        return f"Problem record {problem_number} not found. [{mode}]"

    related = ", ".join(problem["related_incidents"]) or "None"

    return f"""**Problem Record: {problem['number']}** [{mode}]

**Summary:** {problem['short_description']}
**Description:** {problem['description']}

**Status:**
- State: {problem['state']}
- Priority: {problem['priority']}

**Analysis:**
- Root Cause: {problem['root_cause'] or 'Under investigation'}
- Workaround: {problem['workaround'] or 'None available'}

**Impact:**
- Affected CI: {problem['affected_ci']}
- Related Incidents: {related}

**Assignment:**
- Assigned To: {problem['assigned_to']}
- Created: {problem['created']}"""


@tool
def create_problem(
    short_description: str,
    description: str,
    priority: Literal["1", "2", "3", "4"] = "3",
    related_incidents: list[str] | None = None,
    affected_ci: str | None = None,
) -> str:
    """Create a new problem record for root cause analysis.

    Use this when a pattern of incidents suggests an underlying problem.

    Args:
        short_description: Brief title of the problem.
        description: Detailed description of the pattern observed.
        priority: Priority level (1-Critical, 2-High, 3-Moderate, 4-Low).
        related_incidents: List of related incident numbers.
        affected_ci: Primary configuration item affected.

    Returns:
        Created problem record number and details.
    """
    problem_number = f"PRB{str(uuid.uuid4().int)[:7]}"
    priority_map = {
        "1": "1 - Critical",
        "2": "2 - High",
        "3": "3 - Moderate",
        "4": "4 - Low",
    }

    mode = "LIVE" if is_live_mode() else "SIMULATION"

    incidents_list = ", ".join(related_incidents) if related_incidents else "None"

    return f"""**Problem Record Created** [{mode}]

**Number:** {problem_number}
**Title:** {short_description}
**Priority:** {priority_map.get(priority, '3 - Moderate')}
**State:** New

**Related Incidents:** {incidents_list}
**Affected CI:** {affected_ci or 'To be determined'}

Next Steps:
1. Assign to appropriate team for investigation
2. Gather data from related incidents
3. Perform root cause analysis
4. Document workaround if available"""


@tool
def link_incidents_to_problem(
    problem_number: str,
    incident_numbers: list[str],
) -> str:
    """Link incidents to a problem record.

    Use this to associate related incidents with a problem for pattern analysis.

    Args:
        problem_number: The problem number.
        incident_numbers: List of incident numbers to link.

    Returns:
        Confirmation of linked incidents.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    return f"""**Incidents Linked to {problem_number}** [{mode}]

Linked Incidents:
{chr(10).join('- ' + inc for inc in incident_numbers)}

Total Linked: {len(incident_numbers)}

The incidents are now associated with this problem record for
root cause analysis and pattern identification."""


@tool
def create_known_error(
    problem_number: str,
    root_cause: str,
    workaround: str,
    permanent_fix: str | None = None,
) -> str:
    """Convert a problem to a Known Error with documented solution.

    Use this when root cause is identified and a workaround is available.

    Args:
        problem_number: The problem number.
        root_cause: Identified root cause.
        workaround: Documented workaround steps.
        permanent_fix: Planned permanent fix if known.

    Returns:
        Known Error documentation.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    fix_info = f"\n**Permanent Fix:** {permanent_fix}" if permanent_fix else ""

    return f"""**Known Error Created** [{mode}]

**Problem:** {problem_number}
**State:** Known Error

**Root Cause:**
{root_cause}

**Workaround:**
{workaround}
{fix_info}

This Known Error is now available in the knowledge base for
Service Desk reference when handling related incidents."""
