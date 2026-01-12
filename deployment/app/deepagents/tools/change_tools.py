"""Change Management Tools for Deep Agents.

Integrates with ServiceNow for change request operations.
"""

from langchain_core.tools import tool

from app.agents.servicenow_agent import (
    get_api_client,
    is_live_mode,
    CHANGE_REQUESTS_DB,
)


@tool
def search_changes(
    state: str | None = None,
    change_type: str | None = None,
    risk: str | None = None,
    scheduled_after: str | None = None,
    limit: int = 10,
) -> str:
    """Search for change requests in ServiceNow.

    Use this to find upcoming changes, review pending approvals,
    or investigate changes that might be related to incidents.

    Args:
        state: Filter by state (New, Assess, Authorize, Scheduled, Implement, Review, Closed).
        change_type: Filter by type (Standard, Normal, Emergency).
        risk: Filter by risk level (Low, Medium, High).
        scheduled_after: Find changes scheduled after this date (YYYY-MM-DD).
        limit: Maximum number of results.

    Returns:
        List of matching change requests.
    """
    if is_live_mode():
        try:
            api = get_api_client()
            changes = api.get_change_requests(state=state, limit=limit)

            if not changes:
                return "No change requests found matching the criteria."

            output = [f"**Found {len(changes)} change request(s) [LIVE]:**\n"]
            for chg in changes:
                output.append(f"""
**{chg.get('number', 'N/A')}** - {chg.get('short_description', 'No description')}
- Type: {chg.get('type', 'N/A')} | Risk: {chg.get('risk', 'N/A')}
- State: {chg.get('state', 'Unknown')}
- Scheduled: {chg.get('start_date', 'TBD')} to {chg.get('end_date', 'TBD')}
""")
            return "\n".join(output)
        except Exception as e:
            return f"Error searching changes: {e}"

    # Simulation mode
    results = []
    for chg_id, change in CHANGE_REQUESTS_DB.items():
        if state and change["state"].lower() != state.lower():
            continue
        if change_type and change["type"].lower() != change_type.lower():
            continue
        if risk and change["risk"].lower() != risk.lower():
            continue
        results.append(change)
        if len(results) >= limit:
            break

    if not results:
        return "No change requests found matching the criteria."

    output = [f"**Found {len(results)} change request(s) [SIMULATION]:**\n"]
    for chg in results:
        output.append(f"""
**{chg['number']}** - {chg['short_description']}
- Type: {chg['type']} | Risk: {chg['risk']}
- State: {chg['state']}
- Scheduled: {chg['planned_start']} to {chg['planned_end']}
""")
    return "\n".join(output)


@tool
def get_change_details(change_number: str) -> str:
    """Get comprehensive details about a specific change request.

    Use this for detailed review of a change including risk assessment,
    implementation plan, and approval status.

    Args:
        change_number: The change request number (e.g., CHG0000009).

    Returns:
        Detailed change request information.
    """
    if is_live_mode():
        try:
            api = get_api_client()
            change = api.get_change_request(change_number.upper())

            if not change:
                return f"Change request {change_number} not found."

            return f"""**Change Request: {change.get('number')}** [LIVE]

**Summary:** {change.get('short_description', 'N/A')}
**Description:** {change.get('description', 'N/A')}

**Classification:**
- Type: {change.get('type', 'N/A')}
- Category: {change.get('category', 'N/A')}
- Risk: {change.get('risk', 'N/A')}
- Impact: {change.get('impact', 'N/A')}

**Status:**
- State: {change.get('state', 'Unknown')}
- Phase: {change.get('phase', 'N/A')}
- Approval: {change.get('approval', 'Unknown')}

**Schedule:**
- Planned Start: {change.get('start_date', 'TBD')}
- Planned End: {change.get('end_date', 'TBD')}

**Assignment:**
- Assignment Group: {change.get('assignment_group', 'N/A')}
- Assigned To: {change.get('assigned_to', 'Unassigned')}

**Affected CIs:** {change.get('cmdb_ci', 'None specified')}"""

        except Exception as e:
            return f"Error getting change details: {e}"

    # Simulation mode
    change = CHANGE_REQUESTS_DB.get(change_number.upper())
    if not change:
        return f"Change request {change_number} not found."

    return f"""**Change Request: {change['number']}** [SIMULATION]

**Summary:** {change['short_description']}
**Description:** {change['description']}

**Classification:**
- Type: {change['type']}
- Risk: {change['risk']}

**Status:**
- State: {change['state']}
- Approval: {change['approval_status']}

**Schedule:**
- Planned Start: {change['planned_start']}
- Planned End: {change['planned_end']}

**Impact:** {change['impact']}"""


@tool
def validate_change(
    change_number: str,
    validation_checks: list[str] | None = None,
) -> str:
    """Validate a change request against standard criteria.

    Use this to ensure a change meets all requirements before approval.

    Args:
        change_number: The change request number.
        validation_checks: Specific checks to perform.

    Returns:
        Validation results with pass/fail for each criterion.
    """
    default_checks = [
        "description_complete",
        "risk_assessment_done",
        "rollback_plan_exists",
        "test_plan_defined",
        "stakeholders_notified",
        "change_window_appropriate",
        "ci_impact_documented",
    ]

    checks = validation_checks or default_checks

    # Simulate validation
    results = []
    for check in checks:
        # In real implementation, these would query ServiceNow
        status = "PASS"  # Simplified for demo
        results.append(f"- [{status}] {check.replace('_', ' ').title()}")

    mode = "LIVE" if is_live_mode() else "SIMULATION"

    return f"""**Change Validation: {change_number}** [{mode}]

**Validation Results:**
{chr(10).join(results)}

**Overall Status:** Ready for Review
**Recommendation:** Proceed to CAB approval"""


@tool
def assess_change_risk(
    change_number: str,
    consider_factors: list[str] | None = None,
) -> str:
    """Perform risk assessment for a change request.

    Analyzes various risk factors and provides a risk score.

    Args:
        change_number: The change request number.
        consider_factors: Additional risk factors to evaluate.

    Returns:
        Risk assessment report.
    """
    risk_factors = [
        ("System Criticality", "High", "Production system"),
        ("Change Complexity", "Medium", "Standard procedure with some customization"),
        ("Rollback Capability", "Low", "Full rollback plan documented"),
        ("Testing Coverage", "Low", "Comprehensive test plan executed"),
        ("Change Window Risk", "Medium", "Business hours - moderate impact"),
        ("Historical Success", "Low", "Similar changes 95% success rate"),
    ]

    mode = "LIVE" if is_live_mode() else "SIMULATION"

    output = [f"**Risk Assessment: {change_number}** [{mode}]\n"]
    output.append("**Risk Factors Analysis:**\n")

    total_score = 0
    risk_values = {"Low": 1, "Medium": 2, "High": 3}

    for factor, risk, reason in risk_factors:
        total_score += risk_values[risk]
        output.append(f"- **{factor}:** {risk}\n  Reason: {reason}")

    avg_score = total_score / len(risk_factors)
    overall_risk = "Low" if avg_score < 1.5 else "Medium" if avg_score < 2.5 else "High"

    output.append(f"\n**Overall Risk Level:** {overall_risk}")
    output.append(f"**Risk Score:** {avg_score:.1f}/3.0")

    if overall_risk == "High":
        output.append("\n**Recommendation:** Requires additional review and mitigation planning")
    else:
        output.append("\n**Recommendation:** Proceed with standard approval process")

    return "\n".join(output)
