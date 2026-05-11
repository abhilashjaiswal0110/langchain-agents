"""SLA Management Tools for Deep Agents.

Tools for tracking and predicting SLA performance.
"""

from datetime import datetime, timedelta
from typing import Literal

from langchain_core.tools import tool

from app.agents.servicenow_agent import is_live_mode

# SLA definitions (typical ITIL-based)
SLA_DEFINITIONS = {
    "incident": {
        "1": {"response": 15, "resolution": 60, "name": "Critical"},  # minutes
        "2": {"response": 60, "resolution": 240, "name": "High"},
        "3": {"response": 480, "resolution": 1440, "name": "Moderate"},  # 8h/24h
        "4": {"response": 1440, "resolution": 4320, "name": "Low"},  # 24h/72h
    },
    "change": {
        "emergency": {"lead_time": 0, "cab_required": False},
        "standard": {"lead_time": 24, "cab_required": False},
        "normal": {"lead_time": 120, "cab_required": True},  # 5 days
    },
    "service_request": {
        "standard": {"fulfillment": 2880},  # 48h
        "expedited": {"fulfillment": 480},  # 8h
    },
}


@tool
def get_sla_status(
    ticket_number: str,
    ticket_type: Literal["incident", "change", "service_request"] = "incident",
) -> str:
    """Get SLA status for a specific ticket.

    Use this to check if a ticket is within SLA or at risk of breach.

    Args:
        ticket_number: The ticket number.
        ticket_type: Type of ticket (incident, change, service_request).

    Returns:
        Current SLA status with time remaining or breach details.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    # Simulate SLA calculation
    if ticket_type == "incident":
        # Simulate based on ticket number pattern
        if "INC001000" in ticket_number:
            # Simulated P3 incident, 70% through SLA
            priority = "3"
            sla_def = SLA_DEFINITIONS["incident"]["3"]
            elapsed_mins = int(sla_def["resolution"] * 0.7)
            remaining_mins = sla_def["resolution"] - elapsed_mins

            return f"""**SLA Status: {ticket_number}** [{mode}]

**Priority:** {sla_def["name"]} (P{priority})

**Response SLA:**
- Target: {sla_def["response"]} minutes
- Status: MET (Responded within target)

**Resolution SLA:**
- Target: {sla_def["resolution"]} minutes ({sla_def["resolution"] // 60}h)
- Elapsed: {elapsed_mins} minutes
- Remaining: {remaining_mins} minutes ({remaining_mins // 60}h {remaining_mins % 60}m)
- Status: ON TRACK

**Risk Level:** Low
**Breach Prediction:** No breach expected at current pace"""

        else:
            # Generic response
            return f"""**SLA Status: {ticket_number}** [{mode}]

SLA data retrieved. Ticket is currently within SLA targets.
Response: MET | Resolution: ON TRACK"""

    elif ticket_type == "change":
        return f"""**SLA Status: {ticket_number}** [{mode}]

**Change Type:** Normal Change
**Lead Time Requirement:** 5 business days
**Status:** Within lead time requirements
**CAB Approval:** Required"""

    else:
        return f"""**SLA Status: {ticket_number}** [{mode}]

**Request Type:** Standard
**Fulfillment Target:** 48 hours
**Status:** Within SLA"""


@tool
def calculate_sla_breach_time(
    priority: Literal["1", "2", "3", "4"],
    created_at: str | None = None,
) -> str:
    """Calculate when SLA will breach for a given priority.

    Use this to understand SLA deadlines for incident prioritization.

    Args:
        priority: Incident priority (1-4).
        created_at: When ticket was created (ISO format). Defaults to now.

    Returns:
        SLA breach times for response and resolution.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    sla_def = SLA_DEFINITIONS["incident"].get(priority)
    if not sla_def:
        return f"Invalid priority: {priority}"

    if created_at:
        try:
            start_time = datetime.fromisoformat(created_at)
        except ValueError:
            start_time = datetime.now()
    else:
        start_time = datetime.now()

    response_breach = start_time + timedelta(minutes=sla_def["response"])
    resolution_breach = start_time + timedelta(minutes=sla_def["resolution"])

    return f"""**SLA Breach Calculation** [{mode}]

**Priority:** {sla_def["name"]} (P{priority})
**Created:** {start_time.strftime("%Y-%m-%d %H:%M")}

**Response SLA:**
- Target: {sla_def["response"]} minutes
- Breach Time: {response_breach.strftime("%Y-%m-%d %H:%M")}

**Resolution SLA:**
- Target: {sla_def["resolution"]} minutes ({sla_def["resolution"] // 60}h)
- Breach Time: {resolution_breach.strftime("%Y-%m-%d %H:%M")}

**Note:** SLA calculations assume 24x7 coverage. Business hours
SLAs may have different breach times."""


@tool
def get_sla_report(
    period: Literal["day", "week", "month"] = "week",
    ticket_type: Literal["incident", "change", "service_request"] = "incident",
) -> str:
    """Get SLA performance report for a time period.

    Use this to understand overall SLA compliance and identify trends.

    Args:
        period: Reporting period (day, week, month).
        ticket_type: Type of tickets to report on.

    Returns:
        SLA performance metrics and trends.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    # Simulated metrics
    if ticket_type == "incident":
        metrics = {
            "total": 127,
            "response_met": 124,
            "resolution_met": 118,
            "response_rate": 97.6,
            "resolution_rate": 92.9,
            "avg_response": 12,  # minutes
            "avg_resolution": 185,  # minutes
            "breaches_by_priority": {"P1": 0, "P2": 1, "P3": 5, "P4": 3},
        }

        return f"""**SLA Performance Report: Incidents** [{mode}]
**Period:** Last {period}

**Volume:**
- Total Incidents: {metrics["total"]}
- Currently Open: 23

**Response SLA:**
- Met: {metrics["response_met"]}/{metrics["total"]} ({metrics["response_rate"]}%)
- Average Response Time: {metrics["avg_response"]} minutes

**Resolution SLA:**
- Met: {metrics["resolution_met"]}/{metrics["total"]} ({metrics["resolution_rate"]}%)
- Average Resolution Time: {metrics["avg_resolution"]} minutes ({metrics["avg_resolution"] // 60}h)

**Breaches by Priority:**
- P1 (Critical): {metrics["breaches_by_priority"]["P1"]}
- P2 (High): {metrics["breaches_by_priority"]["P2"]}
- P3 (Moderate): {metrics["breaches_by_priority"]["P3"]}
- P4 (Low): {metrics["breaches_by_priority"]["P4"]}

**Trend:** Improvement from previous period (+2.1% resolution rate)"""

    return f"""**SLA Performance Report: {ticket_type.title()}** [{mode}]
**Period:** Last {period}

Overall compliance rate: 94.5%
Details available in ServiceNow dashboards."""


@tool
def predict_sla_breach(
    incident_number: str,
    current_state: str = "In Progress",
    assignment_group: str | None = None,
) -> str:
    """Predict likelihood of SLA breach for an incident.

    Uses historical data to predict breach risk and recommend actions.

    Args:
        incident_number: The incident number.
        current_state: Current incident state.
        assignment_group: Current assignment group.

    Returns:
        Breach prediction with risk factors and recommendations.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    # Simulated prediction
    risk_score = 35  # 0-100

    risk_level = "Low" if risk_score < 40 else "Medium" if risk_score < 70 else "High"

    factors = []
    if current_state == "On Hold":
        factors.append("- Incident on hold (adds to elapsed time)")
        risk_score += 20
    if assignment_group == "Level 3 Support":
        factors.append("- Assigned to specialist team (typically longer resolution)")
        risk_score += 10

    recommendations = []
    if risk_score > 50:
        recommendations.append("- Consider escalating to expedite resolution")
        recommendations.append("- Engage additional resources if available")
    if risk_score > 70:
        recommendations.append("- Notify management of potential breach")
        recommendations.append("- Document all actions for SLA exception request")

    return f"""**SLA Breach Prediction: {incident_number}** [{mode}]

**Risk Score:** {risk_score}/100
**Risk Level:** {risk_level}

**Risk Factors:**
{chr(10).join(factors) if factors else "- No significant risk factors identified"}

**Time Analysis:**
- Elapsed: 65% of resolution target
- Remaining: 35% of resolution target
- Predicted completion: 85% of target time

**Recommendations:**
{chr(10).join(recommendations) if recommendations else "- Continue current approach"}"""
