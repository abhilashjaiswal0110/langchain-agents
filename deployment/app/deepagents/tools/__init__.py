"""IT Operations Tools for Deep Agents.

This module provides specialized tools for IT Managed Services operations,
integrating with ServiceNow for real ITSM data.
"""

from app.deepagents.tools.incident_tools import (
    search_incidents,
    get_incident_details,
    create_incident,
    update_incident,
    escalate_incident,
)
from app.deepagents.tools.change_tools import (
    search_changes,
    get_change_details,
    validate_change,
    assess_change_risk,
)
from app.deepagents.tools.problem_tools import (
    search_problems,
    get_problem_details,
    create_problem,
    link_incidents_to_problem,
    create_known_error,
)
from app.deepagents.tools.asset_tools import (
    search_cmdb,
    get_ci_details,
    get_ci_relationships,
    get_affected_services,
)
from app.deepagents.tools.sla_tools import (
    get_sla_status,
    calculate_sla_breach_time,
    get_sla_report,
    predict_sla_breach,
)
from app.deepagents.tools.knowledge_tools import (
    search_knowledge_base,
    get_kb_article,
    create_kb_article,
    suggest_kb_articles,
)

__all__ = [
    # Incident tools
    "search_incidents",
    "get_incident_details",
    "create_incident",
    "update_incident",
    "escalate_incident",
    # Change tools
    "search_changes",
    "get_change_details",
    "validate_change",
    "assess_change_risk",
    # Problem tools
    "search_problems",
    "get_problem_details",
    "create_problem",
    "link_incidents_to_problem",
    "create_known_error",
    # Asset tools
    "search_cmdb",
    "get_ci_details",
    "get_ci_relationships",
    "get_affected_services",
    # SLA tools
    "get_sla_status",
    "calculate_sla_breach_time",
    "get_sla_report",
    "predict_sla_breach",
    # Knowledge tools
    "search_knowledge_base",
    "get_kb_article",
    "create_kb_article",
    "suggest_kb_articles",
]
