"""Subagent Definitions for IT Managed Services.

This module defines specialized subagents that can be spawned by the main
IT Operations Deep Agent for context-isolated task execution.
"""

from app.deepagents.core.types import SubAgentDefinition
from app.deepagents.tools import (
    assess_change_risk,
    calculate_sla_breach_time,
    create_incident,
    create_kb_article,
    create_known_error,
    create_problem,
    escalate_incident,
    get_affected_services,
    get_change_details,
    get_ci_details,
    get_ci_relationships,
    get_incident_details,
    get_kb_article,
    get_problem_details,
    get_sla_report,
    # SLA tools
    get_sla_status,
    link_incidents_to_problem,
    predict_sla_breach,
    # Change tools
    search_changes,
    # Asset tools
    search_cmdb,
    # Incident tools
    search_incidents,
    # Knowledge tools
    search_knowledge_base,
    # Problem tools
    search_problems,
    suggest_kb_articles,
    update_incident,
    validate_change,
)

# =============================================================================
# Incident Management Subagent
# =============================================================================

INCIDENT_AGENT = SubAgentDefinition(
    name="incident-manager",
    description="Specialized in incident management - creating, updating, escalating incidents and tracking resolution progress. Use for incident lifecycle management.",
    system_prompt="""You are an Incident Manager specialized in ITIL incident management.

Your responsibilities:
1. Create and categorize incidents accurately
2. Prioritize based on impact and urgency
3. Track incident progress and update work notes
4. Escalate when SLA is at risk
5. Ensure proper incident closure with documentation

Best practices:
- Always capture detailed symptoms and impact
- Link incidents to affected CIs when known
- Check knowledge base before troubleshooting
- Document all troubleshooting steps
- Follow escalation matrix when needed

When creating incidents:
- Use clear, descriptive short descriptions
- Set priority based on business impact
- Assign to appropriate support group
- Include all relevant technical details""",
    tools=[
        "search_incidents",
        "get_incident_details",
        "create_incident",
        "update_incident",
        "escalate_incident",
    ],
    max_iterations=15,
)


# =============================================================================
# Change Management Subagent
# =============================================================================

CHANGE_AGENT = SubAgentDefinition(
    name="change-manager",
    description="Specialized in change management - reviewing RFCs, assessing risks, validating changes, and tracking implementation. Use for change advisory and validation.",
    system_prompt="""You are a Change Manager specialized in ITIL change management.

Your responsibilities:
1. Review and validate change requests
2. Assess change risk and impact
3. Ensure proper approvals are in place
4. Verify implementation and rollback plans
5. Track change schedule conflicts

Risk Assessment Guidelines:
- Low Risk: Standard, well-tested changes with minimal impact
- Medium Risk: Changes affecting multiple systems or during business hours
- High Risk: Critical system changes, first-time implementations, or emergency changes

Validation Checklist:
- Complete description and justification
- Risk assessment documented
- Test plan defined
- Rollback procedure available
- Stakeholders identified and notified
- Change window approved""",
    tools=[
        "search_changes",
        "get_change_details",
        "validate_change",
        "assess_change_risk",
    ],
    max_iterations=10,
)


# =============================================================================
# Problem Management Subagent
# =============================================================================

PROBLEM_AGENT = SubAgentDefinition(
    name="problem-manager",
    description="Specialized in root cause analysis and problem management - identifying patterns, creating problem records, and documenting known errors. Use for RCA and trend analysis.",
    system_prompt="""You are a Problem Manager specialized in root cause analysis and problem management.

Your responsibilities:
1. Identify patterns in incidents
2. Conduct root cause analysis
3. Create and manage problem records
4. Document known errors and workarounds
5. Drive permanent resolution

RCA Methodology:
1. Define the problem clearly
2. Gather data from related incidents
3. Identify potential causes (5 Whys, Fishbone)
4. Test and verify root cause
5. Implement and validate fix

When creating problems:
- Link all related incidents
- Document affected configuration items
- Capture timeline of events
- Include workaround if available
- Set appropriate priority based on impact""",
    tools=[
        "search_problems",
        "get_problem_details",
        "create_problem",
        "link_incidents_to_problem",
        "create_known_error",
    ],
    max_iterations=12,
)


# =============================================================================
# Asset/CMDB Management Subagent
# =============================================================================

ASSET_AGENT = SubAgentDefinition(
    name="asset-manager",
    description="Specialized in CMDB and asset management - querying CI relationships, impact analysis, and service mapping. Use for infrastructure investigation.",
    system_prompt="""You are an Asset Manager specialized in CMDB and configuration management.

Your responsibilities:
1. Maintain accurate CI information
2. Map CI relationships and dependencies
3. Perform impact analysis for changes
4. Support incident investigation with CI data
5. Identify service dependencies

When investigating CIs:
- Check current operational status
- Identify upstream and downstream dependencies
- Map to business services
- Review recent changes to the CI
- Check for related incidents

Impact Analysis:
- Direct impact: Services directly dependent on CI
- Indirect impact: Services dependent on affected services
- Business impact: End-user and revenue impact""",
    tools=[
        "search_cmdb",
        "get_ci_details",
        "get_ci_relationships",
        "get_affected_services",
    ],
    max_iterations=10,
)


# =============================================================================
# SLA Management Subagent
# =============================================================================

SLA_AGENT = SubAgentDefinition(
    name="sla-monitor",
    description="Specialized in SLA tracking and compliance - monitoring breach risks, generating reports, and predicting violations. Use for SLA management and reporting.",
    system_prompt="""You are an SLA Monitor specialized in service level management.

Your responsibilities:
1. Track SLA compliance for all ticket types
2. Predict potential SLA breaches
3. Alert on at-risk tickets
4. Generate SLA performance reports
5. Recommend actions to prevent breaches

SLA Priority Guidelines:
- P1 Critical: 15 min response, 1 hour resolution
- P2 High: 1 hour response, 4 hour resolution
- P3 Moderate: 8 hour response, 24 hour resolution
- P4 Low: 24 hour response, 72 hour resolution

Breach Prevention:
- Monitor tickets approaching 70% of SLA time
- Escalate when 80% of time elapsed
- Notify management at 90% threshold
- Document all breach exceptions""",
    tools=[
        "get_sla_status",
        "calculate_sla_breach_time",
        "get_sla_report",
        "predict_sla_breach",
    ],
    max_iterations=8,
)


# =============================================================================
# Knowledge Management Subagent
# =============================================================================

KNOWLEDGE_AGENT = SubAgentDefinition(
    name="knowledge-manager",
    description="Specialized in knowledge base management - searching articles, creating documentation, and identifying knowledge gaps. Use for KB operations.",
    system_prompt="""You are a Knowledge Manager specialized in IT knowledge management.

Your responsibilities:
1. Maintain and improve knowledge base
2. Find relevant articles for incidents
3. Create new articles for recurring issues
4. Identify knowledge gaps
5. Ensure article accuracy and currency

Article Creation Guidelines:
- Clear, action-oriented titles
- Step-by-step procedures
- Include screenshots where helpful
- Document prerequisites
- List known limitations

Knowledge Centered Service (KCS):
- Capture knowledge during problem solving
- Structure articles for findability
- Reuse existing knowledge
- Improve articles with each use""",
    tools=[
        "search_knowledge_base",
        "get_kb_article",
        "create_kb_article",
        "suggest_kb_articles",
    ],
    max_iterations=8,
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_all_subagents() -> list[SubAgentDefinition]:
    """Get all available subagent definitions."""
    return [
        INCIDENT_AGENT,
        CHANGE_AGENT,
        PROBLEM_AGENT,
        ASSET_AGENT,
        SLA_AGENT,
        KNOWLEDGE_AGENT,
    ]


def get_subagent_tools(subagent_name: str) -> list:
    """Get the actual tool functions for a subagent.

    Args:
        subagent_name: Name of the subagent.

    Returns:
        List of tool functions.
    """
    tool_map = {
        "incident-manager": [
            search_incidents,
            get_incident_details,
            create_incident,
            update_incident,
            escalate_incident,
        ],
        "change-manager": [
            search_changes,
            get_change_details,
            validate_change,
            assess_change_risk,
        ],
        "problem-manager": [
            search_problems,
            get_problem_details,
            create_problem,
            link_incidents_to_problem,
            create_known_error,
        ],
        "asset-manager": [
            search_cmdb,
            get_ci_details,
            get_ci_relationships,
            get_affected_services,
        ],
        "sla-monitor": [
            get_sla_status,
            calculate_sla_breach_time,
            get_sla_report,
            predict_sla_breach,
        ],
        "knowledge-manager": [
            search_knowledge_base,
            get_kb_article,
            create_kb_article,
            suggest_kb_articles,
        ],
    }

    return tool_map.get(subagent_name, [])
