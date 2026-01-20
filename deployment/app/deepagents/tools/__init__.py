"""Deep Agents Tools Module.

This module provides specialized tools for Deep Agents including:
- IT Managed Services operations (integrating with ServiceNow)
- Sales & Pre-Sales Intelligence (CRM, proposals, pricing, analytics)
"""

# =============================================================================
# IT Operations Tools
# =============================================================================
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

# =============================================================================
# Sales & Pre-Sales Intelligence Tools
# =============================================================================
from app.deepagents.tools.crm_tools import (
    search_opportunities,
    get_deal_details,
    update_opportunity_stage,
    get_customer_history,
    get_pipeline_summary,
)
from app.deepagents.tools.proposal_tools import (
    search_rfp_templates,
    get_template_details,
    extract_requirements,
    draft_proposal_section,
    generate_executive_summary,
    search_past_proposals,
)
from app.deepagents.tools.competitor_tools import (
    get_competitive_analysis,
    compare_solutions,
    suggest_differentiators,
    get_objection_handler,
)
from app.deepagents.tools.pricing_tools import (
    calculate_pricing,
    analyze_margin,
    generate_pricing_options,
    get_pricing_model_recommendation,
)
from app.deepagents.tools.analytics_tools import (
    calculate_win_probability,
    assess_deal_risk,
    get_similar_deals,
    get_sales_performance_summary,
)

# =============================================================================
# Document Tools (Shared by both agents)
# =============================================================================
from app.deepagents.tools.document_tools import (
    search_attachments,
    list_attachments,
    get_attachment_summary,
    clear_attachments,
    process_and_store_document,
    get_document_context,
)

__all__ = [
    # ==========================================================================
    # IT Operations Tools
    # ==========================================================================
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
    # ==========================================================================
    # Sales & Pre-Sales Intelligence Tools
    # ==========================================================================
    # CRM tools
    "search_opportunities",
    "get_deal_details",
    "update_opportunity_stage",
    "get_customer_history",
    "get_pipeline_summary",
    # Proposal tools
    "search_rfp_templates",
    "get_template_details",
    "extract_requirements",
    "draft_proposal_section",
    "generate_executive_summary",
    "search_past_proposals",
    # Competitor tools
    "get_competitive_analysis",
    "compare_solutions",
    "suggest_differentiators",
    "get_objection_handler",
    # Pricing tools
    "calculate_pricing",
    "analyze_margin",
    "generate_pricing_options",
    "get_pricing_model_recommendation",
    # Analytics tools
    "calculate_win_probability",
    "assess_deal_risk",
    "get_similar_deals",
    "get_sales_performance_summary",
    # ==========================================================================
    # Document Tools (Shared)
    # ==========================================================================
    "search_attachments",
    "list_attachments",
    "get_attachment_summary",
    "clear_attachments",
    "process_and_store_document",
    "get_document_context",
]
