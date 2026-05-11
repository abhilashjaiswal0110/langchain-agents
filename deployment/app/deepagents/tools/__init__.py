"""Deep Agents Tools Module.

This module provides specialized tools for Deep Agents including:
- IT Managed Services operations (integrating with ServiceNow)
- Sales & Pre-Sales Intelligence (CRM, proposals, pricing, analytics)
- Recruitment & Talent Acquisition (SharePoint, screening, interviews, scoring)
"""

# =============================================================================
# IT Operations Tools
# =============================================================================
from app.deepagents.tools.analytics_tools import (
    assess_deal_risk,
    calculate_win_probability,
    get_sales_performance_summary,
    get_similar_deals,
)
from app.deepagents.tools.asset_tools import (
    get_affected_services,
    get_ci_details,
    get_ci_relationships,
    search_cmdb,
)
from app.deepagents.tools.change_tools import (
    assess_change_risk,
    get_change_details,
    search_changes,
    validate_change,
)
from app.deepagents.tools.competitor_tools import (
    compare_solutions,
    get_competitive_analysis,
    get_objection_handler,
    suggest_differentiators,
)

# =============================================================================
# Sales & Pre-Sales Intelligence Tools
# =============================================================================
from app.deepagents.tools.crm_tools import (
    get_customer_history,
    get_deal_details,
    get_pipeline_summary,
    search_opportunities,
    update_opportunity_stage,
)

# =============================================================================
# Document Tools (Shared by both agents)
# =============================================================================
from app.deepagents.tools.document_tools import (
    clear_attachments,
    get_attachment_summary,
    get_current_session,
    get_document_context,
    list_attachments,
    process_and_store_document,
    search_attachments,
    set_current_session,
)
from app.deepagents.tools.incident_tools import (
    create_incident,
    escalate_incident,
    get_incident_details,
    search_incidents,
    update_incident,
)
from app.deepagents.tools.interview_tools import (
    evaluate_candidate_answers,
    export_question_set,
    generate_interview_questions,
    get_candidate_score,
    list_question_sets,
    submit_candidate_answers,
)
from app.deepagents.tools.knowledge_tools import (
    create_kb_article,
    get_kb_article,
    search_knowledge_base,
    suggest_kb_articles,
)
from app.deepagents.tools.pricing_tools import (
    analyze_margin,
    calculate_pricing,
    generate_pricing_options,
    get_pricing_model_recommendation,
)
from app.deepagents.tools.problem_tools import (
    create_known_error,
    create_problem,
    get_problem_details,
    link_incidents_to_problem,
    search_problems,
)
from app.deepagents.tools.proposal_tools import (
    draft_proposal_section,
    extract_requirements,
    generate_executive_summary,
    get_template_details,
    search_past_proposals,
    search_rfp_templates,
)
from app.deepagents.tools.recruitment_tools import (
    batch_screen_resumes,
    clear_session_data,
    get_candidate_profile,
    get_session_dashboard,
    get_shortlisted_candidates,
    list_candidates,
    list_job_descriptions,
    parse_job_description,
    parse_resume,
    screen_candidate,
)
from app.deepagents.tools.scoring_tools import (
    export_scoring_excel,
    generate_scoring_report,
    generate_shortlist_report,
    get_passing_score_thresholds,
    get_ranking_summary,
)

# =============================================================================
# Recruitment & Talent Acquisition Tools
# =============================================================================
from app.deepagents.tools.sharepoint_tools import (
    create_sharepoint_folder,
    download_sharepoint_document,
    get_cached_document,
    list_sharepoint_folder,
    search_sharepoint_documents,
    upload_to_sharepoint,
)
from app.deepagents.tools.sla_tools import (
    calculate_sla_breach_time,
    get_sla_report,
    get_sla_status,
    predict_sla_breach,
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
    "set_current_session",
    "get_current_session",
    # ==========================================================================
    # Recruitment & Talent Acquisition Tools
    # ==========================================================================
    # SharePoint tools
    "list_sharepoint_folder",
    "download_sharepoint_document",
    "upload_to_sharepoint",
    "search_sharepoint_documents",
    "get_cached_document",
    "create_sharepoint_folder",
    # Recruitment/screening tools
    "parse_resume",
    "parse_job_description",
    "screen_candidate",
    "batch_screen_resumes",
    "get_candidate_profile",
    "list_candidates",
    "list_job_descriptions",
    "get_shortlisted_candidates",
    # Session management tools
    "get_session_dashboard",
    "clear_session_data",
    # Interview tools
    "generate_interview_questions",
    "export_question_set",
    "submit_candidate_answers",
    "evaluate_candidate_answers",
    "get_candidate_score",
    "list_question_sets",
    # Scoring tools
    "generate_scoring_report",
    "export_scoring_excel",
    "get_ranking_summary",
    "get_passing_score_thresholds",
    "generate_shortlist_report",
]
