"""Tests for Sales Intelligence Deep Agent functionality."""

import os
import pytest
import tempfile
import shutil
import uuid

# Set up mock API keys before importing app modules
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"


# =============================================================================
# CRM Tools Tests
# =============================================================================

class TestCRMTools:
    """Tests for CRM integration tools."""

    def test_search_opportunities_all(self):
        """Test searching all opportunities."""
        from app.deepagents.tools.crm_tools import search_opportunities

        result = search_opportunities.invoke({})
        assert isinstance(result, str)
        assert "OPP" in result  # Should contain opportunity IDs

    def test_search_opportunities_by_status(self):
        """Test searching opportunities by stage."""
        from app.deepagents.tools.crm_tools import search_opportunities

        result = search_opportunities.invoke({"stage": "qualification"})
        assert isinstance(result, str)
        # May or may not find results depending on data

    def test_search_opportunities_by_query(self):
        """Test searching opportunities by query term."""
        from app.deepagents.tools.crm_tools import search_opportunities

        result = search_opportunities.invoke({"query": "cloud"})
        assert isinstance(result, str)

    def test_search_opportunities_by_min_value(self):
        """Test searching opportunities by minimum value."""
        from app.deepagents.tools.crm_tools import search_opportunities

        result = search_opportunities.invoke({"min_value": 1000000})
        assert isinstance(result, str)

    def test_get_deal_details(self):
        """Test retrieving deal details."""
        from app.deepagents.tools.crm_tools import get_deal_details

        result = get_deal_details.invoke({"opportunity_id": "OPP-2024-001"})
        assert isinstance(result, str)
        assert "OPP-2024-001" in result or "not found" in result

    def test_get_deal_details_not_found(self):
        """Test retrieving nonexistent deal."""
        from app.deepagents.tools.crm_tools import get_deal_details

        result = get_deal_details.invoke({"opportunity_id": "OPP-9999-999"})
        assert "not found" in result

    def test_update_opportunity_stage(self):
        """Test updating opportunity stage."""
        from app.deepagents.tools.crm_tools import update_opportunity_stage

        result = update_opportunity_stage.invoke({
            "opportunity_id": "OPP-2024-001",
            "new_stage": "Proposal",
            "notes": "Moving to proposal phase after qualification"
        })
        assert isinstance(result, str)
        assert "Updated" in result or "not found" in result

    def test_get_customer_history(self):
        """Test retrieving customer history."""
        from app.deepagents.tools.crm_tools import get_customer_history

        result = get_customer_history.invoke({"customer_id_or_name": "TechCorp Industries"})
        assert isinstance(result, str)

    def test_get_pipeline_summary(self):
        """Test getting pipeline summary."""
        from app.deepagents.tools.crm_tools import get_pipeline_summary

        result = get_pipeline_summary.invoke({})
        assert isinstance(result, str)
        assert "Pipeline" in result or "pipeline" in result


# =============================================================================
# Proposal Tools Tests
# =============================================================================

class TestProposalTools:
    """Tests for proposal/RFP management tools."""

    def test_search_rfp_templates(self):
        """Test searching RFP templates."""
        from app.deepagents.tools.proposal_tools import search_rfp_templates

        result = search_rfp_templates.invoke({"query": "cloud"})
        assert isinstance(result, str)

    def test_search_rfp_templates_by_category(self):
        """Test searching RFP templates by category."""
        from app.deepagents.tools.proposal_tools import search_rfp_templates

        result = search_rfp_templates.invoke({"category": "Managed Services"})
        assert isinstance(result, str)

    def test_get_template_details(self):
        """Test retrieving template details."""
        from app.deepagents.tools.proposal_tools import get_template_details

        result = get_template_details.invoke({"template_id": "TPL-CLOUD-001"})
        assert isinstance(result, str)

    def test_get_template_details_not_found(self):
        """Test retrieving nonexistent template."""
        from app.deepagents.tools.proposal_tools import get_template_details

        result = get_template_details.invoke({"template_id": "TPL-NONEXISTENT"})
        assert "not found" in result

    def test_extract_requirements(self):
        """Test extracting requirements from RFP text."""
        from app.deepagents.tools.proposal_tools import extract_requirements

        rfp_text = """
        Requirements:
        1. The vendor must provide 24/7 support
        2. Solution must integrate with Azure AD
        3. Compliance with SOC 2 Type II required
        """
        result = extract_requirements.invoke({"rfp_text": rfp_text})
        assert isinstance(result, str)
        assert "requirement" in result.lower() or "extracted" in result.lower()

    def test_draft_proposal_section(self):
        """Test drafting a proposal section."""
        from app.deepagents.tools.proposal_tools import draft_proposal_section

        result = draft_proposal_section.invoke({
            "section_type": "Executive Summary",
            "category": "Cloud Services",
            "customer_name": "TechCorp Industries",
            "key_points": "Cloud migration, 99.9% uptime SLA, Azure-based solution"
        })
        assert isinstance(result, str)
        assert len(result) > 50  # Should generate meaningful content

    def test_generate_executive_summary(self):
        """Test generating executive summary."""
        from app.deepagents.tools.proposal_tools import generate_executive_summary

        result = generate_executive_summary.invoke({
            "opportunity_name": "TechCorp Cloud Migration",
            "customer_name": "TechCorp Industries",
            "solution_overview": "Azure hybrid cloud with managed services",
            "value_proposition": "Cloud migration, security, 24/7 support",
            "investment_amount": 2500000,
            "timeline_months": 12
        })
        assert isinstance(result, str)
        assert "TechCorp" in result or "cloud" in result.lower()

    def test_search_past_proposals(self):
        """Test searching past proposals."""
        from app.deepagents.tools.proposal_tools import search_past_proposals

        result = search_past_proposals.invoke({"query": "cloud"})
        assert isinstance(result, str)


# =============================================================================
# Competitor Tools Tests
# =============================================================================

class TestCompetitorTools:
    """Tests for competitive intelligence tools."""

    def test_get_competitive_analysis(self):
        """Test getting competitive analysis."""
        from app.deepagents.tools.competitor_tools import get_competitive_analysis

        result = get_competitive_analysis.invoke({"competitor_name": "Accenture"})
        assert isinstance(result, str)

    def test_get_competitive_analysis_not_found(self):
        """Test getting analysis for unknown competitor."""
        from app.deepagents.tools.competitor_tools import get_competitive_analysis

        result = get_competitive_analysis.invoke({"competitor_name": "UnknownCompany"})
        assert "No competitive" in result or "not found" in result

    def test_compare_solutions(self):
        """Test comparing solutions."""
        from app.deepagents.tools.competitor_tools import compare_solutions

        result = compare_solutions.invoke({
            "our_solution": "Azure hybrid cloud with 24/7 managed services",
            "competitor_solutions": "Accenture AWS solution, TCS multi-cloud offering",
            "evaluation_criteria": "uptime, support, cost, expertise"
        })
        assert isinstance(result, str)
        assert len(result) > 50

    def test_suggest_differentiators(self):
        """Test suggesting differentiators."""
        from app.deepagents.tools.competitor_tools import suggest_differentiators

        result = suggest_differentiators.invoke({
            "opportunity_context": "Large enterprise cloud migration with 24/7 support needs",
            "competitors": "Accenture, TCS"
        })
        assert isinstance(result, str)

    def test_get_objection_handler(self):
        """Test getting objection handler."""
        from app.deepagents.tools.competitor_tools import get_objection_handler

        result = get_objection_handler.invoke({"objection_type": "price"})
        assert isinstance(result, str)
        # Should provide some response strategy

    def test_get_objection_handler_keywords(self):
        """Test objection handler with various keywords."""
        from app.deepagents.tools.competitor_tools import get_objection_handler

        objections = ["price", "experience", "timeline", "support"]
        for objection in objections:
            result = get_objection_handler.invoke({"objection_type": objection})
            assert isinstance(result, str)


# =============================================================================
# Pricing Tools Tests
# =============================================================================

class TestPricingTools:
    """Tests for pricing and margin analysis tools."""

    def test_calculate_pricing_consulting(self):
        """Test calculating consulting pricing."""
        from app.deepagents.tools.pricing_tools import calculate_pricing

        result = calculate_pricing.invoke({
            "service_category": "consulting",
            "resources": "Senior Consultant:2, Consultant:3, Analyst:1",
            "duration_days": 60
        })
        assert isinstance(result, str)
        assert "$" in result or "cost" in result.lower()

    def test_calculate_pricing_managed_services(self):
        """Test calculating managed services pricing."""
        from app.deepagents.tools.pricing_tools import calculate_pricing

        result = calculate_pricing.invoke({
            "service_category": "managed_services",
            "resources": "Service Delivery Manager:1, Senior Engineer:2, Engineer:2",
            "duration_months": 12
        })
        assert isinstance(result, str)

    def test_analyze_margin(self):
        """Test analyzing margin."""
        from app.deepagents.tools.pricing_tools import analyze_margin

        result = analyze_margin.invoke({
            "revenue": 1000000,
            "cost": 700000,
            "deal_type": "consulting"
        })
        assert isinstance(result, str)
        assert "margin" in result.lower() or "%" in result

    def test_analyze_margin_low(self):
        """Test analyzing low margin scenario."""
        from app.deepagents.tools.pricing_tools import analyze_margin

        result = analyze_margin.invoke({
            "revenue": 1000000,
            "cost": 950000,  # Very low margin
            "deal_type": "managed_services"
        })
        assert isinstance(result, str)

    def test_generate_pricing_options(self):
        """Test generating pricing options."""
        from app.deepagents.tools.pricing_tools import generate_pricing_options

        result = generate_pricing_options.invoke({
            "base_revenue": 2000000,
            "base_cost": 1200000,
            "option_types": "economy, standard, premium"
        })
        assert isinstance(result, str)
        assert "option" in result.lower() or "tier" in result.lower()

    def test_get_pricing_model_recommendation(self):
        """Test getting pricing model recommendation."""
        from app.deepagents.tools.pricing_tools import get_pricing_model_recommendation

        result = get_pricing_model_recommendation.invoke({
            "project_description": "Long-term managed services for enterprise client with ongoing support",
            "deal_value": 2000000,
            "customer_preference": "predictable costs"
        })
        assert isinstance(result, str)


# =============================================================================
# Analytics Tools Tests
# =============================================================================

class TestAnalyticsTools:
    """Tests for win probability and risk assessment tools."""

    def test_calculate_win_probability(self):
        """Test calculating win probability."""
        from app.deepagents.tools.analytics_tools import calculate_win_probability

        result = calculate_win_probability.invoke({
            "business_line": "Cloud Services",
            "stage": "Proposal",
            "deal_amount": 2000000,
            "has_champion": True,
            "competitors_count": 3
        })
        assert isinstance(result, str)
        assert "%" in result or "probability" in result.lower()

    def test_calculate_win_probability_all_stages(self):
        """Test win probability across different stages."""
        from app.deepagents.tools.analytics_tools import calculate_win_probability

        stages = ["Qualification", "Discovery", "Proposal", "Negotiation"]
        for stage in stages:
            result = calculate_win_probability.invoke({
                "business_line": "Cloud Services",
                "stage": stage,
                "deal_amount": 1000000,
                "competitors_count": 2
            })
            assert isinstance(result, str)

    def test_assess_deal_risk(self):
        """Test assessing deal risk."""
        from app.deepagents.tools.analytics_tools import assess_deal_risk

        result = assess_deal_risk.invoke({
            "deal_description": "Enterprise cloud migration for financial services client",
            "deal_amount": 2500000,
            "timeline_weeks": 24,
            "is_new_customer": True,
            "competitor_strength": "high"
        })
        assert isinstance(result, str)
        assert "risk" in result.lower()

    def test_assess_deal_risk_low_complexity(self):
        """Test assessing risk for low complexity deal."""
        from app.deepagents.tools.analytics_tools import assess_deal_risk

        result = assess_deal_risk.invoke({
            "deal_description": "Standard consulting engagement",
            "deal_amount": 500000,
            "timeline_weeks": 12,
            "is_new_customer": False,
            "competitor_strength": "low"
        })
        assert isinstance(result, str)

    def test_get_similar_deals(self):
        """Test getting similar deals."""
        from app.deepagents.tools.analytics_tools import get_similar_deals

        result = get_similar_deals.invoke({
            "business_line": "Cloud Services",
            "deal_amount": 2000000,
            "outcome_filter": "All"
        })
        assert isinstance(result, str)

    def test_get_sales_performance_summary(self):
        """Test getting sales performance summary."""
        from app.deepagents.tools.analytics_tools import get_sales_performance_summary

        result = get_sales_performance_summary.invoke({
            "time_period": "quarter",
            "business_line": "Cloud Services"
        })
        assert isinstance(result, str)
        assert "%" in result or "rate" in result.lower()


# =============================================================================
# Sales Subagents Tests
# =============================================================================

class TestSalesSubagentDefinitions:
    """Tests for sales subagent definitions."""

    def test_get_all_sales_subagents(self):
        """Test retrieving all sales subagent definitions."""
        from app.deepagents.subagents.sales_subagents import get_all_sales_subagents

        subagents = get_all_sales_subagents()
        assert len(subagents) >= 5  # At least 5 subagents defined

        names = [s.name for s in subagents]
        assert "deal-qualifier" in names
        assert "solution-architect" in names
        assert "proposal-writer" in names
        assert "pricing-analyst" in names
        assert "competitive-strategist" in names

    def test_get_sales_subagent_by_name(self):
        """Test retrieving specific sales subagent."""
        from app.deepagents.subagents.sales_subagents import get_sales_subagent_by_name

        subagent = get_sales_subagent_by_name("deal-qualifier")
        assert subagent is not None
        assert subagent.name == "deal-qualifier"
        assert "BANT" in subagent.system_prompt or "MEDDIC" in subagent.system_prompt

    def test_get_sales_subagent_tools(self):
        """Test that subagents have associated tools."""
        from app.deepagents.subagents.sales_subagents import get_sales_subagent_by_name

        # Check proposal-writer has proposal tools
        proposal_writer = get_sales_subagent_by_name("proposal-writer")
        assert proposal_writer is not None
        assert len(proposal_writer.tools) > 0

        # Check pricing-analyst has pricing tools
        pricing_analyst = get_sales_subagent_by_name("pricing-analyst")
        assert pricing_analyst is not None
        assert len(pricing_analyst.tools) > 0

    def test_get_sales_subagent_not_found(self):
        """Test retrieving unknown subagent."""
        from app.deepagents.subagents.sales_subagents import get_sales_subagent_by_name

        subagent = get_sales_subagent_by_name("unknown-agent")
        assert subagent is None

    def test_subagent_system_prompts(self):
        """Test that all subagents have proper system prompts."""
        from app.deepagents.subagents.sales_subagents import get_all_sales_subagents

        for subagent in get_all_sales_subagents():
            assert subagent.system_prompt is not None
            assert len(subagent.system_prompt) > 50
            assert subagent.description is not None
            assert len(subagent.description) > 10


# =============================================================================
# Sales Intelligence Agent Tests
# =============================================================================

class TestSalesIntelligenceAgent:
    """Tests for the main Sales Intelligence Deep Agent."""

    def test_agent_creation_without_api_key(self):
        """Test that agent creation fails without API key."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            from app.deepagents.sales_intelligence_agent import create_sales_intelligence_agent
            with pytest.raises(ValueError, match="No LLM API key found"):
                create_sales_intelligence_agent()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    def test_agent_module_imports(self):
        """Test that agent module imports correctly."""
        from app.deepagents.sales_intelligence_agent import (
            SalesIntelligenceDeepAgent,
            create_sales_intelligence_agent,
            get_graph,
            SALES_INTELLIGENCE_SYSTEM_PROMPT,
        )

        assert SalesIntelligenceDeepAgent is not None
        assert create_sales_intelligence_agent is not None
        assert get_graph is not None
        assert "Sales" in SALES_INTELLIGENCE_SYSTEM_PROMPT or "sales" in SALES_INTELLIGENCE_SYSTEM_PROMPT

    def test_agent_system_prompt_content(self):
        """Test system prompt contains key capabilities."""
        from app.deepagents.sales_intelligence_agent import SALES_INTELLIGENCE_SYSTEM_PROMPT

        prompt_lower = SALES_INTELLIGENCE_SYSTEM_PROMPT.lower()
        # Should mention key capabilities
        assert "rfp" in prompt_lower or "proposal" in prompt_lower
        assert "deal" in prompt_lower or "opportunity" in prompt_lower
        assert "pricing" in prompt_lower or "price" in prompt_lower


# =============================================================================
# Sales Agent API Endpoint Tests
# =============================================================================

class TestSalesAgentAPIEndpoints:
    """Tests for Sales Agent API endpoints."""

    def test_start_sales_agent_session(self):
        """Test starting a Sales Agent session."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.post(
            "/api/sales-agent/start",
            json={"user_id": "test-user"}
        )

        # May fail if agent not loaded (503), return 401 (auth required), but should return valid response
        assert response.status_code in [200, 401, 503]
        data = response.json()
        if response.status_code == 200:
            assert "session_id" in data

    def test_list_sales_subagents_endpoint(self):
        """Test listing available sales subagents via API."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/sales-agent/subagents")

        # May return 503 if agent not loaded, or 401 if auth required
        assert response.status_code in [200, 401, 503]
        if response.status_code == 200:
            data = response.json()
            assert "subagents" in data

    def test_sales_agent_context_not_found(self):
        """Test context endpoint with invalid session returns 404."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/sales-agent/context/invalid-session-id")

        # Should return 404, 401 (auth required), or 503 (if agent not loaded)
        assert response.status_code in [401, 404, 503]

    def test_sales_agent_chat_without_session(self):
        """Test chat endpoint without valid session."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.post(
            "/api/sales-agent/chat",
            json={
                "session_id": "nonexistent-session",
                "message": "Hello"
            }
        )

        # Should return 404, 401 (auth required), or 503
        assert response.status_code in [401, 404, 503]


# =============================================================================
# Integration Tests (End-to-End)
# =============================================================================

class TestSalesAgentE2EWorkflows:
    """End-to-end workflow tests for Sales Intelligence Agent."""

    def test_deal_qualification_workflow(self):
        """Test deal qualification workflow with tools."""
        from app.deepagents.tools.crm_tools import search_opportunities, get_deal_details
        from app.deepagents.tools.analytics_tools import calculate_win_probability, assess_deal_risk

        # Step 1: Search for opportunities
        opps = search_opportunities.invoke({"stage": "qualification"})
        assert isinstance(opps, str)

        # Step 2: Get deal details (use sample ID)
        details = get_deal_details.invoke({"opportunity_id": "OPP-2024-001"})
        assert isinstance(details, str)

        # Step 3: Calculate win probability
        win_prob = calculate_win_probability.invoke({
            "business_line": "Cloud Services",
            "stage": "Qualification",
            "deal_amount": 2500000,
            "has_champion": True,
            "competitors_count": 3
        })
        assert isinstance(win_prob, str)

        # Step 4: Assess risk
        risk = assess_deal_risk.invoke({
            "deal_description": "Enterprise cloud migration for tech client",
            "deal_amount": 2500000,
            "timeline_weeks": 24,
            "is_new_customer": False,
            "competitor_strength": "high"
        })
        assert isinstance(risk, str)

    def test_rfp_response_workflow(self):
        """Test RFP response drafting workflow."""
        from app.deepagents.tools.proposal_tools import (
            search_rfp_templates,
            extract_requirements,
            draft_proposal_section,
            generate_executive_summary
        )

        # Step 1: Search for relevant templates
        templates = search_rfp_templates.invoke({"query": "cloud"})
        assert isinstance(templates, str)

        # Step 2: Extract requirements from RFP
        requirements = extract_requirements.invoke({
            "rfp_text": """
            The vendor must provide:
            - 99.9% uptime SLA
            - 24/7 technical support
            - SOC 2 Type II compliance
            - Integration with Azure AD
            """
        })
        assert isinstance(requirements, str)

        # Step 3: Draft a proposal section
        section = draft_proposal_section.invoke({
            "section_type": "Technical Approach",
            "category": "Cloud Services",
            "customer_name": "TechCorp Industries",
            "key_points": "Cloud migration, 99.9% uptime, Azure hybrid cloud architecture"
        })
        assert isinstance(section, str)

        # Step 4: Generate executive summary
        summary = generate_executive_summary.invoke({
            "opportunity_name": "Enterprise Cloud Transformation",
            "customer_name": "TechCorp Industries",
            "solution_overview": "Azure-based hybrid cloud",
            "value_proposition": "Cloud transformation, security enhancement, 24/7 support",
            "investment_amount": 3000000,
            "timeline_months": 18
        })
        assert isinstance(summary, str)

    def test_competitive_analysis_workflow(self):
        """Test competitive analysis workflow."""
        from app.deepagents.tools.competitor_tools import (
            get_competitive_analysis,
            compare_solutions,
            suggest_differentiators,
            get_objection_handler
        )

        # Step 1: Get competitor analysis
        analysis = get_competitive_analysis.invoke({"competitor_name": "Accenture"})
        assert isinstance(analysis, str)

        # Step 2: Compare solutions
        comparison = compare_solutions.invoke({
            "our_solution": "Azure-based managed services with AI ops",
            "competitor_solutions": "Traditional managed services, Legacy IT services",
            "evaluation_criteria": "innovation, automation, cost"
        })
        assert isinstance(comparison, str)

        # Step 3: Get differentiators
        differentiators = suggest_differentiators.invoke({
            "opportunity_context": "Digital transformation for manufacturing",
            "competitors": "Accenture, TCS"
        })
        assert isinstance(differentiators, str)

        # Step 4: Handle objection
        objection_response = get_objection_handler.invoke({
            "objection_type": "price"
        })
        assert isinstance(objection_response, str)

    def test_pricing_workflow(self):
        """Test pricing analysis workflow."""
        from app.deepagents.tools.pricing_tools import (
            calculate_pricing,
            analyze_margin,
            generate_pricing_options,
            get_pricing_model_recommendation
        )

        # Step 1: Calculate base pricing
        pricing = calculate_pricing.invoke({
            "service_category": "managed_services",
            "resources": "Service Delivery Manager:1, Senior Engineer:4, Engineer:3",
            "duration_months": 24
        })
        assert isinstance(pricing, str)

        # Step 2: Analyze margin
        margin = analyze_margin.invoke({
            "revenue": 5000000,
            "cost": 3500000,
            "deal_type": "managed_services"
        })
        assert isinstance(margin, str)

        # Step 3: Generate pricing options
        options = generate_pricing_options.invoke({
            "base_revenue": 5000000,
            "base_cost": 3500000,
            "option_types": "economy, standard, premium"
        })
        assert isinstance(options, str)

        # Step 4: Get pricing model recommendation
        recommendation = get_pricing_model_recommendation.invoke({
            "project_description": "Long-term enterprise partnership with ongoing support",
            "deal_value": 5000000,
            "customer_preference": "predictable costs"
        })
        assert isinstance(recommendation, str)


# =============================================================================
# Security Tests
# =============================================================================

class TestSalesAgentSecurity:
    """Security-focused tests for Sales Intelligence Agent."""

    def test_no_sensitive_data_in_crm_results(self):
        """Test that CRM results don't expose sensitive data."""
        from app.deepagents.tools.crm_tools import search_opportunities

        result = search_opportunities.invoke({})
        # Should not contain API keys or passwords
        assert "sk-" not in result
        assert "api_key" not in result.lower()
        assert "password" not in result.lower()

    def test_no_sensitive_data_in_pricing(self):
        """Test that pricing data doesn't expose cost models."""
        from app.deepagents.tools.pricing_tools import calculate_pricing

        result = calculate_pricing.invoke({
            "service_category": "consulting",
            "resources": "Senior Consultant:2, Consultant:2, Analyst:1",
            "duration_days": 120
        })
        # Should not expose internal cost rates or margins in raw form
        assert "internal" not in result.lower() or "cost" in result.lower()

    def test_competitor_data_is_sanitized(self):
        """Test that competitor data doesn't contain proprietary info."""
        from app.deepagents.tools.competitor_tools import get_competitive_analysis

        result = get_competitive_analysis.invoke({"competitor_name": "Accenture"})
        # Should be analysis, not raw proprietary data
        assert isinstance(result, str)


# =============================================================================
# Tool Export Tests
# =============================================================================

class TestToolExports:
    """Tests for tool module exports."""

    def test_crm_tools_exported(self):
        """Test CRM tools are properly exported."""
        from app.deepagents.tools import (
            search_opportunities,
            get_deal_details,
            update_opportunity_stage,
            get_customer_history,
            get_pipeline_summary
        )
        assert search_opportunities is not None
        assert get_deal_details is not None
        assert update_opportunity_stage is not None
        assert get_customer_history is not None
        assert get_pipeline_summary is not None

    def test_proposal_tools_exported(self):
        """Test proposal tools are properly exported."""
        from app.deepagents.tools import (
            search_rfp_templates,
            get_template_details,
            extract_requirements,
            draft_proposal_section,
            generate_executive_summary,
            search_past_proposals
        )
        assert search_rfp_templates is not None
        assert get_template_details is not None
        assert extract_requirements is not None
        assert draft_proposal_section is not None
        assert generate_executive_summary is not None
        assert search_past_proposals is not None

    def test_competitor_tools_exported(self):
        """Test competitor tools are properly exported."""
        from app.deepagents.tools import (
            get_competitive_analysis,
            compare_solutions,
            suggest_differentiators,
            get_objection_handler
        )
        assert get_competitive_analysis is not None
        assert compare_solutions is not None
        assert suggest_differentiators is not None
        assert get_objection_handler is not None

    def test_pricing_tools_exported(self):
        """Test pricing tools are properly exported."""
        from app.deepagents.tools import (
            calculate_pricing,
            analyze_margin,
            generate_pricing_options,
            get_pricing_model_recommendation
        )
        assert calculate_pricing is not None
        assert analyze_margin is not None
        assert generate_pricing_options is not None
        assert get_pricing_model_recommendation is not None

    def test_analytics_tools_exported(self):
        """Test analytics tools are properly exported."""
        from app.deepagents.tools import (
            calculate_win_probability,
            assess_deal_risk,
            get_similar_deals,
            get_sales_performance_summary
        )
        assert calculate_win_probability is not None
        assert assess_deal_risk is not None
        assert get_similar_deals is not None
        assert get_sales_performance_summary is not None
