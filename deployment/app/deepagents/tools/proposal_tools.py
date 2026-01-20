"""Proposal and RFP Tools for Sales Intelligence Deep Agent.

Tools for RFP/RFI response drafting, requirement extraction,
and proposal management.
"""

from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

# Simulated RFP templates and proposal repository
RFP_TEMPLATES_DB = {
    "TMPL-CLOUD-001": {
        "id": "TMPL-CLOUD-001",
        "name": "Cloud Migration Proposal Template",
        "category": "Cloud Services",
        "sections": [
            "Executive Summary",
            "Understanding of Requirements",
            "Technical Approach",
            "Migration Methodology",
            "Project Timeline",
            "Team Structure",
            "Risk Mitigation",
            "Pricing",
            "Case Studies",
            "Terms and Conditions",
        ],
        "win_rate": 68,
        "last_updated": "2024-01-10",
        "tags": ["cloud", "migration", "azure", "aws", "infrastructure"],
    },
    "TMPL-MS-001": {
        "id": "TMPL-MS-001",
        "name": "Managed Services Proposal Template",
        "category": "Managed Services",
        "sections": [
            "Executive Summary",
            "Service Scope",
            "Service Level Agreements",
            "Governance Model",
            "Tools and Technology",
            "Transition Plan",
            "Continuous Improvement",
            "Pricing Model",
            "References",
            "Contract Terms",
        ],
        "win_rate": 72,
        "last_updated": "2024-01-15",
        "tags": ["managed services", "itil", "support", "sla", "outsourcing"],
    },
    "TMPL-SEC-001": {
        "id": "TMPL-SEC-001",
        "name": "Cybersecurity Assessment Template",
        "category": "Cybersecurity",
        "sections": [
            "Executive Summary",
            "Assessment Scope",
            "Methodology",
            "Deliverables",
            "Compliance Mapping",
            "Timeline",
            "Team Credentials",
            "Pricing",
            "Confidentiality",
        ],
        "win_rate": 65,
        "last_updated": "2023-12-20",
        "tags": ["security", "assessment", "compliance", "pen testing", "audit"],
    },
    "TMPL-AI-001": {
        "id": "TMPL-AI-001",
        "name": "AI/ML Implementation Proposal Template",
        "category": "Data & AI",
        "sections": [
            "Executive Summary",
            "Business Objectives",
            "Data Requirements",
            "Solution Architecture",
            "ML Model Approach",
            "Implementation Phases",
            "Success Metrics",
            "Change Management",
            "Pricing",
            "Assumptions",
        ],
        "win_rate": 58,
        "last_updated": "2024-01-05",
        "tags": ["ai", "ml", "data science", "analytics", "automation"],
    },
}

PROPOSAL_SECTIONS_DB = {
    "exec-summary-cloud": {
        "section": "Executive Summary",
        "category": "Cloud Services",
        "template": """
## Executive Summary

[Company Name] is pleased to present this proposal for [Customer Name]'s cloud transformation initiative.

**The Challenge:**
[Describe customer's current state and pain points]

**Our Solution:**
We propose a comprehensive cloud migration strategy that will:
- Reduce infrastructure costs by [X]%
- Improve application availability to [X]%
- Enable scalability for future growth
- Modernize legacy applications

**Why [Company Name]:**
- [X] years of cloud expertise
- [X]+ successful migrations
- Strategic partnerships with [Cloud Provider]
- Proven methodology with zero-downtime migrations

**Investment Summary:**
- Total Investment: $[X]
- Timeline: [X] months
- Expected ROI: [X]% within [X] years
""",
    },
    "tech-approach-cloud": {
        "section": "Technical Approach",
        "category": "Cloud Services",
        "template": """
## Technical Approach

### Assessment Phase
- Infrastructure discovery and documentation
- Application dependency mapping
- Cloud readiness assessment
- TCO analysis and optimization recommendations

### Design Phase
- Target architecture design
- Landing zone configuration
- Security and compliance framework
- Network and connectivity design

### Migration Phase
- Phased migration approach (Wave planning)
- Lift-and-shift for compatible workloads
- Refactoring for cloud-native optimization
- Data migration with minimal downtime

### Optimization Phase
- Performance tuning
- Cost optimization
- Automation implementation
- Knowledge transfer and training
""",
    },
    "sla-managed-services": {
        "section": "Service Level Agreements",
        "category": "Managed Services",
        "template": """
## Service Level Agreements

### Availability SLAs
| Service Tier | Availability Target | Measurement Window |
|--------------|--------------------|--------------------|
| Critical     | 99.99%             | Monthly            |
| High         | 99.9%              | Monthly            |
| Standard     | 99.5%              | Monthly            |

### Response Time SLAs
| Priority | Response Time | Resolution Target |
|----------|--------------|-------------------|
| P1 - Critical | 15 minutes | 1 hour |
| P2 - High | 1 hour | 4 hours |
| P3 - Medium | 4 hours | 24 hours |
| P4 - Low | 8 hours | 72 hours |

### Service Credits
- 99.99% - 99.9%: 10% credit
- 99.9% - 99.0%: 25% credit
- Below 99.0%: 50% credit
""",
    },
}

PAST_PROPOSALS_DB = {
    "PROP-2023-045": {
        "id": "PROP-2023-045",
        "title": "Cloud Migration for Manufacturing Co",
        "customer": "Manufacturing Co",
        "value": 1800000,
        "outcome": "Won",
        "win_factors": ["Competitive pricing", "Strong team experience", "Flexible payment terms"],
        "category": "Cloud Services",
        "date": "2023-10-15",
    },
    "PROP-2023-052": {
        "id": "PROP-2023-052",
        "title": "Managed Services for LogiTech",
        "customer": "LogiTech Solutions",
        "value": 3500000,
        "outcome": "Won",
        "win_factors": ["24/7 support capability", "ITIL expertise", "Transition methodology"],
        "category": "Managed Services",
        "date": "2023-11-20",
    },
    "PROP-2023-061": {
        "id": "PROP-2023-061",
        "title": "Security Assessment for FinServ Inc",
        "customer": "FinServ Inc",
        "value": 450000,
        "outcome": "Lost",
        "loss_factors": ["Price too high", "Competitor had more compliance certifications"],
        "category": "Cybersecurity",
        "date": "2023-12-05",
    },
}


@tool
def search_rfp_templates(
    category: str | None = None,
    keywords: str | None = None,
    limit: int = 5,
) -> str:
    """Search for RFP/proposal templates.

    Use this to find relevant templates for drafting proposals.

    Args:
        category: Filter by category (Cloud Services, Managed Services, Cybersecurity, Data & AI).
        keywords: Search keywords in template name and tags.
        limit: Maximum number of results.

    Returns:
        List of matching templates.
    """
    results = []
    for tmpl_id, tmpl in RFP_TEMPLATES_DB.items():
        if category and tmpl["category"].lower() != category.lower():
            continue
        if keywords:
            kw_lower = keywords.lower()
            if kw_lower not in tmpl["name"].lower() and not any(kw_lower in tag for tag in tmpl["tags"]):
                continue
        results.append(tmpl)
        if len(results) >= limit:
            break

    if not results:
        return "No templates found matching the criteria."

    output = [f"**Found {len(results)} template(s):**\n"]
    for tmpl in results:
        output.append(f"""
**{tmpl['id']}** - {tmpl['name']}
- Category: {tmpl['category']} | Win Rate: {tmpl['win_rate']}%
- Sections: {', '.join(tmpl['sections'][:5])}...
- Tags: {', '.join(tmpl['tags'])}
- Last Updated: {tmpl['last_updated']}
""")
    return "\n".join(output)


@tool
def get_template_details(template_id: str) -> str:
    """Get detailed information about a proposal template.

    Args:
        template_id: The template ID (e.g., TMPL-CLOUD-001).

    Returns:
        Template details including all sections.
    """
    tmpl = RFP_TEMPLATES_DB.get(template_id.upper())
    if not tmpl:
        return f"Template {template_id} not found."

    sections_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(tmpl["sections"]))

    return f"""
**Template: {tmpl['name']}**

- ID: {tmpl['id']}
- Category: {tmpl['category']}
- Historical Win Rate: {tmpl['win_rate']}%
- Last Updated: {tmpl['last_updated']}

**Sections:**
{sections_list}

**Tags:** {', '.join(tmpl['tags'])}
"""


@tool
def extract_requirements(
    rfp_text: str,
    focus_areas: str | None = None,
) -> str:
    """Extract and categorize requirements from RFP/RFI text.

    Use this to analyze RFP documents and identify key requirements.

    Args:
        rfp_text: The RFP/RFI text to analyze.
        focus_areas: Specific areas to focus on (comma-separated).

    Returns:
        Categorized requirements extracted from the text.
    """
    # Simulated requirement extraction (in production, use NLP/LLM)
    # This provides a structured format for the agent to work with

    categories = {
        "Technical Requirements": [],
        "Compliance Requirements": [],
        "Timeline Requirements": [],
        "Support Requirements": [],
        "Pricing Requirements": [],
    }

    # Simple keyword-based extraction (placeholder for real NLP)
    text_lower = rfp_text.lower()

    if any(kw in text_lower for kw in ["cloud", "migration", "infrastructure", "server", "vm"]):
        categories["Technical Requirements"].append("Cloud/Infrastructure capabilities required")
    if any(kw in text_lower for kw in ["security", "compliance", "hipaa", "soc", "pci", "gdpr"]):
        categories["Compliance Requirements"].append("Security/Compliance certifications required")
    if any(kw in text_lower for kw in ["timeline", "deadline", "due date", "month", "week"]):
        categories["Timeline Requirements"].append("Specific timeline constraints identified")
    if any(kw in text_lower for kw in ["support", "sla", "24/7", "helpdesk", "response"]):
        categories["Support Requirements"].append("Support/SLA requirements specified")
    if any(kw in text_lower for kw in ["price", "cost", "budget", "payment", "invoice"]):
        categories["Pricing Requirements"].append("Pricing/Budget constraints mentioned")

    output = ["**Extracted Requirements**\n"]
    output.append(f"*Analyzed {len(rfp_text)} characters of RFP text*\n")

    for cat, reqs in categories.items():
        if reqs:
            output.append(f"\n**{cat}:**")
            for req in reqs:
                output.append(f"- {req}")

    if focus_areas:
        output.append(f"\n*Focus areas requested: {focus_areas}*")

    output.append("\n\n**Recommendation:** Use the `draft_proposal_section` tool to create responses for each requirement category.")

    return "\n".join(output)


@tool
def draft_proposal_section(
    section_type: Literal[
        "Executive Summary",
        "Technical Approach",
        "Service Level Agreements",
        "Pricing Summary",
        "Risk Mitigation",
        "Team Structure",
        "Timeline",
    ],
    category: str,
    customer_name: str,
    key_points: str | None = None,
) -> str:
    """Draft a proposal section using templates and best practices.

    Use this to generate initial draft content for proposal sections.

    Args:
        section_type: Type of section to draft.
        category: Business category (Cloud Services, Managed Services, etc.).
        customer_name: Name of the customer.
        key_points: Key points to emphasize (comma-separated).

    Returns:
        Draft section content.
    """
    # Try to find a matching template section
    section_key = f"{section_type.lower().replace(' ', '-')}-{category.lower().replace(' ', '-')[:5]}"

    for key, section in PROPOSAL_SECTIONS_DB.items():
        if section["section"] == section_type and section["category"].lower() == category.lower():
            template = section["template"]
            # Personalize the template
            template = template.replace("[Customer Name]", customer_name)
            template = template.replace("[Company Name]", "Atos")

            result = f"**Draft: {section_type}**\n"
            result += f"*Category: {category} | Customer: {customer_name}*\n"
            result += template

            if key_points:
                result += f"\n\n**Key Points to Emphasize:**\n"
                for point in key_points.split(","):
                    result += f"- {point.strip()}\n"

            result += "\n\n*Note: This is a draft template. Customize with specific customer requirements and solution details.*"
            return result

    # Generic section if no template found
    return f"""
**Draft: {section_type}**
*Category: {category} | Customer: {customer_name}*

[Template not found for this specific combination]

**Suggested Content:**
1. Introduction to the section topic
2. How our approach addresses {customer_name}'s needs
3. Key differentiators and value propositions
4. Specific deliverables and outcomes
5. Next steps or call to action

**Key Points:** {key_points or 'None specified'}

*Tip: Search for templates using `search_rfp_templates` for more options.*
"""


@tool
def generate_executive_summary(
    opportunity_name: str,
    customer_name: str,
    solution_overview: str,
    value_proposition: str,
    investment_amount: int,
    timeline_months: int,
) -> str:
    """Generate an executive summary for a proposal.

    Use this to create a compelling executive summary tailored to the opportunity.

    Args:
        opportunity_name: Name of the opportunity/project.
        customer_name: Customer name.
        solution_overview: Brief description of the proposed solution.
        value_proposition: Key value and benefits.
        investment_amount: Total investment amount.
        timeline_months: Project timeline in months.

    Returns:
        Draft executive summary.
    """
    return f"""
# Executive Summary

## Proposal for {opportunity_name}

**Prepared for:** {customer_name}
**Date:** {datetime.now().strftime("%B %d, %Y")}

---

### The Opportunity

{customer_name} is seeking a strategic partner to deliver transformative results. Atos is pleased to present our proposal to address your needs through a comprehensive solution designed for your specific requirements.

### Our Solution

{solution_overview}

### Value Proposition

{value_proposition}

### Key Benefits

- **Reduced Costs:** Optimize operations and reduce total cost of ownership
- **Improved Performance:** Enhance efficiency and service delivery
- **Strategic Alignment:** Support your long-term business objectives
- **Risk Mitigation:** Proven methodology and experienced team

### Investment Summary

| Element | Details |
|---------|---------|
| Total Investment | ${investment_amount:,} |
| Timeline | {timeline_months} months |
| Payment Terms | Net 30, milestone-based |

### Why Atos

With 110,000+ employees across 73 countries, Atos brings:
- Deep industry expertise
- Proven delivery methodology
- Strategic technology partnerships
- Commitment to sustainable digital transformation

---

*We look forward to partnering with {customer_name} on this important initiative.*

**Contact:** [Sales Representative]
**Email:** [email]
**Phone:** [phone]
"""


@tool
def search_past_proposals(
    category: str | None = None,
    outcome: Literal["Won", "Lost", "All"] | None = "All",
    min_value: int | None = None,
    limit: int = 5,
) -> str:
    """Search past proposals for reference and insights.

    Use this to find similar past proposals and understand win/loss factors.

    Args:
        category: Filter by category.
        outcome: Filter by outcome (Won, Lost, All).
        min_value: Minimum proposal value.
        limit: Maximum results.

    Returns:
        List of matching past proposals with win/loss insights.
    """
    results = []
    for prop_id, prop in PAST_PROPOSALS_DB.items():
        if category and prop["category"].lower() != category.lower():
            continue
        if outcome and outcome != "All" and prop["outcome"] != outcome:
            continue
        if min_value and prop["value"] < min_value:
            continue
        results.append(prop)
        if len(results) >= limit:
            break

    if not results:
        return "No past proposals found matching the criteria."

    output = [f"**Found {len(results)} past proposal(s):**\n"]
    for prop in results:
        factors_key = "win_factors" if prop["outcome"] == "Won" else "loss_factors"
        factors = prop.get(factors_key, ["No factors recorded"])

        output.append(f"""
**{prop['id']}** - {prop['title']}
- Customer: {prop['customer']} | Value: ${prop['value']:,}
- Category: {prop['category']} | Outcome: **{prop['outcome']}**
- Date: {prop['date']}
- {'Win' if prop['outcome'] == 'Won' else 'Loss'} Factors: {', '.join(factors)}
""")

    return "\n".join(output)
