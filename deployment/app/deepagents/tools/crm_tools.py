"""CRM Tools for Sales Intelligence Deep Agent.

Tools for customer relationship management, opportunity tracking,
and customer history analysis.
"""

from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

# Simulated CRM database (in production, integrate with Salesforce/HubSpot/Dynamics)
OPPORTUNITIES_DB = {
    "OPP-2024-001": {
        "id": "OPP-2024-001",
        "name": "Enterprise Cloud Migration - TechCorp",
        "customer": "TechCorp Industries",
        "customer_id": "CUST-001",
        "stage": "Proposal",
        "amount": 2500000,
        "probability": 65,
        "close_date": "2024-03-31",
        "owner": "Sarah Johnson",
        "business_line": "Cloud Services",
        "competitors": ["AWS Professional Services", "Accenture"],
        "created_date": "2024-01-15",
        "last_activity": "2024-01-20",
        "description": "Full cloud migration from on-premise to Azure. 500+ VMs, 50 applications.",
        "next_steps": "Submit technical proposal by Feb 1",
        "contacts": ["John Smith (CTO)", "Maria Garcia (VP Engineering)"],
        "requirements": ["Azure migration", "Zero downtime", "24/7 support", "Training included"],
    },
    "OPP-2024-002": {
        "id": "OPP-2024-002",
        "name": "Managed Services Contract - GlobalBank",
        "customer": "GlobalBank Financial",
        "customer_id": "CUST-002",
        "stage": "Negotiation",
        "amount": 5000000,
        "probability": 75,
        "close_date": "2024-02-28",
        "owner": "Mike Chen",
        "business_line": "Managed Services",
        "competitors": ["Infosys", "TCS"],
        "created_date": "2023-11-01",
        "last_activity": "2024-01-22",
        "description": "5-year managed services contract covering IT infrastructure, security, and helpdesk.",
        "next_steps": "Final pricing negotiation meeting Jan 25",
        "contacts": ["Robert Williams (CIO)", "Lisa Anderson (Procurement)"],
        "requirements": ["ITIL compliance", "SOC2 certified", "99.9% SLA", "On-shore support"],
    },
    "OPP-2024-003": {
        "id": "OPP-2024-003",
        "name": "AI/ML Platform Implementation - HealthFirst",
        "customer": "HealthFirst Medical",
        "customer_id": "CUST-003",
        "stage": "Qualification",
        "amount": 1200000,
        "probability": 40,
        "close_date": "2024-06-30",
        "owner": "Sarah Johnson",
        "business_line": "Data & AI",
        "competitors": ["Deloitte", "PwC"],
        "created_date": "2024-01-10",
        "last_activity": "2024-01-18",
        "description": "Implementation of ML platform for predictive patient outcomes and resource optimization.",
        "next_steps": "Discovery workshop scheduled Feb 5",
        "contacts": ["Dr. James Lee (CMO)", "Emma Wilson (IT Director)"],
        "requirements": ["HIPAA compliance", "On-premise option", "Integration with Epic EHR"],
    },
    "OPP-2024-004": {
        "id": "OPP-2024-004",
        "name": "Cybersecurity Assessment - RetailMax",
        "customer": "RetailMax Corp",
        "customer_id": "CUST-004",
        "stage": "Discovery",
        "amount": 350000,
        "probability": 55,
        "close_date": "2024-04-15",
        "owner": "Mike Chen",
        "business_line": "Cybersecurity",
        "competitors": ["CrowdStrike", "Palo Alto Networks"],
        "created_date": "2024-01-05",
        "last_activity": "2024-01-21",
        "description": "Comprehensive security assessment and remediation roadmap following recent breach attempt.",
        "next_steps": "Security questionnaire response due Jan 28",
        "contacts": ["Tom Brown (CISO)", "Nancy Davis (CEO)"],
        "requirements": ["PCI-DSS compliance", "Red team testing", "24/7 SOC", "Incident response"],
    },
}

CUSTOMERS_DB = {
    "CUST-001": {
        "id": "CUST-001",
        "name": "TechCorp Industries",
        "industry": "Technology",
        "size": "Enterprise",
        "employees": 5000,
        "revenue": "$2B",
        "location": "San Francisco, CA",
        "relationship_start": "2020-03-15",
        "account_manager": "Sarah Johnson",
        "total_contract_value": 4500000,
        "active_contracts": 3,
        "satisfaction_score": 4.2,
        "notes": "Strategic account. Strong relationship with CTO. Expansion opportunity in cloud.",
        "past_deals": [
            {"name": "Network Refresh", "value": 800000, "year": 2021, "outcome": "Won"},
            {"name": "Security Audit", "value": 150000, "year": 2022, "outcome": "Won"},
            {"name": "SAP Migration", "value": 3500000, "year": 2023, "outcome": "Lost to Accenture"},
        ],
    },
    "CUST-002": {
        "id": "CUST-002",
        "name": "GlobalBank Financial",
        "industry": "Financial Services",
        "size": "Enterprise",
        "employees": 25000,
        "revenue": "$15B",
        "location": "New York, NY",
        "relationship_start": "2018-01-10",
        "account_manager": "Mike Chen",
        "total_contract_value": 12000000,
        "active_contracts": 5,
        "satisfaction_score": 4.5,
        "notes": "Top 5 account. Very compliance-focused. Long procurement cycles.",
        "past_deals": [
            {"name": "Data Center Consolidation", "value": 5000000, "year": 2020, "outcome": "Won"},
            {"name": "Cloud Hosting", "value": 2000000, "year": 2021, "outcome": "Won"},
            {"name": "DevOps Transformation", "value": 1500000, "year": 2022, "outcome": "Won"},
        ],
    },
    "CUST-003": {
        "id": "CUST-003",
        "name": "HealthFirst Medical",
        "industry": "Healthcare",
        "size": "Mid-Market",
        "employees": 3000,
        "revenue": "$800M",
        "location": "Boston, MA",
        "relationship_start": "2022-06-01",
        "account_manager": "Sarah Johnson",
        "total_contract_value": 500000,
        "active_contracts": 1,
        "satisfaction_score": 3.8,
        "notes": "Growing account. Very focused on HIPAA compliance. Budget constrained.",
        "past_deals": [
            {"name": "Infrastructure Assessment", "value": 100000, "year": 2022, "outcome": "Won"},
            {"name": "Telemedicine Platform", "value": 600000, "year": 2023, "outcome": "Lost to Deloitte"},
        ],
    },
    "CUST-004": {
        "id": "CUST-004",
        "name": "RetailMax Corp",
        "industry": "Retail",
        "size": "Enterprise",
        "employees": 15000,
        "revenue": "$5B",
        "location": "Chicago, IL",
        "relationship_start": "2023-09-01",
        "account_manager": "Mike Chen",
        "total_contract_value": 0,
        "active_contracts": 0,
        "satisfaction_score": None,
        "notes": "New prospect. Came to us after security incident. Fast decision maker.",
        "past_deals": [],
    },
}


def _is_live_mode() -> bool:
    """Check if running in live CRM mode."""
    import os
    return bool(os.getenv("CRM_API_KEY"))


@tool
def search_opportunities(
    customer: str | None = None,
    stage: str | None = None,
    business_line: str | None = None,
    owner: str | None = None,
    min_amount: int | None = None,
    max_amount: int | None = None,
    limit: int = 10,
) -> str:
    """Search for sales opportunities in the CRM.

    Use this to find deals matching specific criteria for pipeline analysis
    or to identify opportunities requiring attention.

    Args:
        customer: Filter by customer name (partial match).
        stage: Filter by stage (Qualification, Discovery, Proposal, Negotiation, Closed Won, Closed Lost).
        business_line: Filter by business line (Cloud Services, Managed Services, Data & AI, Cybersecurity).
        owner: Filter by opportunity owner.
        min_amount: Minimum deal amount.
        max_amount: Maximum deal amount.
        limit: Maximum number of results.

    Returns:
        Formatted list of matching opportunities.
    """
    if _is_live_mode():
        return "Live CRM integration not configured. Using simulation mode."

    # Simulation mode
    results = []
    for opp_id, opp in OPPORTUNITIES_DB.items():
        if customer and customer.lower() not in opp["customer"].lower():
            continue
        if stage and opp["stage"].lower() != stage.lower():
            continue
        if business_line and opp["business_line"].lower() != business_line.lower():
            continue
        if owner and owner.lower() not in opp["owner"].lower():
            continue
        if min_amount and opp["amount"] < min_amount:
            continue
        if max_amount and opp["amount"] > max_amount:
            continue
        results.append(opp)
        if len(results) >= limit:
            break

    if not results:
        return "No opportunities found matching the criteria."

    output = [f"**Found {len(results)} opportunity(s):**\n"]
    for opp in results:
        output.append(f"""
**{opp['id']}** - {opp['name']}
- Customer: {opp['customer']} | Amount: ${opp['amount']:,}
- Stage: {opp['stage']} | Probability: {opp['probability']}%
- Business Line: {opp['business_line']} | Owner: {opp['owner']}
- Close Date: {opp['close_date']}
- Competitors: {', '.join(opp['competitors'])}
""")
    return "\n".join(output)


@tool
def get_deal_details(opportunity_id: str) -> str:
    """Get comprehensive details about a specific opportunity.

    Use this for deep-dive analysis of a deal, including requirements,
    contacts, and next steps.

    Args:
        opportunity_id: The opportunity ID (e.g., OPP-2024-001).

    Returns:
        Detailed opportunity information including requirements and timeline.
    """
    if _is_live_mode():
        return "Live CRM integration not configured. Using simulation mode."

    opp = OPPORTUNITIES_DB.get(opportunity_id.upper())
    if not opp:
        return f"Opportunity {opportunity_id} not found."

    return f"""
**Opportunity Details: {opp['id']}**

**Overview**
- Name: {opp['name']}
- Customer: {opp['customer']}
- Amount: ${opp['amount']:,}
- Stage: {opp['stage']} | Probability: {opp['probability']}%
- Close Date: {opp['close_date']}
- Owner: {opp['owner']}
- Business Line: {opp['business_line']}

**Description**
{opp['description']}

**Key Requirements**
{chr(10).join('- ' + req for req in opp['requirements'])}

**Key Contacts**
{chr(10).join('- ' + contact for contact in opp['contacts'])}

**Competition**
{chr(10).join('- ' + comp for comp in opp['competitors'])}

**Next Steps**
{opp['next_steps']}

**Timeline**
- Created: {opp['created_date']}
- Last Activity: {opp['last_activity']}
"""


@tool
def update_opportunity_stage(
    opportunity_id: str,
    new_stage: Literal["Qualification", "Discovery", "Proposal", "Negotiation", "Closed Won", "Closed Lost"],
    notes: str | None = None,
) -> str:
    """Update the stage of an opportunity.

    Use this when deal progresses through the pipeline.

    Args:
        opportunity_id: The opportunity ID.
        new_stage: New stage for the opportunity.
        notes: Notes about the stage change.

    Returns:
        Confirmation of the update.
    """
    if _is_live_mode():
        return "Live CRM integration not configured. Using simulation mode."

    opp = OPPORTUNITIES_DB.get(opportunity_id.upper())
    if not opp:
        return f"Opportunity {opportunity_id} not found."

    old_stage = opp["stage"]
    opp["stage"] = new_stage
    opp["last_activity"] = datetime.now().strftime("%Y-%m-%d")

    result = f"""
**Stage Updated Successfully**
- Opportunity: {opp['id']} - {opp['name']}
- Previous Stage: {old_stage}
- New Stage: {new_stage}
"""
    if notes:
        result += f"- Notes: {notes}\n"

    return result


@tool
def get_customer_history(customer_id_or_name: str) -> str:
    """Get comprehensive history and context for a customer.

    Use this to understand customer relationship, past deals, and preferences
    before engaging on a new opportunity.

    Args:
        customer_id_or_name: Customer ID (CUST-xxx) or customer name.

    Returns:
        Customer profile including history and insights.
    """
    if _is_live_mode():
        return "Live CRM integration not configured. Using simulation mode."

    # Find customer by ID or name
    customer = None
    for cust_id, cust in CUSTOMERS_DB.items():
        if cust_id == customer_id_or_name.upper() or customer_id_or_name.lower() in cust["name"].lower():
            customer = cust
            break

    if not customer:
        return f"Customer '{customer_id_or_name}' not found."

    # Get active opportunities
    active_opps = [
        opp for opp in OPPORTUNITIES_DB.values()
        if opp["customer_id"] == customer["id"]
    ]

    past_deals_text = "\n".join(
        f"- {d['name']} ({d['year']}): ${d['value']:,} - {d['outcome']}"
        for d in customer["past_deals"]
    ) if customer["past_deals"] else "No past deals on record."

    active_opps_text = "\n".join(
        f"- {opp['id']}: {opp['name']} (${opp['amount']:,}) - {opp['stage']}"
        for opp in active_opps
    ) if active_opps else "No active opportunities."

    satisfaction = f"{customer['satisfaction_score']}/5" if customer["satisfaction_score"] else "Not rated"

    return f"""
**Customer Profile: {customer['name']}**

**Company Information**
- ID: {customer['id']}
- Industry: {customer['industry']}
- Size: {customer['size']} ({customer['employees']:,} employees)
- Revenue: {customer['revenue']}
- Location: {customer['location']}

**Relationship Summary**
- Account Manager: {customer['account_manager']}
- Relationship Since: {customer['relationship_start']}
- Total Contract Value: ${customer['total_contract_value']:,}
- Active Contracts: {customer['active_contracts']}
- Satisfaction Score: {satisfaction}

**Account Notes**
{customer['notes']}

**Past Deals**
{past_deals_text}

**Active Opportunities**
{active_opps_text}
"""


@tool
def get_pipeline_summary(
    owner: str | None = None,
    business_line: str | None = None,
) -> str:
    """Get a summary of the sales pipeline.

    Use this for pipeline analysis and forecasting.

    Args:
        owner: Filter by opportunity owner.
        business_line: Filter by business line.

    Returns:
        Pipeline summary with totals by stage.
    """
    if _is_live_mode():
        return "Live CRM integration not configured. Using simulation mode."

    # Filter opportunities
    opps = list(OPPORTUNITIES_DB.values())
    if owner:
        opps = [o for o in opps if owner.lower() in o["owner"].lower()]
    if business_line:
        opps = [o for o in opps if business_line.lower() == o["business_line"].lower()]

    if not opps:
        return "No opportunities found matching criteria."

    # Aggregate by stage
    stages = {}
    for opp in opps:
        stage = opp["stage"]
        if stage not in stages:
            stages[stage] = {"count": 0, "value": 0, "weighted": 0}
        stages[stage]["count"] += 1
        stages[stage]["value"] += opp["amount"]
        stages[stage]["weighted"] += opp["amount"] * (opp["probability"] / 100)

    total_value = sum(s["value"] for s in stages.values())
    total_weighted = sum(s["weighted"] for s in stages.values())

    output = ["**Pipeline Summary**\n"]

    stage_order = ["Qualification", "Discovery", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    for stage in stage_order:
        if stage in stages:
            s = stages[stage]
            output.append(f"**{stage}**: {s['count']} deals | ${s['value']:,.0f} | Weighted: ${s['weighted']:,.0f}")

    output.append(f"\n**Totals**")
    output.append(f"- Total Pipeline: ${total_value:,.0f}")
    output.append(f"- Weighted Forecast: ${total_weighted:,.0f}")
    output.append(f"- Total Deals: {len(opps)}")

    return "\n".join(output)
