"""Sales Analytics Tools for Sales Intelligence Deep Agent.

Tools for win probability, risk assessment, and deal analytics.
"""

from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

# Historical win/loss data for analysis
WIN_LOSS_DB = {
    "Cloud Services": {
        "total_deals": 45,
        "won": 28,
        "lost": 17,
        "win_rate": 62.2,
        "avg_deal_size": 1800000,
        "avg_sales_cycle_days": 95,
        "top_win_factors": ["Technical expertise", "Competitive pricing", "Reference customers"],
        "top_loss_factors": ["Price too high", "Incumbent advantage", "Missing certifications"],
        "stage_conversion": {
            "Qualification": 0.75,
            "Discovery": 0.65,
            "Proposal": 0.55,
            "Negotiation": 0.85,
        },
    },
    "Managed Services": {
        "total_deals": 38,
        "won": 27,
        "lost": 11,
        "win_rate": 71.1,
        "avg_deal_size": 4200000,
        "avg_sales_cycle_days": 145,
        "top_win_factors": ["Strong SLAs", "Transition methodology", "Industry references"],
        "top_loss_factors": ["Not lowest price", "Limited geographic coverage", "Scope concerns"],
        "stage_conversion": {
            "Qualification": 0.80,
            "Discovery": 0.70,
            "Proposal": 0.65,
            "Negotiation": 0.90,
        },
    },
    "Data & AI": {
        "total_deals": 22,
        "won": 11,
        "lost": 11,
        "win_rate": 50.0,
        "avg_deal_size": 950000,
        "avg_sales_cycle_days": 120,
        "top_win_factors": ["Proven use cases", "Data expertise", "Quick POC delivery"],
        "top_loss_factors": ["Lack of industry data experience", "Big 4 competition", "Budget constraints"],
        "stage_conversion": {
            "Qualification": 0.65,
            "Discovery": 0.55,
            "Proposal": 0.50,
            "Negotiation": 0.80,
        },
    },
    "Cybersecurity": {
        "total_deals": 30,
        "won": 19,
        "lost": 11,
        "win_rate": 63.3,
        "avg_deal_size": 550000,
        "avg_sales_cycle_days": 75,
        "top_win_factors": ["Certifications", "Rapid response capability", "SOC capabilities"],
        "top_loss_factors": ["Niche competitor expertise", "Pricing", "Team availability"],
        "stage_conversion": {
            "Qualification": 0.70,
            "Discovery": 0.60,
            "Proposal": 0.60,
            "Negotiation": 0.85,
        },
    },
}

RISK_FACTORS = {
    "deal_size_risk": {
        "high": "Deal size significantly above average - increased scrutiny expected",
        "medium": "Deal size within normal range",
        "low": "Smaller deal - faster decision process likely",
    },
    "competition_risk": {
        "high": "Strong incumbent or Big 4 competitor - differentiation critical",
        "medium": "Known competitors - standard competitive positioning needed",
        "low": "Limited competition or sole-source opportunity",
    },
    "timeline_risk": {
        "high": "Compressed timeline or urgent deadline - execution risk",
        "medium": "Standard timeline - normal delivery risk",
        "low": "Flexible timeline - low schedule risk",
    },
    "relationship_risk": {
        "high": "New customer with no prior relationship",
        "medium": "Some prior engagement or known contacts",
        "low": "Strong existing relationship and track record",
    },
    "technical_risk": {
        "high": "Complex requirements or new technology stack",
        "medium": "Standard technical requirements",
        "low": "Proven solution with existing templates",
    },
}


@tool
def calculate_win_probability(
    business_line: str,
    stage: str,
    deal_amount: float,
    has_champion: bool = False,
    has_reference: bool = False,
    is_incumbent: bool = False,
    competitors_count: int = 2,
) -> str:
    """Calculate win probability for an opportunity.

    Use this to assess deal likelihood and identify improvement areas.

    Args:
        business_line: Business line (Cloud Services, Managed Services, Data & AI, Cybersecurity).
        stage: Current stage (Qualification, Discovery, Proposal, Negotiation).
        deal_amount: Deal value.
        has_champion: Whether we have an internal champion at the customer.
        has_reference: Whether we have relevant reference customers.
        is_incumbent: Whether we are the incumbent provider.
        competitors_count: Number of known competitors.

    Returns:
        Win probability analysis with recommendations.
    """
    # Get baseline from historical data
    history = WIN_LOSS_DB.get(business_line)
    if not history:
        return f"Business line '{business_line}' not found. Available: {', '.join(WIN_LOSS_DB.keys())}"

    # Base probability from stage conversion
    base_prob = history["stage_conversion"].get(stage, 0.5)

    # Adjustments
    adjustments = []

    # Champion impact (+15%)
    if has_champion:
        base_prob += 0.15
        adjustments.append("✅ Internal champion identified (+15%)")
    else:
        adjustments.append("⚠️ No internal champion (-5%)")
        base_prob -= 0.05

    # Reference impact (+10%)
    if has_reference:
        base_prob += 0.10
        adjustments.append("✅ Relevant references available (+10%)")
    else:
        adjustments.append("⚠️ No direct references (-3%)")
        base_prob -= 0.03

    # Incumbent advantage
    if is_incumbent:
        base_prob += 0.15
        adjustments.append("✅ Incumbent advantage (+15%)")
    else:
        adjustments.append("➖ Not incumbent (baseline)")

    # Competition impact
    if competitors_count == 0:
        base_prob += 0.20
        adjustments.append("✅ No competition identified (+20%)")
    elif competitors_count == 1:
        adjustments.append("➖ One competitor (baseline)")
    elif competitors_count <= 3:
        base_prob -= 0.05
        adjustments.append("⚠️ Multiple competitors (-5%)")
    else:
        base_prob -= 0.15
        adjustments.append("🔴 Crowded competition (-15%)")

    # Deal size impact
    if deal_amount > history["avg_deal_size"] * 2:
        base_prob -= 0.10
        adjustments.append("⚠️ Deal significantly above average size (-10%)")
    elif deal_amount > history["avg_deal_size"]:
        base_prob -= 0.05
        adjustments.append("⚠️ Deal above average size (-5%)")

    # Cap probability
    final_prob = max(0.05, min(0.95, base_prob))
    prob_pct = final_prob * 100

    # Determine rating
    if prob_pct >= 75:
        rating = "🌟 Strong"
        color = "green"
    elif prob_pct >= 50:
        rating = "✅ Good"
        color = "yellow"
    elif prob_pct >= 30:
        rating = "⚠️ Moderate"
        color = "orange"
    else:
        rating = "🔴 At Risk"
        color = "red"

    output = f"""
**Win Probability Analysis**

**Opportunity:** {business_line} | Stage: {stage}
**Deal Value:** ${deal_amount:,.0f}

---

## Win Probability: {prob_pct:.0f}% {rating}

**Historical Baseline:**
- Business Line Win Rate: {history['win_rate']}%
- Stage Conversion: {history['stage_conversion'].get(stage, 50)*100:.0f}%

**Factors Applied:**
{chr(10).join(adjustments)}

---

**To Improve Win Probability:**
"""

    # Recommendations
    if not has_champion:
        output += "\n1. **Find a Champion** - Identify and develop an internal advocate"
    if not has_reference:
        output += "\n2. **Secure References** - Line up relevant customer references"
    if competitors_count > 2:
        output += "\n3. **Differentiate** - Sharpen competitive positioning"
    if prob_pct < 50:
        output += "\n4. **De-risk** - Consider pilot or phased approach"

    output += f"""

**Benchmark Data ({business_line}):**
- Average Deal Size: ${history['avg_deal_size']:,}
- Average Sales Cycle: {history['avg_sales_cycle_days']} days
- Top Win Factors: {', '.join(history['top_win_factors'])}
"""

    return output


@tool
def assess_deal_risk(
    deal_description: str,
    deal_amount: float,
    timeline_weeks: int,
    is_new_customer: bool = True,
    is_new_technology: bool = False,
    competitor_strength: Literal["low", "medium", "high"] = "medium",
) -> str:
    """Assess risks for a deal and recommend mitigations.

    Use this to identify and address deal risks proactively.

    Args:
        deal_description: Description of the opportunity.
        deal_amount: Deal value.
        timeline_weeks: Project timeline in weeks.
        is_new_customer: Whether this is a new customer.
        is_new_technology: Whether this involves new/unproven technology.
        competitor_strength: Strength of competition (low, medium, high).

    Returns:
        Risk assessment with mitigation strategies.
    """
    risks = []
    risk_score = 0

    # Deal size risk
    if deal_amount > 5000000:
        risks.append({
            "category": "Deal Size",
            "level": "High",
            "description": RISK_FACTORS["deal_size_risk"]["high"],
            "mitigation": "Involve executive sponsors, prepare detailed business case, consider phased approach",
        })
        risk_score += 3
    elif deal_amount > 2000000:
        risks.append({
            "category": "Deal Size",
            "level": "Medium",
            "description": RISK_FACTORS["deal_size_risk"]["medium"],
            "mitigation": "Standard approval process, competitive positioning",
        })
        risk_score += 2
    else:
        risks.append({
            "category": "Deal Size",
            "level": "Low",
            "description": RISK_FACTORS["deal_size_risk"]["low"],
            "mitigation": "Standard sales process",
        })
        risk_score += 1

    # Relationship risk
    if is_new_customer:
        risks.append({
            "category": "Relationship",
            "level": "High",
            "description": RISK_FACTORS["relationship_risk"]["high"],
            "mitigation": "Invest in relationship building, provide strong references, consider pilot",
        })
        risk_score += 3
    else:
        risks.append({
            "category": "Relationship",
            "level": "Low",
            "description": RISK_FACTORS["relationship_risk"]["low"],
            "mitigation": "Leverage existing relationship",
        })
        risk_score += 1

    # Technical risk
    if is_new_technology:
        risks.append({
            "category": "Technical",
            "level": "High",
            "description": RISK_FACTORS["technical_risk"]["high"],
            "mitigation": "Include POC phase, technical advisors, risk contingency in pricing",
        })
        risk_score += 3
    else:
        risks.append({
            "category": "Technical",
            "level": "Low",
            "description": RISK_FACTORS["technical_risk"]["low"],
            "mitigation": "Use proven methodologies and templates",
        })
        risk_score += 1

    # Competition risk
    comp_level = competitor_strength
    risks.append({
        "category": "Competition",
        "level": comp_level.capitalize(),
        "description": RISK_FACTORS["competition_risk"][comp_level],
        "mitigation": "Sharp competitive positioning, clear differentiators" if comp_level == "high" else "Standard competitive approach",
    })
    risk_score += {"low": 1, "medium": 2, "high": 3}[comp_level]

    # Timeline risk
    if timeline_weeks < 8:
        risks.append({
            "category": "Timeline",
            "level": "High",
            "description": RISK_FACTORS["timeline_risk"]["high"],
            "mitigation": "Experienced team, clear scope boundaries, change control process",
        })
        risk_score += 3
    elif timeline_weeks < 16:
        risks.append({
            "category": "Timeline",
            "level": "Medium",
            "description": RISK_FACTORS["timeline_risk"]["medium"],
            "mitigation": "Standard project management rigor",
        })
        risk_score += 2
    else:
        risks.append({
            "category": "Timeline",
            "level": "Low",
            "description": RISK_FACTORS["timeline_risk"]["low"],
            "mitigation": "Normal delivery planning",
        })
        risk_score += 1

    # Overall risk rating
    if risk_score >= 12:
        overall = "🔴 HIGH RISK"
        recommendation = "Executive review required. Consider risk mitigation strategies before proceeding."
    elif risk_score >= 8:
        overall = "🟡 MEDIUM RISK"
        recommendation = "Manageable with proper attention. Implement mitigations proactively."
    else:
        overall = "🟢 LOW RISK"
        recommendation = "Standard deal process appropriate. Monitor for changes."

    output = f"""
**Deal Risk Assessment**

**Deal:** {deal_description[:100]}...
**Value:** ${deal_amount:,.0f} | **Timeline:** {timeline_weeks} weeks

---

## Overall Risk: {overall}
*Risk Score: {risk_score}/15*

---

**Risk Breakdown:**
"""

    for risk in risks:
        level_icon = "🔴" if risk["level"] == "High" else ("🟡" if risk["level"] == "Medium" else "🟢")
        output += f"""
### {level_icon} {risk['category']} Risk: {risk['level']}
*{risk['description']}*
**Mitigation:** {risk['mitigation']}
"""

    output += f"""
---

**Recommendation:**
{recommendation}

**Next Steps:**
1. Review each risk category with sales team
2. Implement mitigations before proposal submission
3. Document risk acceptance for high-risk items
4. Plan for contingencies in pricing and timeline
"""

    return output


@tool
def get_similar_deals(
    business_line: str,
    deal_amount: float,
    outcome_filter: Literal["Won", "Lost", "All"] = "All",
    limit: int = 5,
) -> str:
    """Find similar historical deals for reference.

    Use this to learn from past wins and losses.

    Args:
        business_line: Business line to search.
        deal_amount: Deal size for comparison.
        outcome_filter: Filter by outcome (Won, Lost, All).
        limit: Maximum results.

    Returns:
        List of similar deals with insights.
    """
    # Simulated similar deals (in production, query from CRM)
    similar_deals = [
        {
            "name": "Cloud Migration - Manufacturing Inc",
            "business_line": "Cloud Services",
            "value": 1500000,
            "outcome": "Won",
            "factors": ["Strong POC", "Competitive pricing", "AWS expertise"],
            "lessons": "POC was decisive - customer valued hands-on demonstration",
        },
        {
            "name": "Cloud Transformation - Energy Co",
            "business_line": "Cloud Services",
            "value": 2200000,
            "outcome": "Lost",
            "factors": ["Incumbent advantage", "Price sensitivity", "Timeline concerns"],
            "lessons": "Started late in cycle - need earlier engagement",
        },
        {
            "name": "Managed Services - Insurance Group",
            "business_line": "Managed Services",
            "value": 5500000,
            "outcome": "Won",
            "factors": ["Transition methodology", "SLA flexibility", "Industry references"],
            "lessons": "References from same industry were crucial",
        },
        {
            "name": "IT Operations - Healthcare Network",
            "business_line": "Managed Services",
            "value": 3800000,
            "outcome": "Lost",
            "factors": ["Not lowest price", "Limited local presence", "Competitor relationships"],
            "lessons": "Need to invest in relationship earlier in cycle",
        },
        {
            "name": "AI Platform - Retail Chain",
            "business_line": "Data & AI",
            "value": 800000,
            "outcome": "Won",
            "factors": ["Quick POC", "Business case quality", "Executive sponsorship"],
            "lessons": "Executive alignment accelerated decision",
        },
    ]

    # Filter by business line and outcome
    filtered = [
        d for d in similar_deals
        if d["business_line"].lower() == business_line.lower()
        and (outcome_filter == "All" or d["outcome"] == outcome_filter)
    ][:limit]

    if not filtered:
        filtered = similar_deals[:limit]  # Show any deals if no match

    history = WIN_LOSS_DB.get(business_line, {})

    output = [f"**Similar Deals Analysis**\n"]
    output.append(f"*Business Line: {business_line} | Target Value: ${deal_amount:,.0f}*\n")

    for deal in filtered:
        outcome_icon = "✅" if deal["outcome"] == "Won" else "❌"
        output.append(f"""
### {outcome_icon} {deal['name']}
**Value:** ${deal['value']:,} | **Outcome:** {deal['outcome']}

**Key Factors:** {', '.join(deal['factors'])}

**Lesson Learned:** {deal['lessons']}
""")

    if history:
        output.append(f"""
---

**Business Line Insights ({business_line}):**
- Overall Win Rate: {history.get('win_rate', 'N/A')}%
- Avg Deal Size: ${history.get('avg_deal_size', 0):,}
- Top Win Factors: {', '.join(history.get('top_win_factors', [])[:3])}
- Top Loss Factors: {', '.join(history.get('top_loss_factors', [])[:3])}
""")

    return "\n".join(output)


@tool
def get_sales_performance_summary(
    time_period: Literal["quarter", "year"] = "quarter",
    business_line: str | None = None,
) -> str:
    """Get sales performance summary and KPIs.

    Use this for pipeline reviews and performance analysis.

    Args:
        time_period: Time period for analysis (quarter, year).
        business_line: Specific business line (optional, all if not specified).

    Returns:
        Performance summary with KPIs.
    """
    # Aggregate data (simulated)
    if business_line:
        lines = [business_line]
    else:
        lines = list(WIN_LOSS_DB.keys())

    total_deals = 0
    total_won = 0
    total_revenue = 0
    weighted_cycle = 0

    for line in lines:
        data = WIN_LOSS_DB.get(line, {})
        if data:
            total_deals += data.get("total_deals", 0)
            total_won += data.get("won", 0)
            total_revenue += data.get("won", 0) * data.get("avg_deal_size", 0)
            weighted_cycle += data.get("avg_sales_cycle_days", 0) * data.get("total_deals", 0)

    avg_cycle = weighted_cycle / total_deals if total_deals > 0 else 0
    win_rate = (total_won / total_deals * 100) if total_deals > 0 else 0

    output = f"""
**Sales Performance Summary**

**Period:** {time_period.capitalize()}
**Scope:** {business_line or 'All Business Lines'}

---

## Key Metrics

| KPI | Value | Target | Status |
|-----|-------|--------|--------|
| Total Deals | {total_deals} | 120 | {'✅' if total_deals >= 100 else '⚠️'} |
| Win Rate | {win_rate:.1f}% | 65% | {'✅' if win_rate >= 65 else '⚠️'} |
| Total Revenue | ${total_revenue:,.0f} | $100M | {'✅' if total_revenue >= 90000000 else '⚠️'} |
| Avg Sales Cycle | {avg_cycle:.0f} days | 100 days | {'✅' if avg_cycle <= 100 else '⚠️'} |

---

## Performance by Business Line

"""

    for line, data in WIN_LOSS_DB.items():
        if business_line and line != business_line:
            continue
        output += f"""
### {line}
- Deals: {data['total_deals']} (Won: {data['won']}, Lost: {data['lost']})
- Win Rate: {data['win_rate']}%
- Avg Deal: ${data['avg_deal_size']:,}
- Avg Cycle: {data['avg_sales_cycle_days']} days
"""

    output += """
---

**KPI Definitions:**
- *Win Rate*: Closed Won / (Closed Won + Closed Lost)
- *Proposal Cycle Time*: Days from proposal submission to decision
- *Deal Quality Score*: Composite of margin, strategic value, and customer fit
- *Rework Reduction*: Proposals needing revision before acceptance
"""

    return output
