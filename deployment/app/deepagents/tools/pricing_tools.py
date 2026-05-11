"""Pricing Tools for Sales Intelligence Deep Agent.

Tools for pricing calculations, margin analysis, and pricing optimization.
"""

from typing import Literal

from langchain_core.tools import tool

# Pricing models and rate cards
RATE_CARDS = {
    "consulting": {
        "name": "Consulting Services",
        "rates": {
            "Principal Consultant": {"day_rate": 3500, "cost": 1800},
            "Senior Consultant": {"day_rate": 2800, "cost": 1400},
            "Consultant": {"day_rate": 2000, "cost": 1000},
            "Associate Consultant": {"day_rate": 1500, "cost": 750},
            "Analyst": {"day_rate": 1200, "cost": 600},
        },
        "typical_margin": 45,
    },
    "managed_services": {
        "name": "Managed Services",
        "rates": {
            "Service Delivery Manager": {"monthly": 22000, "cost": 12000},
            "Senior Engineer": {"monthly": 15000, "cost": 8000},
            "Engineer": {"monthly": 12000, "cost": 6500},
            "L2 Support": {"monthly": 8000, "cost": 4500},
            "L1 Support": {"monthly": 5500, "cost": 3000},
        },
        "typical_margin": 40,
    },
    "cloud": {
        "name": "Cloud Services",
        "rates": {
            "Cloud Architect": {"day_rate": 3200, "cost": 1600},
            "Senior Cloud Engineer": {"day_rate": 2500, "cost": 1250},
            "Cloud Engineer": {"day_rate": 1800, "cost": 900},
            "DevOps Engineer": {"day_rate": 2200, "cost": 1100},
            "SRE": {"day_rate": 2400, "cost": 1200},
        },
        "typical_margin": 42,
    },
    "security": {
        "name": "Cybersecurity",
        "rates": {
            "Security Architect": {"day_rate": 3500, "cost": 1750},
            "Penetration Tester": {"day_rate": 2800, "cost": 1400},
            "Security Analyst": {"day_rate": 2000, "cost": 1000},
            "SOC Analyst L3": {"day_rate": 1800, "cost": 900},
            "SOC Analyst L2": {"day_rate": 1400, "cost": 700},
        },
        "typical_margin": 48,
    },
}

PRICING_MODELS = {
    "time_and_materials": {
        "name": "Time & Materials",
        "description": "Billing based on actual time spent at agreed rates",
        "best_for": ["Undefined scope", "Discovery phases", "Staff augmentation"],
        "risks": ["Budget uncertainty for client", "Scope creep", "Revenue variability"],
        "typical_discount": 0,
    },
    "fixed_price": {
        "name": "Fixed Price",
        "description": "Agreed price for defined deliverables",
        "best_for": ["Well-defined projects", "Product implementations", "Migrations"],
        "risks": ["Scope disputes", "Margin erosion on overruns", "Change order friction"],
        "typical_discount": 5,
    },
    "outcome_based": {
        "name": "Outcome-Based",
        "description": "Pricing linked to achieving specific business outcomes",
        "best_for": ["Transformation programs", "Cost reduction initiatives", "Revenue growth projects"],
        "risks": ["Outcome measurement complexity", "External factor dependency", "Longer payment cycles"],
        "typical_discount": 0,
    },
    "managed_service": {
        "name": "Managed Service (Monthly)",
        "description": "Fixed monthly fee for defined service scope",
        "best_for": ["Ongoing operations", "Support services", "Infrastructure management"],
        "risks": ["Scope expansion pressure", "Service level penalties", "Volume variability"],
        "typical_discount": 10,
    },
}


@tool
def calculate_pricing(
    service_category: Literal["consulting", "managed_services", "cloud", "security"],
    resources: str,
    duration_days: int | None = None,
    duration_months: int | None = None,
    target_margin: float | None = None,
) -> str:
    """Calculate pricing for a solution based on resource mix.

    Use this to generate pricing estimates for proposals.

    Args:
        service_category: Type of service (consulting, managed_services, cloud, security).
        resources: Resource requirements as 'Role:Count' pairs, comma-separated.
            Example: 'Senior Consultant:2, Consultant:3, Analyst:1'
        duration_days: Project duration in days (for project-based work).
        duration_months: Duration in months (for managed services).
        target_margin: Target gross margin percentage (optional, uses default if not specified).

    Returns:
        Detailed pricing breakdown with costs and margins.
    """
    rate_card = RATE_CARDS.get(service_category)
    if not rate_card:
        return f"Service category '{service_category}' not found. Available: {', '.join(RATE_CARDS.keys())}"

    # Parse resources
    resource_list = []
    for item in resources.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            role = parts[0].strip()
            count = int(parts[1].strip())
            resource_list.append({"role": role, "count": count})

    if not resource_list:
        return "No valid resources specified. Format: 'Role:Count, Role:Count'"

    # Calculate pricing
    margin_target = target_margin or rate_card["typical_margin"]
    is_monthly = service_category == "managed_services" or duration_months

    total_revenue = 0
    total_cost = 0
    breakdown = []

    for res in resource_list:
        role = res["role"]
        count = res["count"]

        # Find matching role (partial match)
        matching_role = None
        for rate_role, rates in rate_card["rates"].items():
            if role.lower() in rate_role.lower() or rate_role.lower() in role.lower():
                matching_role = rate_role
                role_rates = rates
                break

        if not matching_role:
            breakdown.append(f"⚠️ Role '{role}' not found in rate card - skipped")
            continue

        if is_monthly:
            duration = duration_months or 12
            if "monthly" in role_rates:
                revenue = role_rates["monthly"] * count * duration
                cost = role_rates["cost"] * count * duration
            else:
                # Convert day rate to monthly (20 working days)
                revenue = role_rates["day_rate"] * 20 * count * duration
                cost = role_rates["cost"] * 20 * count * duration
        else:
            duration = duration_days or 20
            if "day_rate" in role_rates:
                revenue = role_rates["day_rate"] * count * duration
                cost = role_rates["cost"] * count * duration
            else:
                # Convert monthly to days
                revenue = (role_rates["monthly"] / 20) * count * duration
                cost = (role_rates["cost"] / 20) * count * duration

        total_revenue += revenue
        total_cost += cost

        margin_pct = ((revenue - cost) / revenue) * 100 if revenue > 0 else 0
        breakdown.append(f"• {matching_role} x{count}: ${revenue:,.0f} (cost: ${cost:,.0f}, margin: {margin_pct:.1f}%)")

    actual_margin = ((total_revenue - total_cost) / total_revenue) * 100 if total_revenue > 0 else 0
    gross_profit = total_revenue - total_cost

    # Duration text
    if is_monthly:
        duration_text = f"{duration_months or 12} months"
    else:
        duration_text = f"{duration_days or 20} days"

    output = f"""
**Pricing Estimate: {rate_card["name"]}**

**Duration:** {duration_text}
**Target Margin:** {margin_target}%

**Resource Breakdown:**
{chr(10).join(breakdown)}

---

**Summary:**
| Metric | Value |
|--------|-------|
| Total Revenue | ${total_revenue:,.0f} |
| Total Cost | ${total_cost:,.0f} |
| Gross Profit | ${gross_profit:,.0f} |
| Gross Margin | {actual_margin:.1f}% |

**Margin Assessment:**
"""

    if actual_margin >= margin_target:
        output += f"✅ Margin ({actual_margin:.1f}%) meets target ({margin_target}%)"
    else:
        gap = margin_target - actual_margin
        output += f"⚠️ Margin ({actual_margin:.1f}%) is {gap:.1f}% below target ({margin_target}%)"
        output += "\n*Consider: Rate increases, resource mix optimization, or scope reduction*"

    return output


@tool
def analyze_margin(
    revenue: float,
    cost: float,
    deal_type: str | None = None,
) -> str:
    """Analyze margin and profitability for a deal.

    Use this to validate pricing decisions and compare to benchmarks.

    Args:
        revenue: Total contract revenue.
        cost: Total delivery cost.
        deal_type: Type of deal for benchmark comparison (optional).

    Returns:
        Margin analysis with benchmarks and recommendations.
    """
    if revenue <= 0:
        return "Revenue must be greater than zero."

    gross_profit = revenue - cost
    gross_margin = (gross_profit / revenue) * 100
    markup = ((revenue - cost) / cost) * 100 if cost > 0 else 0

    # Benchmark thresholds
    benchmarks = {
        "excellent": 50,
        "good": 40,
        "acceptable": 30,
        "concerning": 20,
    }

    # Determine rating
    if gross_margin >= benchmarks["excellent"]:
        rating = "🌟 Excellent"
        recommendation = "Strong margin. Consider strategic value-adds or contingency buffer."
    elif gross_margin >= benchmarks["good"]:
        rating = "✅ Good"
        recommendation = "Healthy margin. Maintain rate discipline during negotiations."
    elif gross_margin >= benchmarks["acceptable"]:
        rating = "⚠️ Acceptable"
        recommendation = "Monitor closely. Look for optimization opportunities or rate improvements."
    elif gross_margin >= benchmarks["concerning"]:
        rating = "🔴 Concerning"
        recommendation = "Below target. Consider scope reduction, rate increases, or resource mix changes."
    else:
        rating = "❌ Critical"
        recommendation = "Margin too low. Deal requires executive review before proceeding."

    output = f"""
**Margin Analysis**

**Financial Summary:**
| Metric | Value |
|--------|-------|
| Revenue | ${revenue:,.0f} |
| Cost | ${cost:,.0f} |
| Gross Profit | ${gross_profit:,.0f} |
| Gross Margin | {gross_margin:.1f}% |
| Markup on Cost | {markup:.1f}% |

**Rating:** {rating}

**Industry Benchmarks:**
- Excellent: ≥50%
- Good: ≥40%
- Acceptable: ≥30%
- Concerning: ≥20%
- Critical: <20%

**Recommendation:**
{recommendation}
"""

    if deal_type:
        output += f"\n*Deal Type: {deal_type}*"

    if gross_margin < 30:
        output += """

**Margin Improvement Strategies:**
1. Increase senior/junior resource ratio
2. Negotiate rate increases for specialized skills
3. Reduce project scope to essentials
4. Add value-priced deliverables
5. Extend timeline to reduce peak staffing
"""

    return output


@tool
def generate_pricing_options(
    base_revenue: float,
    base_cost: float,
    option_types: str | None = None,
) -> str:
    """Generate multiple pricing options for customer negotiation.

    Use this to create good/better/best pricing scenarios.

    Args:
        base_revenue: Base pricing option revenue.
        base_cost: Base delivery cost.
        option_types: Comma-separated option types (economy, standard, premium).

    Returns:
        Multiple pricing options with trade-offs.
    """
    if option_types:
        options = [o.strip().lower() for o in option_types.split(",")]
    else:
        options = ["economy", "standard", "premium"]

    base_margin = ((base_revenue - base_cost) / base_revenue) * 100 if base_revenue > 0 else 0

    pricing_options = []

    for option in options:
        if option == "economy":
            multiplier = 0.85
            scope_desc = "Core deliverables, reduced support, standard resources"
            features = ["Essential scope only", "Standard SLAs", "Email support", "Core team"]
        elif option == "standard":
            multiplier = 1.0
            scope_desc = "Full scope as proposed, standard team, normal support"
            features = ["Full scope", "Enhanced SLAs", "Phone + email support", "Named resources"]
        elif option == "premium":
            multiplier = 1.25
            scope_desc = "Extended scope, senior team, premium support, accelerated timeline"
            features = [
                "Extended scope",
                "Premium SLAs",
                "24/7 dedicated support",
                "Senior team",
                "Accelerated delivery",
            ]
        else:
            continue

        opt_revenue = base_revenue * multiplier
        # Cost doesn't scale linearly - economy saves less, premium costs more
        cost_multiplier = 0.90 if option == "economy" else (1.15 if option == "premium" else 1.0)
        opt_cost = base_cost * cost_multiplier
        opt_margin = ((opt_revenue - opt_cost) / opt_revenue) * 100 if opt_revenue > 0 else 0

        pricing_options.append(
            {
                "name": option.capitalize(),
                "revenue": opt_revenue,
                "cost": opt_cost,
                "margin": opt_margin,
                "description": scope_desc,
                "features": features,
            }
        )

    output = ["**Pricing Options**\n"]

    for opt in pricing_options:
        margin_indicator = "✅" if opt["margin"] >= 35 else ("⚠️" if opt["margin"] >= 25 else "🔴")

        output.append(f"""
### Option: {opt["name"]} {margin_indicator}
**Price:** ${opt["revenue"]:,.0f}
**Gross Margin:** {opt["margin"]:.1f}%

*{opt["description"]}*

**Includes:**
{chr(10).join("• " + f for f in opt["features"])}
""")

    output.append("""
---

**Negotiation Guidance:**
- Start with **Standard** as the anchor
- Use **Economy** to show flexibility on budget constraints
- Use **Premium** to upsell or create contrast
- Avoid discounting Standard - redirect to Economy option instead
""")

    return "\n".join(output)


@tool
def get_pricing_model_recommendation(
    project_description: str,
    deal_value: float,
    customer_preference: str | None = None,
) -> str:
    """Recommend the best pricing model for a project.

    Use this to determine optimal pricing approach for a deal.

    Args:
        project_description: Description of the project/engagement.
        deal_value: Estimated deal value.
        customer_preference: Customer's stated preference if known.

    Returns:
        Pricing model recommendation with rationale.
    """
    desc_lower = project_description.lower()

    recommendations = []

    # Analyze context and recommend
    if any(kw in desc_lower for kw in ["discovery", "assessment", "unclear", "exploratory"]):
        recommendations.append(
            {
                "model": "time_and_materials",
                "fit": "High",
                "reason": "Scope is undefined - T&M provides flexibility to discover and adapt",
            }
        )

    if any(kw in desc_lower for kw in ["migration", "implementation", "deploy", "fixed"]):
        recommendations.append(
            {
                "model": "fixed_price",
                "fit": "High",
                "reason": "Well-defined deliverables suited for fixed price commitment",
            }
        )

    if any(kw in desc_lower for kw in ["managed", "support", "ongoing", "operations"]):
        recommendations.append(
            {
                "model": "managed_service",
                "fit": "High",
                "reason": "Ongoing services best delivered as managed service contract",
            }
        )

    if any(kw in desc_lower for kw in ["transform", "outcome", "roi", "savings", "growth"]):
        recommendations.append(
            {
                "model": "outcome_based",
                "fit": "Medium",
                "reason": "Outcome focus could align pricing with customer value realization",
            }
        )

    # If no specific match, recommend based on deal size
    if not recommendations:
        if deal_value > 2000000:
            recommendations.append(
                {
                    "model": "fixed_price",
                    "fit": "Medium",
                    "reason": "Large deals typically benefit from fixed price certainty",
                }
            )
        else:
            recommendations.append(
                {
                    "model": "time_and_materials",
                    "fit": "Medium",
                    "reason": "Standard recommendation for typical engagements",
                }
            )

    output = ["**Pricing Model Recommendation**\n"]
    output.append(f"*Deal Value: ${deal_value:,.0f}*\n")

    for rec in recommendations:
        model = PRICING_MODELS.get(rec["model"], {})
        output.append(f"""
### {model.get("name", rec["model"])} - Fit: {rec["fit"]}

**Rationale:** {rec["reason"]}

**Description:** {model.get("description", "N/A")}

**Best For:**
{chr(10).join("• " + b for b in model.get("best_for", []))}

**Risks to Consider:**
{chr(10).join("• " + r for r in model.get("risks", []))}
""")

    if customer_preference:
        output.append(f"\n**Customer Preference:** {customer_preference}")
        output.append("*Consider customer preference in final recommendation, but advise on risks if misaligned.*")

    return "\n".join(output)
