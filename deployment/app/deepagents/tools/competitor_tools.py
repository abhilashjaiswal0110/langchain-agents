"""Competitive Intelligence Tools for Sales Intelligence Deep Agent.

Tools for competitive analysis, positioning, and objection handling.
"""

from langchain_core.tools import tool

# Competitive intelligence database
COMPETITORS_DB = {
    "accenture": {
        "name": "Accenture",
        "overview": "Global professional services company with strong consulting heritage.",
        "strengths": [
            "Strong brand recognition",
            "Deep industry expertise",
            "Large delivery capacity",
            "Strong C-suite relationships",
            "End-to-end transformation capabilities",
        ],
        "weaknesses": [
            "Premium pricing",
            "Can be slow to mobilize",
            "Less flexible on small deals",
            "High staff turnover",
            "Bureaucratic processes",
        ],
        "pricing_position": "Premium (15-25% above market)",
        "key_differentiators": [
            "Strategy + Implementation combined",
            "Industry-specific solutions",
            "Global delivery network",
        ],
        "common_objections": [
            "Too expensive for our budget",
            "Prefer a more specialized provider",
            "Concerned about attention to our size of project",
        ],
        "win_against_strategy": [
            "Emphasize agility and faster time-to-value",
            "Highlight dedicated account team vs. rotation",
            "Compete on total cost of ownership, not just rates",
            "Position industry-specific expertise and similar references",
        ],
        "typical_clients": "Fortune 500, Large Government",
        "service_areas": ["Consulting", "Technology", "Operations", "Strategy"],
    },
    "tcs": {
        "name": "Tata Consultancy Services (TCS)",
        "overview": "Indian multinational IT services and consulting company.",
        "strengths": [
            "Very competitive pricing",
            "Large talent pool",
            "Strong in application services",
            "Good offshore delivery",
            "Long-term client relationships",
        ],
        "weaknesses": [
            "Less consulting expertise",
            "Communication challenges",
            "Quality variability",
            "Limited innovation perception",
            "Onshore presence gaps",
        ],
        "pricing_position": "Value (20-30% below market)",
        "key_differentiators": [
            "Cost efficiency",
            "Scale and capacity",
            "Application modernization expertise",
        ],
        "common_objections": [
            "Need stronger onshore presence",
            "Want more strategic advisory",
            "Concerned about quality control",
        ],
        "win_against_strategy": [
            "Emphasize quality and innovation",
            "Highlight strategic consulting capabilities",
            "Show strong local presence and accountability",
            "Focus on total value, not just labor rates",
        ],
        "typical_clients": "Cost-conscious enterprises, Banks",
        "service_areas": ["IT Services", "BPO", "Consulting", "Digital"],
    },
    "infosys": {
        "name": "Infosys",
        "overview": "Global leader in next-generation digital services and consulting.",
        "strengths": [
            "Strong in digital transformation",
            "Good training and upskilling",
            "Competitive pricing",
            "Innovation labs and IP",
            "Agile delivery models",
        ],
        "weaknesses": [
            "Less brand recognition vs. Accenture",
            "Limited consulting depth",
            "Staff attrition",
            "Smaller onshore teams",
        ],
        "pricing_position": "Competitive (10-20% below market)",
        "key_differentiators": [
            "AI-first approach",
            "Proprietary platforms",
            "Design thinking methodology",
        ],
        "common_objections": [
            "Need more senior consultants",
            "Want local delivery team",
            "Concerned about staff turnover",
        ],
        "win_against_strategy": [
            "Showcase senior expertise and stability",
            "Emphasize local accountability",
            "Highlight industry-specific experience",
            "Demonstrate innovation with pragmatism",
        ],
        "typical_clients": "Mid to large enterprises",
        "service_areas": ["Digital", "Cloud", "Data Analytics", "Consulting"],
    },
    "deloitte": {
        "name": "Deloitte",
        "overview": "Big Four professional services firm with strong audit and consulting.",
        "strengths": [
            "Trusted advisor status",
            "Deep industry expertise",
            "C-suite access",
            "Regulatory expertise",
            "Risk and compliance strength",
        ],
        "weaknesses": [
            "Very high pricing",
            "Staff heavily leveraged",
            "Junior-heavy teams",
            "Less technical depth",
            "Slow decision making",
        ],
        "pricing_position": "Premium (20-35% above market)",
        "key_differentiators": [
            "Industry and regulatory expertise",
            "Brand trust",
            "Full-service capabilities",
        ],
        "common_objections": [
            "Too expensive",
            "Want more technical depth",
            "Concerned about junior staff doing the work",
        ],
        "win_against_strategy": [
            "Compete on delivery expertise and technical depth",
            "Highlight senior team involvement throughout",
            "Show better value for technical implementations",
            "Emphasize technology partnerships",
        ],
        "typical_clients": "Fortune 500, Regulated Industries",
        "service_areas": ["Consulting", "Risk", "Tax", "Audit", "Technology"],
    },
    "aws-ps": {
        "name": "AWS Professional Services",
        "overview": "Amazon's professional services arm for AWS implementations.",
        "strengths": [
            "Deep AWS expertise",
            "Access to product teams",
            "Often bundled with credits",
            "Latest technology access",
            "Strong certification program",
        ],
        "weaknesses": [
            "AWS-only perspective",
            "Limited business consulting",
            "Capacity constraints",
            "Less industry expertise",
            "Not vendor-agnostic",
        ],
        "pricing_position": "Premium for AWS work",
        "key_differentiators": [
            "Direct AWS access",
            "Cloud credits programs",
            "Early access to new services",
        ],
        "common_objections": [
            "Only focused on AWS, not our business",
            "Want multi-cloud strategy",
            "Need broader transformation support",
        ],
        "win_against_strategy": [
            "Position as vendor-agnostic advisor",
            "Emphasize business outcome focus vs. just tech",
            "Highlight multi-cloud and hybrid expertise",
            "Show industry-specific use cases",
        ],
        "typical_clients": "AWS-committed organizations",
        "service_areas": ["Cloud Migration", "Cloud Native", "DevOps", "Data"],
    },
}

OBJECTION_HANDLERS_DB = {
    "price_too_high": {
        "objection": "Your price is too high",
        "category": "Pricing",
        "responses": [
            "I understand budget is a key consideration. Let me walk you through the total value equation including risk reduction, faster time-to-value, and ongoing operational savings.",
            "Our pricing reflects the senior expertise and proven methodology that reduces project risk and rework. Would it help to see a TCO comparison?",
            "We can explore different engagement models - perhaps a phased approach or outcome-based pricing that better aligns with your budget cycle?",
        ],
        "evidence_points": [
            "Average project overrun with lower-cost providers: 25-40%",
            "Cost of rework from failed implementations",
            "Value of faster time-to-market",
        ],
    },
    "incumbent_relationship": {
        "objection": "We have an existing relationship with another vendor",
        "category": "Competition",
        "responses": [
            "That's valuable - a known partner reduces risk. We often complement existing relationships, perhaps starting with a specific workstream where we can add unique value?",
            "We respect long-term partnerships. Many of our best clients started with a small proof-of-concept that demonstrated our differentiated approach. Would that be worth exploring?",
            "What aspects of your current relationship are working well, and where might you want to see different results?",
        ],
        "evidence_points": [
            "Case studies of successful vendor transitions",
            "Specific capability gaps we can fill",
            "Low-risk pilot approach",
        ],
    },
    "not_right_time": {
        "objection": "It's not the right time / We need to wait",
        "category": "Timing",
        "responses": [
            "I understand timing is critical. What would need to happen for this to become a priority? Perhaps we can help build the business case.",
            "Many organizations find that waiting increases the ultimate cost. Would it help to quantify the cost of delay?",
            "We could start with a discovery phase now, so you're ready to move quickly when the time is right.",
        ],
        "evidence_points": [
            "Cost of delay analysis",
            "Competitive pressure examples",
            "Risk of technical debt accumulation",
        ],
    },
    "need_references": {
        "objection": "We need to see more references in our industry",
        "category": "Credibility",
        "responses": [
            "Absolutely - references are crucial. We have several clients in [industry] who would be happy to speak with you. Let me arrange a call.",
            "Beyond references, we can also provide a site visit or joint workshop with a similar client to see our approach firsthand.",
            "I can share detailed case studies and connect you with the delivery leads from those projects.",
        ],
        "evidence_points": [
            "Specific industry references",
            "Case study documentation",
            "Award and recognition proof points",
        ],
    },
    "team_concerns": {
        "objection": "Concerned about the team you'll assign",
        "category": "Resources",
        "responses": [
            "Team quality is paramount. We commit to naming key team members in our proposal, and you'll have interview rights for senior roles.",
            "Our delivery model includes named senior advisors who stay with the project throughout, not rotating junior resources.",
            "We can include contractual commitments around team continuity and notice periods for any changes.",
        ],
        "evidence_points": [
            "Team bios and CVs",
            "Continuity commitment",
            "Performance guarantees",
        ],
    },
}


@tool
def get_competitive_analysis(competitor_name: str) -> str:
    """Get detailed competitive intelligence on a specific competitor.

    Use this to understand competitor strengths, weaknesses, and how to win against them.

    Args:
        competitor_name: Name of the competitor (e.g., Accenture, TCS, Infosys, Deloitte, AWS-PS).

    Returns:
        Comprehensive competitive analysis.
    """
    # Find competitor by name (partial match)
    competitor = None
    for key, comp in COMPETITORS_DB.items():
        if competitor_name.lower() in key.lower() or competitor_name.lower() in comp["name"].lower():
            competitor = comp
            break

    if not competitor:
        available = ", ".join(c["name"] for c in COMPETITORS_DB.values())
        return f"Competitor '{competitor_name}' not found. Available: {available}"

    return f"""
**Competitive Analysis: {competitor["name"]}**

**Overview**
{competitor["overview"]}

**Pricing Position:** {competitor["pricing_position"]}
**Typical Clients:** {competitor["typical_clients"]}

**Strengths**
{chr(10).join("✓ " + s for s in competitor["strengths"])}

**Weaknesses**
{chr(10).join("✗ " + w for w in competitor["weaknesses"])}

**Key Differentiators**
{chr(10).join("• " + d for d in competitor["key_differentiators"])}

**Service Areas:** {", ".join(competitor["service_areas"])}

---

**How to Win Against {competitor["name"]}**

**Common Objections We Hear:**
{chr(10).join('• "' + o + '"' for o in competitor["common_objections"])}

**Win Strategies:**
{chr(10).join("→ " + s for s in competitor["win_against_strategy"])}
"""


@tool
def compare_solutions(
    our_solution: str,
    competitor_solutions: str,
    evaluation_criteria: str | None = None,
) -> str:
    """Generate a solution comparison matrix.

    Use this to create competitive positioning for proposals.

    Args:
        our_solution: Description of our proposed solution.
        competitor_solutions: Comma-separated list of competitor solutions to compare.
        evaluation_criteria: Comma-separated evaluation criteria (optional).

    Returns:
        Comparison matrix with competitive positioning.
    """
    competitors = [c.strip() for c in competitor_solutions.split(",")]

    # Default criteria if not provided
    if evaluation_criteria:
        criteria = [c.strip() for c in evaluation_criteria.split(",")]
    else:
        criteria = [
            "Technical Expertise",
            "Industry Experience",
            "Delivery Methodology",
            "Team Quality",
            "Pricing Value",
            "Innovation",
            "Support Model",
            "Risk Mitigation",
        ]

    output = ["**Solution Comparison Matrix**\n"]
    output.append(f"*Our Solution: {our_solution}*\n")

    # Create comparison table header
    header = "| Criteria | Atos |"
    separator = "|----------|------|"
    for comp in competitors:
        header += f" {comp} |"
        separator += "------|"

    output.append(header)
    output.append(separator)

    # Add criteria rows (simulated ratings)
    for criterion in criteria:
        row = f"| {criterion} | ⭐⭐⭐⭐ |"
        for _ in competitors:
            # Simulated competitor ratings (in production, pull from real data)
            row += " ⭐⭐⭐ |"
        output.append(row)

    output.append("\n**Atos Differentiators:**")
    output.append("- Vendor-agnostic approach with best-of-breed solutions")
    output.append("- Senior team commitment throughout project lifecycle")
    output.append("- Proven methodology with measurable outcomes")
    output.append("- Strong local presence combined with global capabilities")

    output.append("\n**Competitive Positioning Recommendations:**")
    for comp in competitors:
        comp_data = None
        for key, data in COMPETITORS_DB.items():
            if comp.lower() in key.lower() or comp.lower() in data["name"].lower():
                comp_data = data
                break
        if comp_data:
            output.append(f"\n*vs. {comp_data['name']}:*")
            if comp_data.get("win_against_strategy"):
                output.append(f"  → {comp_data['win_against_strategy'][0]}")

    return "\n".join(output)


@tool
def suggest_differentiators(
    opportunity_context: str,
    competitors: str | None = None,
) -> str:
    """Suggest key differentiators for a specific opportunity.

    Use this to identify the strongest positioning points for a deal.

    Args:
        opportunity_context: Description of the opportunity and customer needs.
        competitors: Known competitors in the deal (comma-separated).

    Returns:
        Prioritized list of differentiators to emphasize.
    """
    context_lower = opportunity_context.lower()

    differentiators = []

    # Context-based differentiator selection
    if any(kw in context_lower for kw in ["cloud", "migration", "azure", "aws"]):
        differentiators.append(
            {
                "theme": "Cloud Expertise",
                "message": "Multi-cloud expertise with 1000+ successful migrations",
                "proof_points": [
                    "Strategic partnerships with AWS, Azure, GCP",
                    "Certified architects across platforms",
                ],
            }
        )

    if any(kw in context_lower for kw in ["security", "compliance", "regulated", "hipaa", "pci"]):
        differentiators.append(
            {
                "theme": "Security & Compliance",
                "message": "Deep expertise in regulated industries with proven compliance frameworks",
                "proof_points": [
                    "SOC2, ISO27001 certified delivery centers",
                    "Industry-specific compliance accelerators",
                ],
            }
        )

    if any(kw in context_lower for kw in ["support", "managed", "24/7", "sla"]):
        differentiators.append(
            {
                "theme": "Managed Services Excellence",
                "message": "ITIL-certified delivery with industry-leading SLAs",
                "proof_points": ["99.9% SLA achievement", "Follow-the-sun support model"],
            }
        )

    if any(kw in context_lower for kw in ["ai", "ml", "data", "analytics"]):
        differentiators.append(
            {
                "theme": "AI & Innovation",
                "message": "Practical AI implementation with measurable business outcomes",
                "proof_points": ["AI Center of Excellence", "Industry-specific AI solutions"],
            }
        )

    # Always add these core differentiators
    differentiators.extend(
        [
            {
                "theme": "Senior Team Commitment",
                "message": "Named senior resources committed throughout the engagement",
                "proof_points": ["Contract commitment to team continuity", "Interview rights for key roles"],
            },
            {
                "theme": "Proven Delivery",
                "message": "Track record of on-time, on-budget delivery",
                "proof_points": ["95% project success rate", "Industry references available"],
            },
        ]
    )

    output = ["**Recommended Differentiators**\n"]
    output.append(f"*Based on: {opportunity_context[:100]}...*\n")

    for i, diff in enumerate(differentiators[:5], 1):
        output.append(f"\n**{i}. {diff['theme']}**")
        output.append(f"*Key Message:* {diff['message']}")
        output.append("*Proof Points:*")
        for point in diff["proof_points"]:
            output.append(f"  • {point}")

    if competitors:
        output.append(f"\n---\n*Consider competitor-specific positioning for: {competitors}*")
        output.append("*Use `get_competitive_analysis` for detailed win strategies.*")

    return "\n".join(output)


@tool
def get_objection_handler(objection_type: str) -> str:
    """Get recommended responses for common sales objections.

    Use this to prepare for or respond to customer objections.

    Args:
        objection_type: Type of objection (price, timing, competition, resources, credibility).

    Returns:
        Recommended responses and evidence points.
    """
    # Map common keywords to objection types
    objection_map = {
        "price": "price_too_high",
        "cost": "price_too_high",
        "expensive": "price_too_high",
        "budget": "price_too_high",
        "incumbent": "incumbent_relationship",
        "existing": "incumbent_relationship",
        "vendor": "incumbent_relationship",
        "competition": "incumbent_relationship",
        "timing": "not_right_time",
        "wait": "not_right_time",
        "later": "not_right_time",
        "reference": "need_references",
        "credibility": "need_references",
        "proof": "need_references",
        "team": "team_concerns",
        "resources": "team_concerns",
        "staff": "team_concerns",
    }

    # Find matching objection
    handler = None
    for keyword, handler_key in objection_map.items():
        if keyword in objection_type.lower():
            handler = OBJECTION_HANDLERS_DB.get(handler_key)
            break

    if not handler:
        available = list(set(objection_map.values()))
        return f"Objection type '{objection_type}' not found. Try: price, timing, competition, references, or team"

    output = [f'**Handling Objection: "{handler["objection"]}"**\n']
    output.append(f"*Category: {handler['category']}*\n")

    output.append("**Recommended Responses:**")
    for i, response in enumerate(handler["responses"], 1):
        output.append(f"\n{i}. {response}")

    output.append("\n**Evidence Points to Support:**")
    for point in handler["evidence_points"]:
        output.append(f"• {point}")

    output.append("\n---")
    output.append("*Tip: Always acknowledge the concern before responding.*")
    output.append("*Listen → Acknowledge → Respond → Confirm*")

    return "\n".join(output)
